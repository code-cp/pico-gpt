use anyhow::{Context, Result};
use fancy_regex::Regex;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use crate::ext::HashMapExt;
use crate::utils::{bpe_ranks_from_path, serde_json_from_path, StringPair};

pub type Token = String;
pub type TokenId = u64;

pub struct Encoder {
    /// disallow merging from separate words
    /// to prevent dog, dog!, dog? to be separate tokens
    /// eg dog! with be separated as dog, !
    word_re: Regex,
    /// byte encode -> encode
    /// then byte decode -> decode  
    byte_to_char: HashMap<u8, char>,
    char_to_byte: HashMap<char, u8>,
    token_to_id: HashMap<Token, TokenId>,
    id_to_token: HashMap<TokenId, Token>,
    /// priority of BPE merges
    bpe_ranks: HashMap<StringPair, usize>,
    tokenize_cache: HashMap<String, Vec<TokenId>>,
}

impl Encoder {
    pub fn from_dir<P: AsRef<Path>>(model_dir: P) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let byte_to_char = Self::byte_to_char();

        // 256 raw byte tokens + 5000 merges + 1 special token end of text
        let token_to_id: HashMap<Token, TokenId> =
            serde_json_from_path(model_dir.join("encoder.json"))?;

        Ok(Self {
            word_re: Regex::new(
                r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
            )
            .unwrap(),
            char_to_byte: byte_to_char.invert(),
            byte_to_char,
            id_to_token: token_to_id.invert(),
            token_to_id,
            bpe_ranks: bpe_ranks_from_path(model_dir.join("vocab.bpe"))?,
            tokenize_cache: HashMap::new(),
        })
    }

    pub fn encode(&mut self, text: &str) -> Result<Vec<TokenId>> {
        let mut token_ids = Vec::new();

        for word in self.word_re.find_iter(text) {
            let word: String = word
                .context("failed to match word")?
                .as_str()
                .as_bytes()
                .iter()
                .map(|b| self.byte_to_char.get(b).unwrap())
                .collect();

            if let Some(cached_token_ids) = self.tokenize_cache.get(&word) {
                // if this word is already tokenized, just look up the token
                token_ids.extend(cached_token_ids.iter());
            } else {
                // if this is an unseen word, then tokenize it
                let tokens = self.tokenize(word.clone());
                let new_token_ids: Vec<TokenId> = tokens
                    .into_iter()
                    .map(|token| {
                        *self
                            .token_to_id
                            .get(&token)
                            .with_context(|| format!("unexpected token {token}"))
                            .unwrap()
                    })
                    .collect();

                token_ids.extend(new_token_ids.iter());

                self.tokenize_cache.insert(word, new_token_ids);
            }
        }

        Ok(token_ids)
    }

    pub fn decode(&self, token_ids: &[TokenId]) -> String {
        let tokens: String = token_ids
            .iter()
            .map(|token_id| {
                self.id_to_token
                    .get(token_id)
                    .with_context(|| format!("unexpected token id: {token_id}"))
                    .unwrap()
                    .as_str()
            })
            .collect::<Vec<_>>()
            .join("");

        String::from_utf8(
            tokens
                .chars()
                .map(|c| *self.char_to_byte.get(&c).unwrap())
                .collect(),
        )
        .unwrap()
    }

    /// map bytes to chars
    /// a simple one-to-one mapping of bytes 0..255 into unicode characters
    /// that "look nice"
    /// eg 32 -> Ġ
    /// ref <https://github.com/karpathy/minGPT/blob/37baab71b9abea1b76ab957409a1cc2fbfba8a26/mingpt/bpe.py#L22-L33>
    /// need to allow converting a character literal (char) to a u8
    #[allow(clippy::char_lit_as_u8)]
    fn byte_to_char() -> HashMap<u8, char> {
        // first map the bytes to printable chars
        let mut printable: HashSet<u8> = (b'!'..=b'~').collect();
        printable.extend('¡' as u8..='¬' as u8);
        printable.extend('®' as u8..='ÿ' as u8);

        let mut map = HashMap::new();
        let mut n = 2u32.pow(8);

        // map the remaining bytes to chars > 256, which are mostly printable
        for byte in u8::MIN..=u8::MAX {
            if !printable.contains(&byte) {
                map.insert(byte, char::from_u32(n).unwrap());

                n += 1;
            }
        }

        map.extend(printable.into_iter().map(|byte| (byte, byte as char)));

        map
    }

    /// all adjacent pairs
    fn parts_to_pairs(parts: &[String]) -> HashSet<StringPair> {
        std::iter::zip(parts.iter().cloned(), parts.iter().skip(1).cloned()).collect()
    }

    /// tokenizes a string using the merges in BPE
    fn tokenize(&self, word: String) -> Vec<Token> {
        let mut parts: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        loop {
            let pairs: HashSet<StringPair> = Self::parts_to_pairs(&parts);

            // get the highest priority merge
            let Some(bigram) = pairs
                .iter()
                .min_by_key(|pair| self.bpe_ranks.get(pair).copied().unwrap_or(usize::MAX))
            else {
                // no more merges
                // else block handles the case when the self.bpe_ranks.get() function doesn't return any value
                return vec![word];
            };

            // handles the case when string pair's priority is usize::MAX
            if !self.bpe_ranks.contains_key(bigram) {
                // no more bpe merges
                break;
            }

            let (first, second) = bigram;

            // use the merged pair to replace the old pair
            let mut new_parts = Vec::new();
            let mut i = 0;
            while i < parts.len() {
                if i == parts.len() - 1 {
                    new_parts.push(parts[i].to_string());
                    break;
                }

                if &parts[i] == first && &parts[i + 1] == second {
                    new_parts.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    new_parts.push(parts[i].clone());
                    i += 1;
                }
            }

            parts = new_parts;
        }

        parts
    }
}
