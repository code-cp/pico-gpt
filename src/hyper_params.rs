use std::{fs::File, io::BufReader, path::Path};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Deserialize)]
pub struct HyperParams {
    /// Number of tokens in the vocabulary.
    #[serde(rename = "n_vocab")]
    pub size_vocab: usize,
    /// Maximum context / prompt sequence.
    #[serde(rename = "n_ctx")]
    pub max_context: usize,
    /// Number of attention heads.
    /// Must be a divisor of `network_width`.
    #[serde(rename = "n_head")]
    pub num_heads: usize,
    /// Width of the network, or the embeding dimension.
    #[serde(rename = "n_embd")]
    pub network_width: usize,
    /// Width of the network.
    #[serde(rename = "n_layer")]
    pub network_depth: usize,
}

impl HyperParams {
    pub fn from_dir<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file =
            File::open(path.join("hparams.json")).with_context(|| format!("file: {path:?}"))?;
        let buffer = BufReader::new(file);

        Ok(serde_json::from_reader(buffer)?)
    }
}
