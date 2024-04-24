use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub type StringPair = (String, String);

pub fn serde_json_from_path<P: AsRef<Path>, T: DeserializeOwned>(path: P) -> Result<T> {
    let file = File::open(&path).with_context(|| format!("file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);

    Ok(serde_json::from_reader(reader)?)
}

/// load the bpe results and their priority
pub fn bpe_ranks_from_path<P: AsRef<Path>>(path: P) -> Result<HashMap<StringPair, usize>> {
    let file = File::open(&path).with_context(|| format!("file: {:?}", path.as_ref()))?;
    let reader = BufReader::new(file);

    let mut bpe_ranks = HashMap::new();

    for (priority, line) in reader.lines().skip(1).enumerate() {
        let line = line.context("cannot read line")?;
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split(' ').collect();
        anyhow::ensure!(parts.len() == 2);

        bpe_ranks.insert((parts[0].to_string(), parts[1].to_string()), priority);
    }

    Ok(bpe_ranks)
}
