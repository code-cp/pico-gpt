use anyhow::{Context, Result};
use burn::backend::wgpu::WgpuDevice;
use std::env;

use pico_gpt::encoder::Encoder;
use pico_gpt::gpt2::{Model, ModelConfig};

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

fn main() -> Result<()> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let model_dir = current_dir.join("data/124M");

    let prompt = "Alan Turing theorized that computers would one day become";
    let num_tokens = 8;

    let model_config = ModelConfig::from_dir(model_dir.clone());
    let device = WgpuDevice::default();
    let model: Model<Backend> = model_config.init(model_dir.join("exploded_model"), &device);

    let mut encoder = Encoder::from_dir(model_dir).context("cannot load encoder")?;

    let token_ids = encoder.encode(&prompt).context("cannot encode prompt")?;
    anyhow::ensure!(token_ids.len() < model_config.n_ctx, "input too large");

    let output_ids = model.generate(token_ids, num_tokens);
    let decoded = encoder.decode(&output_ids);
    println!("output = {decoded:?}");

    Ok(())
}
