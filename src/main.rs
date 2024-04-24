use anyhow::{Context, Result};
use burn::backend::wgpu::WgpuDevice;
use burn::backend::{wgpu::AutoGraphicsApi, Autodiff, Wgpu};
use std::env;
use std::path::PathBuf;

use pico_gpt::encoder::Encoder;
use pico_gpt::gpt2::Model;
use pico_gpt::hyper_params::HyperParams;

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

fn main() -> Result<()> {
    let current_dir = env::current_dir().expect("Failed to get current directory");
    let model_dir = current_dir.join("data/124M");

    let prompt = "Alan Turing theorized that computers would one day become";
    let num_tokens = 8;

    let hyper_params = HyperParams::from_dir(&model_dir).context("cannot load hyper params")?;
    let model = Model::<Backend>::from_dir(
        model_dir.join("exploded_model"),
        hyper_params.num_heads,
        hyper_params.network_depth,
    )
    .context("params")?;

    let mut encoder = Encoder::from_dir(&model_dir).context("cannot load encoder")?;

    let token_ids = encoder.encode(&prompt).context("cannot encode prompt")?;
    anyhow::ensure!(
        token_ids.len() < hyper_params.max_context,
        "input too large"
    );

    let output_ids = model.generate(token_ids, num_tokens);
    let decoded = encoder.decode(&output_ids);
    println!("output = {decoded:?}");

    Ok(())
}
