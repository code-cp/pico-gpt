use anyhow::{Context, Result};
use clap::parser;
use std::path::Path;

use pico_gpt::encoder::Encoder;
use pico_gpt::gpt2::Model;
use pico_gpt::hyper_params::HyperParams;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,

    #[arg(long)]
    prompt: String,

    #[arg(long)]
    num_tokens: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let hyper_params =
        HyperParams::from_dir(&args.model_dir).context("cannot load hyper params")?;
    let model = Model::from_dir(
        args.model_dir.join("exploded_model"),
        hyper_params.num_heads,
        hyper_params.network_depth,
    )
    .context("params")?;

    let mut encoder = Encoder::from_dir(&args.model_dir).context("cannot load encoder")?;

    let token_ids = encoder
        .encode(&args.prompt)
        .context("cannot encode prompt")?;
    anyhow::ensure!(
        token_ids.len() < hyper_params.max_context,
        "input too large"
    );

    let output_ids = model.generate(token_ids, args.num_tokens);
    let decoded = encoder.decode(&output_ids);
    println!("output = {decoded:?}");

    Ok(())
}
