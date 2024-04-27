use anyhow::{bail, Context, Result};
use burn::{
    config::Config,
    module::Module,
    nn::{attention, Gelu, LayerNorm},
    tensor::{backend::Backend, ElementConversion, Tensor},
};
use burn_tensor::{activation, Data, Int, Shape};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::path::{Path, PathBuf};
use std::{fs::File, io::BufReader};

use crate::encoder::{Token, TokenId};

#[derive(Config)]
pub struct ModelConfig {
    /// Number of tokens in the vocabulary.
    /// GPT-2 has a vocabulary size of 50,257,
    /// which corresponds to the 256 bytes base tokens,
    /// a special end-of-text token and the symbols learned with 50,000 merges
    pub n_vocab: usize,
    /// Maximum context / prompt sequence.
    pub n_ctx: usize,
    /// Number of attention heads.
    /// Must be a divisor of `network_width`.
    pub n_head: usize,
    /// Width of the network, or the embeding dimension.
    pub n_embd: usize,
    /// Width of the network.
    pub n_layer: usize,
}

impl ModelConfig {
    pub fn from_dir(model_dir: PathBuf) -> Self {
        let file = File::open(model_dir.join("hparams.json")).expect("should load params");

        let buffer = BufReader::new(file);

        let config: Self = serde_json::from_reader(buffer).expect("should load params");

        config
    }

    pub fn init<B: Backend>(&self, model_dir: PathBuf, device: &B::Device) -> Model<B> {
        let token_embedding_arr: Array2<f32> =
            ndarray_npy::read_npy(model_dir.join("wte.npy")).expect("should load wte");
        let token_embedding_vec: Vec<f32> = token_embedding_arr.iter().copied().collect();

        let position_embedding_arr: Array2<f32> =
            ndarray_npy::read_npy(model_dir.join("wpe.npy")).expect("should load wpe");
        let position_embedding_vec: Vec<f32> = position_embedding_arr.iter().copied().collect();

        let token_embedding: Tensor<B, 2> = Tensor::<B, 2>::from_data(
            Data::new(
                token_embedding_vec.clone(),
                Shape::new([
                    token_embedding_arr.shape()[0],
                    token_embedding_arr.shape()[1],
                ]),
            )
            .convert(),
            device,
        );

        let position_embedding: Tensor<B, 2> = Tensor::<B, 2>::from_data(
            Data::new(
                position_embedding_vec.clone(),
                Shape::new([
                    position_embedding_arr.shape()[0],
                    position_embedding_arr.shape()[1],
                ]),
            )
            .convert(),
            device,
        );

        let layer_norm_config = Gpt2LayerNormConfig {
            layer_norm_dir: model_dir.join("ln_f"),
        };

        let block_config = BlockConfig {
            model_dir: model_dir.to_owned(),
            num_heads: self.n_head,
            depth: self.n_layer,
        };

        Model {
            token_embedding,
            position_embedding,
            blocks: block_config.init(device),
            layer_norm: layer_norm_config.init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub token_embedding: Tensor<B, 2>,
    pub position_embedding: Tensor<B, 2>,
    pub blocks: Vec<Block<B>>,
    layer_norm: Gpt2LayerNorm<B>,
}

impl<B: Backend> Model<B> {
    pub fn generate(&self, mut inputs: Vec<TokenId>, num_tokens: usize) -> Vec<TokenId> {
        let device = B::Device::default();

        for _ in 0..num_tokens {
            let logits = self.forward(&inputs);

            // select the last row from logits
            // which corresponds to the logits for next token
            let indices = Tensor::<B, 1, Int>::from_data(
                Data::new(vec![(logits.dims()[0] - 1) as i32], Shape::new([1])).convert(),
                &device,
            );
            // size (1xvocab size)
            let selected_logits = logits.select(0, indices);
            // argmax along dim 1, not dim 0
            let next_token_id = selected_logits.argmax(1);
            let next_token_id = next_token_id.into_scalar().elem::<i32>();

            inputs.push(next_token_id as u64);
        }

        inputs[inputs.len() - num_tokens..].to_vec()
    }

    fn forward(&self, inputs: &[TokenId]) -> Tensor<B, 2> {
        let device = B::Device::default();

        #[allow(clippy::cast_possible_truncation)]
        let inputs: Vec<i32> = inputs.iter().map(|token_id| *token_id as i32).collect();
        let indices = Tensor::<B, 1, Int>::from_data(
            Data::new(inputs.clone(), Shape::new([inputs.len()])).convert(),
            &device,
        );
        let token_embeddings = self.token_embedding.clone().select(0, indices);

        let indices = (0..inputs.len()).map(|i| i as i32).collect::<Vec<_>>();
        let indices = Tensor::<B, 1, Int>::from_data(
            Data::new(indices.clone(), Shape::new([indices.len()])).convert(),
            &device,
        );
        let position_embeddings = self.position_embedding.clone().select(0, indices);

        // x size (10x768)
        let mut x = token_embeddings + position_embeddings;

        for block in &self.blocks {
            x = block.forward(x);
        }

        let x = self.layer_norm.forward(x);
        // reuse the embedding matrix wte for the projection
        let x = x.matmul(self.token_embedding.clone().transpose());

        x
    }
}

#[derive(Config)]
pub struct Gpt2LinearLayerConfig {
    pub weights_dir: PathBuf,
}

impl Gpt2LinearLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2LinearLayer<B> {
        let weights_arr: Array2<f32> =
            ndarray_npy::read_npy(self.weights_dir.join("w.npy")).expect("should load w.npy");
        let bias_arr: Array1<f32> =
            ndarray_npy::read_npy(self.weights_dir.join("b.npy")).expect("should load b.npy");

        let weights_vec = weights_arr.iter().cloned().collect::<Vec<_>>();
        let bias_vec = bias_arr.iter().cloned().collect::<Vec<_>>();

        let weights: Tensor<B, 2> = Tensor::<B, 2>::from_data(
            Data::new(
                weights_vec.clone(),
                Shape::new([weights_arr.shape()[0], weights_arr.shape()[1]]),
            )
            .convert(),
            device,
        );
        let bias: Tensor<B, 1> = Tensor::<B, 1>::from_data(
            Data::new(weights_vec.clone(), Shape::new([bias_vec.len()])).convert(),
            device,
        );

        Gpt2LinearLayer { weights, bias }
    }
}

#[derive(Module, Debug)]
pub struct Gpt2LinearLayer<B: Backend> {
    pub weights: Tensor<B, 2>,
    pub bias: Tensor<B, 1>,
}

impl<B: Backend> Gpt2LinearLayer<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let bias = self.bias.clone().unsqueeze::<2>();
        x.matmul(self.weights.clone()) + bias
    }
}

#[derive(Config)]
pub struct FeedForwardConfig {
    pub block_dir: PathBuf,
}

impl FeedForwardConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> FeedForward<B> {
        let expand_config = Gpt2LinearLayerConfig {
            weights_dir: self.block_dir.join("mlp/c_fc"),
        };
        let contract_config = Gpt2LinearLayerConfig {
            weights_dir: self.block_dir.join("mlp/c_proj"),
        };

        let layer_norm_config = Gpt2LayerNormConfig {
            layer_norm_dir: self.block_dir.join("ln_2"),
        };

        FeedForward {
            layer_norm: layer_norm_config.init(device),
            expand: expand_config.init(device),
            contract: contract_config.init(device),
            activation: Gelu::new(),
        }
    }
}

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub layer_norm: Gpt2LayerNorm<B>,
    pub expand: Gpt2LinearLayer<B>,
    pub contract: Gpt2LinearLayer<B>,
    pub activation: Gelu,
}

impl<B: Backend> FeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_norm.forward(x);
        let x = self.expand.forward(x);
        let x = self.activation.forward(x);

        let output = self.contract.forward(x);

        output
    }
}

#[derive(Config)]
pub struct AttentionConfig {
    pub block_dir: PathBuf,
    pub num_heads: usize,
}

impl AttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Attention<B> {
        let expand_config = Gpt2LinearLayerConfig {
            weights_dir: self.block_dir.join("attn/c_attn"),
        };
        let contract_config = Gpt2LinearLayerConfig {
            weights_dir: self.block_dir.join("attn/c_proj"),
        };

        let layer_norm_config = Gpt2LayerNormConfig {
            layer_norm_dir: self.block_dir.join("ln_1"),
        };

        Attention {
            layer_norm: layer_norm_config.init(device),
            expand: expand_config.init(device),
            contract: contract_config.init(device),
            num_heads: self.num_heads,
        }
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub layer_norm: Gpt2LayerNorm<B>,
    pub expand: Gpt2LinearLayer<B>,
    pub contract: Gpt2LinearLayer<B>,
    pub num_heads: usize,
}

impl<B: Backend> Attention<B> {
    fn attention(
        q: &Tensor<B, 2>,
        k: &Tensor<B, 2>,
        v: &Tensor<B, 2>,
        causal_mask: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let d = (k.dims()[1] as f32).sqrt();
        let kt = k.clone().transpose();
        let qk = q.clone().matmul(kt) / d + causal_mask.clone();
        let probs = activation::softmax(qk, 1);
        let v = probs.matmul(v.clone());
        v
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_norm.forward(x);

        // x size (10x2034)
        let x = self.expand.forward(x.clone());
        let qkv = x.clone().chunk(3, 1);
        let qkv_heads = qkv
            .iter()
            .map(|v| v.clone().chunk(self.num_heads, 1))
            .collect::<Vec<_>>();

        let device = B::Device::default();
        let head_shape = x.dims()[0];
        // mask size (10x10)
        let causal_mask = (Tensor::ones(Shape::new([head_shape, head_shape]), &device)
            - Tensor::ones(Shape::new([head_shape, head_shape]), &device).tril(0))
            * -1.0e4;

        let out_heads = std::iter::zip(std::iter::zip(&qkv_heads[0], &qkv_heads[1]), &qkv_heads[2])
            .map(|((q, k), v)| Self::attention(q, k, v, &causal_mask))
            .collect();
        let out_heads_concat = Tensor::cat(out_heads, 1);
        let x = self.contract.forward(out_heads_concat);

        x
    }
}

#[derive(Config)]
pub struct BlockConfig {
    pub model_dir: PathBuf,
    pub num_heads: usize,
    pub depth: usize,
}

impl BlockConfig {
    pub fn init_block<B: Backend>(&self, block_dir: PathBuf, device: &B::Device) -> Block<B> {
        let attention_config = AttentionConfig {
            block_dir: block_dir.clone(),
            num_heads: self.num_heads,
        };
        let attention = attention_config.init(device);

        let feedforward_config = FeedForwardConfig {
            block_dir: block_dir.to_path_buf(),
        };
        let feedforward = feedforward_config.init(device);

        Block {
            attention,
            feedforward,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> Vec<Block<B>> {
        (0..self.depth)
            .map(|block_idx| self.init_block(self.model_dir.join(format!("h{block_idx}")), device))
            .collect()
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub attention: Attention<B>,
    pub feedforward: FeedForward<B>,
}

impl<B: Backend> Block<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = x.clone() + self.attention.forward(x);
        let x = x.clone() + self.feedforward.forward(x);
        x
    }
}

#[derive(Config)]
pub struct Gpt2LayerNormConfig {
    pub layer_norm_dir: PathBuf,
}

impl Gpt2LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Gpt2LayerNorm<B> {
        let beta_arr: Array1<f32> =
            ndarray_npy::read_npy(self.layer_norm_dir.join("b.npy")).expect("should load b.npy");
        let beta_vec = beta_arr.iter().cloned().collect::<Vec<_>>();
        let gamma_arr: Array1<f32> =
            ndarray_npy::read_npy(self.layer_norm_dir.join("g.npy")).expect("should load g.npy");
        let gamma_vec = gamma_arr.iter().cloned().collect::<Vec<_>>();

        let beta: Tensor<B, 1> = Tensor::<B, 1>::from_data(
            Data::new(beta_vec.clone(), Shape::new([beta_vec.len()])).convert(),
            device,
        );
        let gamma: Tensor<B, 1> = Tensor::<B, 1>::from_data(
            Data::new(gamma_vec.clone(), Shape::new([gamma_vec.len()])).convert(),
            device,
        );

        Gpt2LayerNorm { beta, gamma }
    }
}

/// this struct loads the learned parameters for layer norm
#[derive(Module, Debug)]
pub struct Gpt2LayerNorm<B: Backend> {
    pub beta: Tensor<B, 1>,
    pub gamma: Tensor<B, 1>,
}

impl<B: Backend> Gpt2LayerNorm<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let eps = 1e-5;

        let mean = x.clone().mean_dim(1);
        let var = x.clone().var(1);

        let x = (x - mean) / (var + eps).sqrt();

        let gamma = self.gamma.clone().unsqueeze::<2>();
        let gamma = gamma.repeat(0, x.dims()[0]);
        let beta: Tensor<B, 2> = self.beta.clone().unsqueeze::<2>();
        let beta = beta.repeat(0, x.dims()[0]);

        // size (10x1) * (10x768) + (10x1)
        let output = gamma * x + beta;

        output
    }
}
