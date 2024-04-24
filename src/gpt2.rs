use anyhow::{bail, Context, Result};
use burn::{
    config::Config,
    module::Module,
    nn::{Gelu, LayerNorm},
    tensor::{backend::Backend, ElementConversion, Tensor},
};
use burn_tensor::{activation, Data, Int, Shape};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::path::Path;

use crate::encoder::{Token, TokenId};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub token_embedding: Tensor<B, 2>,
    pub position_embedding: Tensor<B, 2>,
    pub blocks: Vec<Block<B>>,
    layer_norm: MyLayerNorm<B>,
}

impl<B: Backend> Model<B> {
    pub fn from_dir<P: AsRef<Path>>(model_dir: P, num_heads: usize, depth: usize) -> Result<Self> {
        let model_dir = model_dir.as_ref();

        let token_embedding_arr: Array2<f32> =
            ndarray_npy::read_npy(model_dir.join("wte.npy")).context("cannot load wte")?;
        let token_embedding_vec: Vec<f32> = token_embedding_arr.iter().copied().collect();

        let position_embedding_arr: Array2<f32> =
            ndarray_npy::read_npy(model_dir.join("wpe.npy")).context("cannot load wpe")?;
        let position_embedding_vec: Vec<f32> = position_embedding_arr.iter().copied().collect();

        let device = B::Device::default();

        let token_embedding: Tensor<B, 2> = Tensor::<B, 2>::from_data(
            Data::new(
                token_embedding_vec.clone(),
                Shape::new([
                    token_embedding_arr.shape()[0],
                    token_embedding_arr.shape()[1],
                ]),
            )
            .convert(),
            &device,
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
            &device,
        );

        Ok(Self {
            token_embedding,
            position_embedding,
            blocks: Block::from_dirs(model_dir, num_heads, depth).context("cannot load block")?,
            layer_norm: MyLayerNorm::from_dir(model_dir.join("ln_f"), &device)
                .context("cannot load layer norm")?,
        })
    }

    pub fn generate(&self, mut inputs: Vec<TokenId>, num_tokens: usize) -> Vec<TokenId> {
        let device = B::Device::default();

        for _ in 0..num_tokens {
            let logits = self.forward(&inputs);

            let indices = Tensor::<B, 1, Int>::from_data(
                Data::new(vec![(logits.dims()[0] - 1) as i32], Shape::new([1])).convert(),
                &device,
            );
            let next_token_id: Tensor<B, 1, Int> = logits.select(1, indices).argmax(0).unsqueeze();
            let next_token_id = next_token_id.into_scalar().elem::<f32>();

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

        let inputs = (0..inputs.len()).map(|i| i as i32).collect::<Vec<_>>();
        let indices = Tensor::<B, 1, Int>::from_data(
            Data::new(inputs.clone(), Shape::new([inputs.len()])).convert(),
            &device,
        );
        let position_embeddings = self.position_embedding.clone().select(0, indices);

        let mut x = token_embeddings + position_embeddings;

        for block in &self.blocks {
            x = block.forward(x);
        }

        let x = self.layer_norm.forward(x);
        let x = x.matmul(self.token_embedding.clone().transpose());

        x
    }
}

#[derive(Module, Debug)]
pub struct MyLinearLayer<B: Backend> {
    pub weights: Tensor<B, 2>,
    pub bias: Tensor<B, 1>,
}

impl<B: Backend> MyLinearLayer<B> {
    pub fn from_dir<P: AsRef<Path>>(weights_dir: P, device: &B::Device) -> Result<Self> {
        let weights_dir = weights_dir.as_ref();

        let weights_arr: Array2<f32> = ndarray_npy::read_npy(weights_dir.join("w.npy"))?;
        let bias_arr: Array1<f32> = ndarray_npy::read_npy(weights_dir.join("b.npy"))?;

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

        Ok(Self { weights, bias })
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let bias = self.bias.clone().unsqueeze::<2>();
        x.matmul(self.weights.clone()) + bias
    }
}

#[derive(Module, Debug)]
pub struct Network<B: Backend> {
    pub layer_norm: MyLayerNorm<B>,
    pub expand: MyLinearLayer<B>,
    pub contract: MyLinearLayer<B>,
    pub activation: Gelu,
}

impl<B: Backend> Network<B> {
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_norm.forward(x);
        let x = self.expand.forward(x);
        let x = self.activation.forward(x);

        self.contract.forward(x)
    }
}

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub layer_norm: MyLayerNorm<B>,
    pub expand: MyLinearLayer<B>,
    pub contract: MyLinearLayer<B>,
    pub num_heads: usize,
}

impl<B: Backend> Attention<B> {
    fn split_n(x: Tensor<B, 2>, num: usize) -> Result<Vec<Tensor<B, 2>>> {
        let last_axis_length = x.dims()[1];

        if last_axis_length % num != 0 {
            bail!("axis of size {last_axis_length} is not divisible by {num}");
        }

        let subarray_size = last_axis_length / num;
        let mut splits = Vec::new();

        let device = B::Device::default();

        for i in 0..num {
            let indices = Tensor::<B, 1, Int>::from_data(
                Data::new(
                    (0..subarray_size)
                        .map(|j| (i * subarray_size + j) as i32)
                        .collect::<Vec<_>>(),
                    Shape::new([subarray_size]),
                )
                .convert(),
                &device,
            );
            let split = x.clone().select(1, indices);
            splits.push(split);
        }

        Ok(splits)
    }

    fn attention(
        q: &Tensor<B, 2>,
        k: &Tensor<B, 2>,
        v: &Tensor<B, 2>,
        causal_mask: &Tensor<B, 2>,
    ) -> Tensor<B, 2> {
        let d = (k.dims()[2] as f32).sqrt();
        let kt = k.clone().transpose();
        let qk = q.clone().matmul(kt) / d + causal_mask.clone();
        let probs = activation::softmax(qk, 1);
        let v = probs.matmul(v.clone());
        v
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer_norm.forward(x);

        let x = self.expand.forward(x);
        let qkv = Self::split_n(x, 3).unwrap();
        let qkv_heads = qkv
            .iter()
            .map(|v| Self::split_n(v.clone(), self.num_heads).unwrap())
            .collect::<Vec<_>>();

        let device = B::Device::default();
        let head_shape = qkv_heads[0][0].dims();
        let causal_mask = (Tensor::ones(head_shape, &device)
            - Tensor::ones(head_shape, &device).tril(0))
            * -1.0e4;

        let out_heads = std::iter::zip(std::iter::zip(&qkv_heads[0], &qkv_heads[1]), &qkv_heads[2])
            .map(|((q, k), v)| Self::attention(q, k, v, &causal_mask))
            .collect();
        let out_heads_concat = Tensor::cat(out_heads, 1);
        let x = self.contract.forward(out_heads_concat);

        x
    }
}

#[derive(Module, Debug)]
pub struct Block<B: Backend> {
    pub attention: Attention<B>,
    pub network: Network<B>,
}

impl<B: Backend> Block<B> {
    pub fn from_dirs<P: AsRef<Path>>(
        model_dir: P,
        num_heads: usize,
        depth: usize,
    ) -> Result<Vec<Self>> {
        let model_dir = model_dir.as_ref();

        (0..depth)
            .map(|block_idx| Self::from_dir(model_dir.join(format!("h{block_idx}")), num_heads))
            .collect()
    }

    fn from_dir<P: AsRef<Path>>(block_dir: P, num_heads: usize) -> Result<Self> {
        let block_dir = block_dir.as_ref();

        let device = B::Device::default();
        let attention = Attention {
            layer_norm: MyLayerNorm::from_dir(block_dir.join("ln_1"), &device)
                .context("cannot load ln_1")?,
            expand: MyLinearLayer::from_dir(block_dir.join("attn/c_attn"), &device)
                .context("cannot load c_attn")?,
            contract: MyLinearLayer::from_dir(block_dir.join("attn/c_proj"), &device)
                .context("cannot load c_proj")?,
            num_heads,
        };
        let network = Network {
            layer_norm: MyLayerNorm::from_dir(block_dir.join("ln_2"), &device)
                .context("cannot load ln_2")?,
            expand: MyLinearLayer::from_dir(block_dir.join("mlp/c_fc"), &device)
                .context("cannot load mlp/c_fc")?,
            contract: MyLinearLayer::from_dir(block_dir.join("mlp/c_proj"), &device)
                .context("cannot load mlp/c_proj")?,
            activation: Gelu::new(),
        };

        Ok(Self { attention, network })
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = x.clone() + self.attention.forward(x);
        let x = x.clone() + self.network.forward(x);
        x
    }
}

/// this struct loads the learned parameters for layer norm
#[derive(Module, Debug)]
pub struct MyLayerNorm<B: Backend> {
    pub beta: Tensor<B, 1>,
    pub gamma: Tensor<B, 1>,
}

impl<B: Backend> MyLayerNorm<B> {
    fn from_dir<P: AsRef<Path>>(layer_norm_dir: P, device: &B::Device) -> Result<Self> {
        let layer_norm_dir = layer_norm_dir.as_ref();

        let beta_arr: Array1<f32> = ndarray_npy::read_npy(layer_norm_dir.join("b.npy"))?;
        let beta_vec = beta_arr.iter().cloned().collect::<Vec<_>>();
        let gamma_arr: Array1<f32> = ndarray_npy::read_npy(layer_norm_dir.join("g.npy"))?;
        let gamma_vec = gamma_arr.iter().cloned().collect::<Vec<_>>();

        let beta: Tensor<B, 1> = Tensor::<B, 1>::from_data(
            Data::new(beta_vec.clone(), Shape::new([beta_vec.len()])).convert(),
            device,
        );
        let gamma: Tensor<B, 1> = Tensor::<B, 1>::from_data(
            Data::new(gamma_vec.clone(), Shape::new([gamma_vec.len()])).convert(),
            device,
        );

        Ok(Self { beta, gamma })
    }

    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let eps = 1e-5;

        let mean = x.clone().mean_dim(1);
        let var = x.clone().var(1);

        let x = (x - mean) / (var + eps).sqrt();

        let gamma = self.gamma.clone().unsqueeze::<2>();
        let gamma = gamma.repeat(0, x.dims()[1]);
        let beta: Tensor<B, 2> = self.beta.clone().unsqueeze::<2>();

        gamma.matmul(x) + beta
    }
}
