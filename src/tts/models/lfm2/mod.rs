mod attention;
mod conv;
pub mod generate;

use candle_core::{DType, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, Module, VarBuilder};

use super::config::Lfm2Config;
use attention::Lfm2Attention;
use conv::Lfm2ShortConv;

pub use generate::{generate, GenerateParams};

/// RMSNorm — `weight * x / sqrt(mean(x²) + eps)`
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(vb: VarBuilder, dim: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let variance = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

/// SwiGLU MLP: w2(silu(w1(x)) * w3(x))
struct Lfm2Mlp {
    w1: Linear,
    w2: Linear,
    w3: Linear,
}

impl Lfm2Mlp {
    fn load(vb: VarBuilder, dim: usize, ff_dim: usize) -> Result<Self> {
        let w1 = linear_no_bias(dim, ff_dim, vb.pp("w1"))?;
        let w2 = linear_no_bias(ff_dim, dim, vb.pp("w2"))?;
        let w3 = linear_no_bias(dim, ff_dim, vb.pp("w3"))?;
        Ok(Self { w1, w2, w3 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::ops::silu(&self.w1.forward(x)?)?;
        let up = self.w3.forward(x)?;
        self.w2.forward(&(gate * up)?)
    }
}

/// A single LFM2 layer — either conv-based or attention-based.
enum Lfm2Layer {
    Conv {
        operator_norm: RmsNorm,
        conv: Lfm2ShortConv,
        ffn_norm: RmsNorm,
        feed_forward: Lfm2Mlp,
    },
    Attention {
        operator_norm: RmsNorm,
        self_attn: Lfm2Attention,
        ffn_norm: RmsNorm,
        feed_forward: Lfm2Mlp,
    },
}

impl Lfm2Layer {
    fn load(vb: VarBuilder, cfg: &Lfm2Config, layer_type: &str) -> Result<Self> {
        let operator_norm =
            RmsNorm::load(vb.pp("operator_norm"), cfg.hidden_size, cfg.norm_eps)?;
        let ffn_norm = RmsNorm::load(vb.pp("ffn_norm"), cfg.hidden_size, cfg.norm_eps)?;
        let feed_forward =
            Lfm2Mlp::load(vb.pp("feed_forward"), cfg.hidden_size, cfg.intermediate_size)?;

        match layer_type {
            "conv" => {
                let conv = Lfm2ShortConv::load(
                    vb.pp("conv"),
                    cfg.hidden_size,
                    cfg.conv_l_cache,
                )?;
                Ok(Self::Conv {
                    operator_norm,
                    conv,
                    ffn_norm,
                    feed_forward,
                })
            }
            "full_attention" => {
                let self_attn = Lfm2Attention::load(vb.pp("self_attn"), cfg)?;
                Ok(Self::Attention {
                    operator_norm,
                    self_attn,
                    ffn_norm,
                    feed_forward,
                })
            }
            other => candle_core::bail!("Unknown layer type: {other}"),
        }
    }

    /// Forward pass. Returns the residual-added output.
    /// `conv_states` is updated in-place for conv layers during generation.
    /// `kv_cache` is (key, value) for attention layers.
    fn forward(
        &self,
        x: &Tensor,
        conv_state: &mut Option<Tensor>,
        kv_cache: &mut Option<(Tensor, Tensor)>,
        seq_offset: usize,
    ) -> Result<Tensor> {
        match self {
            Self::Conv {
                operator_norm,
                conv,
                ffn_norm,
                feed_forward,
            } => {
                let residual = x;
                let h = operator_norm.forward(x)?;
                let h = conv.forward(&h, conv_state)?;
                let h = (residual + h)?;
                let residual = &h;
                let h = ffn_norm.forward(&h)?;
                let h = feed_forward.forward(&h)?;
                residual + h
            }
            Self::Attention {
                operator_norm,
                self_attn,
                ffn_norm,
                feed_forward,
            } => {
                let residual = x;
                let h = operator_norm.forward(x)?;
                let h = self_attn.forward(&h, kv_cache, seq_offset)?;
                let h = (residual + h)?;
                let residual = &h;
                let h = ffn_norm.forward(&h)?;
                let h = feed_forward.forward(&h)?;
                residual + h
            }
        }
    }
}

/// LFM2 causal language model.
pub struct Lfm2ForCausalLM {
    embed_tokens: Embedding,
    embedding_norm: RmsNorm,
    layers: Vec<Lfm2Layer>,
    /// Pre-transposed F32 lm_head weight: (hidden_size, vocab_size).
    /// Avoids repeated BF16→F32 + transpose on every generation step.
    lm_head_weight_t: Tensor,
    config: Lfm2Config,
}

impl Lfm2ForCausalLM {
    pub fn load(vb: VarBuilder, cfg: &Lfm2Config) -> Result<Self> {
        let model_vb = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, model_vb.pp("embed_tokens"))?;
        let embedding_norm =
            RmsNorm::load(model_vb.pp("embedding_norm"), cfg.hidden_size, cfg.norm_eps)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (i, lt) in cfg.layer_types.iter().enumerate() {
            let layer = Lfm2Layer::load(model_vb.pp(&format!("layers.{i}")), cfg, lt)?;
            layers.push(layer);
        }

        // Pre-transpose and upcast lm_head weight once (tied with embed_tokens)
        let lm_head_weight_t = embed_tokens
            .embeddings()
            .to_dtype(DType::F32)?
            .t()?
            .contiguous()?; // (hidden_size, vocab_size)

        Ok(Self {
            embed_tokens,
            embedding_norm,
            layers,
            lm_head_weight_t,
            config: cfg.clone(),
        })
    }

    /// Forward pass. Returns logits tensor of shape (batch, seq_len, vocab_size).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        conv_states: &mut Vec<Option<Tensor>>,
        kv_caches: &mut Vec<Option<(Tensor, Tensor)>>,
        seq_offset: usize,
    ) -> Result<Tensor> {
        let mut h = self.embed_tokens.forward(input_ids)?;
        // NOTE: embedding_norm is the FINAL layer norm, applied AFTER all layers
        // (not before — the Python reference confirms this)

        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, &mut conv_states[i], &mut kv_caches[i], seq_offset)?;
        }

        // Final RMSNorm before lm_head (prevents hidden state magnitude explosion)
        h = self.embedding_norm.forward(&h)?;

        // lm_head: h @ W_t where W_t is pre-transposed F32 (hidden, vocab)
        let (b, t, d) = h.dims3()?;
        let h_flat = h.to_dtype(DType::F32)?.reshape((b * t, d))?;
        let logits = h_flat.matmul(&self.lm_head_weight_t)?;
        logits.reshape((b, t, self.config.vocab_size))
    }

    pub fn config(&self) -> &Lfm2Config {
        &self.config
    }

    /// Create empty conv states and KV caches for generation.
    pub fn init_state(&self) -> (Vec<Option<Tensor>>, Vec<Option<(Tensor, Tensor)>>) {
        let conv_states = vec![None; self.config.num_hidden_layers];
        let kv_caches = vec![None; self.config.num_hidden_layers];
        (conv_states, kv_caches)
    }
}
