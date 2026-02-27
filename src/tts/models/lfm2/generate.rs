use candle_core::{DType, Device, Result, Tensor};
use tracing::{debug, info};

use super::Lfm2ForCausalLM;

/// Parameters for autoregressive generation.
#[derive(Debug, Clone)]
pub struct GenerateParams {
    pub temperature: f64,
    pub top_p: f64,
    pub max_tokens: usize,
    pub eos_token_id: u32,
}

impl Default for GenerateParams {
    fn default() -> Self {
        Self {
            temperature: 0.8,
            top_p: 1.0,
            max_tokens: 700,
            eos_token_id: 7,
        }
    }
}

/// Generate codec tokens from the LFM2 model.
///
/// Takes tokenized input_ids and generates until EOS or max_tokens.
/// Returns the full generated token sequence (excluding input).
pub fn generate(
    model: &Lfm2ForCausalLM,
    input_ids: &[u32],
    params: &GenerateParams,
    device: &Device,
) -> Result<Vec<u32>> {
    // Create initial input tensor (single H2D at prefill start)
    let input = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), device)?;

    // Initialize state
    let (mut conv_states, mut kv_caches) = model.init_state();

    // Prefill: process all input tokens at once
    let logits = model.forward(&input, &mut conv_states, &mut kv_caches, 0)?;

    // Get logits for the last position
    let seq_len = input_ids.len();
    let mut last_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

    // Diagnostic: show top-5 logits from prefill (expensive D2H — debug only)
    if tracing::enabled!(tracing::Level::DEBUG) {
        let diag = last_logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut indexed: Vec<(usize, f32)> = diag.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top5: Vec<String> = indexed[..5].iter()
            .map(|(i, v)| format!("{i}={v:.3}")).collect();
        let eos_logit = diag.get(params.eos_token_id as usize).copied().unwrap_or(f32::NAN);
        debug!("Prefill logits top5: [{}], eos({})={:.3}", top5.join(", "), params.eos_token_id, eos_logit);
        let nans = diag.iter().filter(|v| v.is_nan()).count();
        let infs = diag.iter().filter(|v| v.is_infinite()).count();
        if nans > 0 || infs > 0 {
            debug!("WARNING: {} NaN, {} Inf in logits", nans, infs);
        }
    }

    let mut generated = Vec::with_capacity(params.max_tokens);
    let mut offset = seq_len;
    let use_gpu_sampling = params.top_p >= 1.0;
    let gen_start = std::time::Instant::now();

    for step in 0..params.max_tokens {
        // Sample next token — GPU-side Gumbel-max for top_p=1.0
        // avoids 77K float D2H per step; only 4 bytes (single u32) transferred
        let (next_token, next_input) = if use_gpu_sampling {
            let (id, tensor) = sample_gumbel_max(&last_logits, params.temperature)?;
            (id, tensor.reshape((1, 1))?)
        } else {
            let id = sample_top_p(&last_logits, params.temperature, params.top_p)?;
            (id, Tensor::from_vec(vec![id], (1, 1), device)?)
        };

        if next_token == params.eos_token_id {
            info!("EOS at step {step}");
            break;
        }
        generated.push(next_token);

        if step % 100 == 99 {
            let elapsed = gen_start.elapsed().as_secs_f32();
            info!("  step {}: tok/s = {:.1}, last_id = {next_token}", step + 1, (step + 1) as f32 / elapsed);
        }

        // Forward pass — next_input already on device (no H2D for token)
        let logits = model.forward(&next_input, &mut conv_states, &mut kv_caches, offset)?;
        last_logits = logits.squeeze(1)?;
        offset += 1;

        // Diagnostic: first 3 steps — show logits (expensive D2H — debug only)
        if step < 3 && tracing::enabled!(tracing::Level::DEBUG) {
            let diag = last_logits.squeeze(0)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
            let mut indexed: Vec<(usize, f32)> = diag.iter().copied().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top5: Vec<String> = indexed[..5].iter()
                .map(|(i, v)| format!("{i}={v:.3}")).collect();
            let tok_logit = diag.get(70774).copied().unwrap_or(f32::NAN);
            debug!("  step {} logits: top5=[{}], 70774={:.3}", step, top5.join(", "), tok_logit);
        }
    }

    let elapsed = gen_start.elapsed().as_secs_f32();
    info!("Generated {} tokens in {:.2}s ({:.1} tok/s)", generated.len(), elapsed, generated.len() as f32 / elapsed);
    Ok(generated)
}

/// GPU-side sampling using the Gumbel-max trick.
///
/// `argmax(logits/T + Gumbel_noise)` samples from `softmax(logits/T)`.
/// Only transfers a single u32 from device to host per step,
/// vs 77K floats for CPU-side categorical sampling.
fn sample_gumbel_max(logits: &Tensor, temperature: f64) -> Result<(u32, Tensor)> {
    let logits = logits.squeeze(0)?; // (vocab_size,)
    let logits_f32 = logits.to_dtype(DType::F32)?;

    // Greedy for very low temperature
    if temperature < 1e-10 {
        let token_tensor = logits_f32.argmax(0)?;
        let token_id = token_tensor.to_scalar::<u32>()?;
        return Ok((token_id, token_tensor));
    }

    let scaled = (logits_f32 / temperature)?;

    // Gumbel noise: -log(-log(U)), U ~ Uniform(eps, 1-eps)
    let device = scaled.device();
    let u = Tensor::rand(1e-7f32, 1f32 - 1e-7, scaled.shape(), device)?;
    let neg_log_u = (u.log()? * (-1.0f64))?;
    let gumbel = (neg_log_u.log()? * (-1.0f64))?;

    let perturbed = (scaled + gumbel)?;
    let token_tensor = perturbed.argmax(0)?; // scalar U32 on device
    let token_id = token_tensor.to_scalar::<u32>()?; // tiny D2H: 4 bytes

    Ok((token_id, token_tensor))
}

/// Top-p (nucleus) sampling with temperature (CPU fallback for top_p < 1.0).
fn sample_top_p(logits: &Tensor, temperature: f64, top_p: f64) -> Result<u32> {
    // logits: (1, vocab_size)
    let logits = logits.squeeze(0)?; // (vocab_size,)
    let logits = if temperature != 1.0 {
        (logits / temperature)?
    } else {
        logits
    };

    // Softmax to get probabilities
    let probs = candle_nn::ops::softmax_last_dim(&logits.unsqueeze(0)?)?.squeeze(0)?;
    let probs_vec: Vec<f32> = probs.to_dtype(DType::F32)?.to_vec1()?;

    if top_p >= 1.0 {
        // Standard categorical sampling
        return categorical_sample(&probs_vec);
    }

    // Top-p: sort by probability descending, keep cumulative sum <= top_p
    let mut indexed: Vec<(usize, f32)> = probs_vec.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut filtered = Vec::new();
    for (idx, prob) in &indexed {
        cumulative += prob;
        filtered.push((*idx, *prob));
        if cumulative >= top_p as f32 {
            break;
        }
    }

    // Renormalize
    let total: f32 = filtered.iter().map(|(_, p)| p).sum();
    let normalized: Vec<f32> = filtered.iter().map(|(_, p)| p / total).collect();

    // Sample from filtered distribution
    let r: f32 = rand_f32();
    let mut cumulative = 0.0f32;
    for (i, prob) in normalized.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            return Ok(filtered[i].0 as u32);
        }
    }

    Ok(filtered.last().map(|(idx, _)| *idx as u32).unwrap_or(0))
}

fn categorical_sample(probs: &[f32]) -> Result<u32> {
    let r = rand_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return Ok(i as u32);
        }
    }
    Ok((probs.len() - 1) as u32)
}

/// Simple thread-local RNG for sampling. No external rand crate needed.
fn rand_f32() -> f32 {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = Cell::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64
        );
    }
    STATE.with(|s| {
        // xorshift64
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        // Convert to [0, 1)
        (x >> 40) as f32 / (1u64 << 24) as f32
    })
}

/// Parse generated tokens to extract `<|s_N|>` codec tokens.
///
/// The LFM2 model outputs special tokens like `<|s_0|>`, `<|s_1|>`, ..., `<|s_12799|>`
/// which represent MioCodec indices. This function extracts those indices from
/// the generated token IDs using the tokenizer's vocabulary.
pub fn extract_codec_tokens(
    generated_ids: &[u32],
    tokenizer: &tokenizers::Tokenizer,
) -> Vec<u32> {
    let mut codec_tokens = Vec::new();

    for &token_id in generated_ids {
        if let Some(token_str) = tokenizer.id_to_token(token_id) {
            if let Some(idx) = parse_codec_token(&token_str) {
                codec_tokens.push(idx);
            }
        }
    }

    codec_tokens
}

/// Parse a token string like `<|s_123|>` → Some(123).
fn parse_codec_token(token: &str) -> Option<u32> {
    let inner = token.strip_prefix("<|s_")?.strip_suffix("|>")?;
    inner.parse().ok()
}

/// Build the chat-formatted prompt for TTS generation.
///
/// Format: `<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n`
pub fn build_prompt(text: &str) -> String {
    format!("<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_codec_token_valid() {
        assert_eq!(parse_codec_token("<|s_0|>"), Some(0));
        assert_eq!(parse_codec_token("<|s_12799|>"), Some(12799));
        assert_eq!(parse_codec_token("<|s_42|>"), Some(42));
    }

    #[test]
    fn parse_codec_token_invalid() {
        assert_eq!(parse_codec_token("hello"), None);
        assert_eq!(parse_codec_token("<|s_|>"), None);
        assert_eq!(parse_codec_token("<|s_abc|>"), None);
        assert_eq!(parse_codec_token("<|im_start|>"), None);
    }

    #[test]
    fn build_prompt_format() {
        let prompt = build_prompt("こんにちは");
        assert_eq!(
            prompt,
            "<|im_start|>user\nこんにちは<|im_end|>\n<|im_start|>assistant\n"
        );
    }
}
