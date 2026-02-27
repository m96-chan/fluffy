use candle_core::{Result, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};

/// LFM2 Short Convolution block — double-gated LIV convolution.
///
/// ```text
/// in_proj(x) → [b_gate, c_gate, x_val]  (3-way split)
/// bx = b_gate * x_val
/// conv_out = causal_conv1d(bx, kernel=L_cache)
/// y = c_gate * conv_out
/// out_proj(y)
/// ```
///
/// During generation (single token), uses a ring buffer of size L_cache.
pub struct Lfm2ShortConv {
    in_proj: Linear,
    conv_weight: Tensor, // shape: (dim, L_cache) — depthwise kernel (squeezed)
    out_proj: Linear,
    dim: usize,
    l_cache: usize,
}

impl Lfm2ShortConv {
    pub fn load(vb: VarBuilder, dim: usize, l_cache: usize) -> Result<Self> {
        // in_proj: (3*dim, dim) — projects to b_gate, c_gate, x_val
        let in_proj = linear_no_bias(dim, 3 * dim, vb.pp("in_proj"))?;
        // conv weight: (dim, 1, L_cache) on disk → squeeze to (dim, L_cache)
        let conv_weight = vb.get((dim, 1, l_cache), "conv.weight")?.squeeze(1)?;
        let out_proj = linear_no_bias(dim, dim, vb.pp("out_proj"))?;

        Ok(Self {
            in_proj,
            conv_weight,
            out_proj,
            dim,
            l_cache,
        })
    }

    /// Forward pass.
    ///
    /// - For prefill (seq_len > 1): full causal conv1d with left-padding.
    /// - For generation (seq_len == 1): uses `conv_state` as a ring buffer.
    ///
    /// `conv_state` shape: (batch, dim, L_cache) — updated in place.
    pub fn forward(&self, x: &Tensor, conv_state: &mut Option<Tensor>) -> Result<Tensor> {
        let (b, t, _d) = x.dims3()?;
        let projected = self.in_proj.forward(x)?;

        // Split into 3 parts along last dim
        let b_gate = projected.narrow(2, 0, self.dim)?;
        let c_gate = projected.narrow(2, self.dim, self.dim)?;
        let x_val = projected.narrow(2, 2 * self.dim, self.dim)?;

        // bx = b_gate * x_val
        let bx = (&b_gate * &x_val)?;

        // Causal conv1d
        let conv_out = if t > 1 {
            // Prefill: full causal conv1d
            // bx shape: (B, T, D) → transpose to (B, D, T) for conv
            let bx_t = bx.transpose(1, 2)?; // (B, D, T)

            // Left-pad with zeros for causal convolution
            let pad = Tensor::zeros((b, self.dim, self.l_cache - 1), bx_t.dtype(), bx_t.device())?;
            let padded = Tensor::cat(&[&pad, &bx_t], 2)?; // (B, D, T + L_cache - 1)

            // Depthwise conv1d via narrow + broadcast_mul (avoids groups=2048 loop)
            // output[:, c, p] = sum_k input[:, c, p+k] * w[c, k]
            // conv_weight_squeezed: (D, L_cache) → narrow to (D, 1) → unsqueeze to (1, D, 1)
            let w = &self.conv_weight;
            let mut out = padded
                .narrow(2, 0, t)?
                .broadcast_mul(&w.narrow(1, 0, 1)?.unsqueeze(0)?)?;
            for k in 1..self.l_cache {
                out = (out
                    + padded
                        .narrow(2, k, t)?
                        .broadcast_mul(&w.narrow(1, k, 1)?.unsqueeze(0)?)?)?;
            }
            // out: (B, D, T)

            // Update conv_state with the last L_cache values
            let state = bx_t.narrow(2, t.saturating_sub(self.l_cache), t.min(self.l_cache))?;
            if state.dim(2)? < self.l_cache {
                let pad_size = self.l_cache - state.dim(2)?;
                let pad = Tensor::zeros(
                    (b, self.dim, pad_size),
                    state.dtype(),
                    state.device(),
                )?;
                *conv_state = Some(Tensor::cat(&[&pad, &state], 2)?);
            } else {
                *conv_state = Some(state);
            }

            out.transpose(1, 2)? // (B, T, D)
        } else {
            // Single-token generation: ring buffer approach
            let bx_squeezed = bx.squeeze(1)?; // (B, D)

            let state = match conv_state.take() {
                Some(mut s) => {
                    // Shift left: drop oldest, append new
                    let new_col = bx_squeezed.unsqueeze(2)?; // (B, D, 1)
                    if self.l_cache > 1 {
                        let kept = s.narrow(2, 1, self.l_cache - 1)?;
                        s = Tensor::cat(&[&kept, &new_col], 2)?;
                    } else {
                        s = new_col;
                    }
                    s
                }
                None => {
                    // First token — pad with zeros
                    let pad = Tensor::zeros(
                        (b, self.dim, self.l_cache - 1),
                        bx_squeezed.dtype(),
                        bx_squeezed.device(),
                    )?;
                    let new_col = bx_squeezed.unsqueeze(2)?;
                    Tensor::cat(&[&pad, &new_col], 2)?
                }
            };

            // Apply conv: sum over the L_cache window (uses pre-squeezed weight)
            // state: (B, D, L_cache), conv_weight_squeezed: (D, L_cache)
            let out = state
                .broadcast_mul(&self.conv_weight.unsqueeze(0)?)?
                .sum(2)?; // (B, D)

            *conv_state = Some(state);
            out.unsqueeze(1)? // (B, 1, D)
        };

        // y = c_gate * conv_out
        let y = (&c_gate * &conv_out)?;
        self.out_proj.forward(&y)
    }
}
