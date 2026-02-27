use candle_core::{DType, Result, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder};

/// Finite Scalar Quantizer — decode path only.
///
/// Converts a flat index (0..product(levels)-1) to a multi-dimensional code
/// and projects it through a linear layer to get the embedding.
///
/// levels = [8, 8, 8, 5, 5] → codebook size = 8*8*8*5*5 = 12800
pub struct FsqDecode {
    proj_out: Linear,
    levels: Vec<usize>,
    /// Basis for decomposing flat index: cumprod([1] + levels[:-1])
    /// For levels [8,8,8,5,5]: basis = [1, 8, 64, 512, 2560]
    basis: Vec<usize>,
    /// Integer half-width for centering: levels // 2
    /// For levels [8,8,8,5,5]: half_width = [4, 4, 4, 2, 2]
    half_width: Vec<usize>,
}

impl FsqDecode {
    pub fn load(vb: VarBuilder, levels: &[usize]) -> Result<Self> {
        let ndim = levels.len();
        let proj_out = linear(ndim, 768, vb.pp("proj_out"))?;

        // basis = cumprod([1] + levels[:-1]) — matches Python FSQ._basis
        let mut basis = vec![1usize; ndim];
        for i in 1..ndim {
            basis[i] = basis[i - 1] * levels[i - 1];
        }

        let half_width: Vec<usize> = levels.iter().map(|&l| l / 2).collect();

        Ok(Self {
            proj_out,
            levels: levels.to_vec(),
            basis,
            half_width,
        })
    }

    /// Decode flat token indices → 768-dim embeddings.
    ///
    /// Input: (seq_len,) u32 tensor with values 0..12799
    /// Output: (seq_len, 768) f32 tensor
    pub fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        let _seq_len = indices.dim(0)?;
        let device = indices.device();
        let ndim = self.levels.len();

        // Convert indices to f32 for arithmetic
        let indices_f32 = indices.to_dtype(DType::F32)?;

        // Decompose flat index → per-dimension codes using basis (LSB-first)
        // Matches Python: codes_non_centered = (indices // basis) % levels
        // Then: _scale_and_shift_inverse = (code - half_width) / half_width
        let mut dim_codes = Vec::with_capacity(ndim);
        for i in 0..ndim {
            let basis = self.basis[i] as f32;
            let level = self.levels[i] as f32;
            let hw = self.half_width[i] as f32;

            let basis_t = Tensor::from_vec(vec![basis], 1, device)?;
            let level_t = Tensor::from_vec(vec![level], 1, device)?;

            // floor(index / basis) % level
            let code = indices_f32.broadcast_div(&basis_t)?.floor()?;
            // Modulo: x - floor(x/level)*level
            let code = (&code - code.broadcast_div(&level_t)?.floor()?.broadcast_mul(&level_t)?)?;

            // Normalize: (code - half_width) / half_width
            let code = ((code - hw as f64)? / hw as f64)?;

            dim_codes.push(code);
        }

        // Stack: (seq_len,) × ndim → (seq_len, ndim)
        let codes = Tensor::stack(&dim_codes, 1)?;

        // Project: (seq_len, ndim) → (seq_len, 768)
        self.proj_out.forward(&codes)
    }
}
