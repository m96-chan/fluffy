use serde::Deserialize;

/// LFM2 model configuration, parsed from config.json.
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct Lfm2Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    #[serde(rename = "conv_L_cache", default = "default_conv_l_cache")]
    pub conv_l_cache: usize,
    #[serde(default)]
    pub layer_types: Vec<String>,
    #[serde(default = "default_true")]
    pub tie_embedding: bool,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default = "default_eos")]
    pub eos_token_id: u32,
    #[serde(default)]
    pub pad_token_id: u32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
}

fn default_norm_eps() -> f64 {
    1e-5
}
fn default_conv_l_cache() -> usize {
    3
}
fn default_true() -> bool {
    true
}
fn default_eos() -> u32 {
    7
}
fn default_max_pos() -> usize {
    128000
}

/// MioCodec decoder configuration, extracted from config.yaml.
#[derive(Debug, Clone)]
pub struct MioCodecConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub wave_upsample_factor: usize,
    pub wave_decoder_dim: usize,
    pub wave_resnet_num_blocks: usize,
    pub wave_resnet_kernel_size: usize,
    pub wave_resnet_num_groups: usize,
    pub wave_upsampler_factors: Vec<usize>,
    pub wave_upsampler_kernel_sizes: Vec<usize>,
    pub fsq_levels: Vec<usize>,
    pub global_ssl_layers: Vec<usize>,
}

impl Default for MioCodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            n_fft: 392,
            hop_length: 98,
            wave_upsample_factor: 2,
            wave_decoder_dim: 512,
            wave_resnet_num_blocks: 2,
            wave_resnet_kernel_size: 3,
            wave_resnet_num_groups: 32,
            wave_upsampler_factors: vec![3, 3],
            wave_upsampler_kernel_sizes: vec![9, 9],
            fsq_levels: vec![8, 8, 8, 5, 5],
            global_ssl_layers: vec![1, 2],
        }
    }
}

/// Parse MioCodec config from YAML string.
/// Since we don't want a YAML dependency, we use sensible defaults
/// and only parse critical values via simple string matching.
impl MioCodecConfig {
    pub fn from_yaml(yaml: &str) -> Self {
        let mut cfg = Self::default();

        for line in yaml.lines() {
            let trimmed = line.trim();
            if let Some((key, val)) = trimmed.split_once(':') {
                let key = key.trim();
                let val = val.trim();
                match key {
                    "sample_rate" => {
                        if let Ok(v) = val.parse() {
                            cfg.sample_rate = v;
                        }
                    }
                    "n_fft" => {
                        if let Ok(v) = val.parse() {
                            cfg.n_fft = v;
                        }
                    }
                    "hop_length" => {
                        if let Ok(v) = val.parse() {
                            cfg.hop_length = v;
                        }
                    }
                    "wave_upsample_factor" | "downsample_factor" => {
                        if let Ok(v) = val.parse() {
                            cfg.wave_upsample_factor = v;
                        }
                    }
                    "wave_decoder_dim" => {
                        if let Ok(v) = val.parse() {
                            cfg.wave_decoder_dim = v;
                        }
                    }
                    "wave_resnet_num_blocks" => {
                        if let Ok(v) = val.parse() {
                            cfg.wave_resnet_num_blocks = v;
                        }
                    }
                    "wave_resnet_kernel_size" => {
                        if let Ok(v) = val.parse() {
                            cfg.wave_resnet_kernel_size = v;
                        }
                    }
                    "wave_resnet_num_groups" => {
                        if let Ok(v) = val.parse() {
                            cfg.wave_resnet_num_groups = v;
                        }
                    }
                    _ => {}
                }
            }
        }

        cfg
    }
}
