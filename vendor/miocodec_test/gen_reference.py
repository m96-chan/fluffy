#!/usr/bin/env python3
"""
Generate reference intermediate tensors from the Python MioCodec implementation.
Saves .npy files for each pipeline stage so Rust tests can compare.

Usage:
  python gen_reference.py                    # random speaker embedding (default)
  python gen_reference.py --wav path/to.wav  # real speaker embedding from WAV
"""
import sys
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import yaml
from pathlib import Path
from safetensors.torch import load_file

# Add MioCodec source to path
sys.path.insert(0, str(Path(__file__).parent / "miocodec_src" / "src"))

MIOCODEC_DIR = Path(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Aratako--MioCodec-25Hz-44.1kHz-v2"
))
SNAPSHOT = list((MIOCODEC_DIR / "snapshots").iterdir())[0]
SAFETENSORS = SNAPSHOT / "model.safetensors"
CONFIG_YAML = SNAPSHOT / "config.yaml"
OUT_DIR = Path(__file__).parent / "reference_data"


def load_config():
    with open(CONFIG_YAML) as f:
        raw = yaml.safe_load(f)
    return raw["model"]["init_args"]


def save(name: str, tensor: torch.Tensor):
    OUT_DIR.mkdir(exist_ok=True)
    arr = tensor.detach().float().cpu().numpy()
    np.save(OUT_DIR / f"{name}.npy", arr)
    print(f"  {name}: shape={list(arr.shape)}, "
          f"mean={arr.mean():.6f}, rms={np.sqrt((arr**2).mean()):.6f}, "
          f"min={arr.min():.6f}, max={arr.max():.6f}")


def load_submodule(state_dict, prefix, module):
    sd = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
    module.load_state_dict(sd, strict=True)
    module.eval()
    return module


def compute_speaker_embedding(wav_path, state_dict, cfg, device):
    """Compute 128-dim speaker embedding from a WAV file using WavLM + GlobalEncoder."""
    from miocodec.model import SSLFeatureExtractor, GlobalEncoder

    print(f"\n=== Computing speaker embedding from {wav_path} ===")
    model_cfg = cfg["config"]

    # Load audio and resample to 44.1kHz (MioCodec's sample rate)
    import scipy.io.wavfile as wavfile
    sr, audio_raw = wavfile.read(str(wav_path))
    if audio_raw.ndim > 1:
        audio_raw = audio_raw.mean(axis=1)
    # Normalize to float32 [-1, 1]
    if audio_raw.dtype == np.int16:
        audio_f32 = audio_raw.astype(np.float32) / 32768.0
    elif audio_raw.dtype == np.int32:
        audio_f32 = audio_raw.astype(np.float32) / 2147483648.0
    else:
        audio_f32 = audio_raw.astype(np.float32)
    waveform = torch.from_numpy(audio_f32).unsqueeze(0)  # (1, T)
    target_sr = model_cfg.get("sample_rate", 44100)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    waveform = waveform.to(device)
    print(f"  Audio: {waveform.shape[1]} samples at {target_sr}Hz ({waveform.shape[1]/target_sr:.2f}s)")

    # Load WavLM (SSLFeatureExtractor handles resampling to 16kHz internally)
    ssl_cfg = cfg.get("ssl_feature_extractor", {}).get("init_args", {})
    ssl_model_name = ssl_cfg.get("model_name", "wavlm_base_plus")
    ssl_extractor = SSLFeatureExtractor(
        model_name=ssl_model_name,
        sample_rate=target_sr,
    ).to(device)
    ssl_extractor.eval()

    # Extract SSL features from specified layers
    global_ssl_layers = model_cfg.get("global_ssl_layers", [1, 2])
    max_layer = max(global_ssl_layers)
    print(f"  SSL layers: {global_ssl_layers}, max={max_layer}")

    with torch.no_grad():
        # Save the 16kHz resampled audio for debugging
        if ssl_extractor.resampler is not None:
            waveform_16k = ssl_extractor.resampler(waveform)
        else:
            waveform_16k = waveform
        save("spk_waveform_16k", waveform_16k)
        print(f"  Waveform 16kHz: {waveform_16k.shape}")

        all_features = ssl_extractor(waveform, num_layers=max_layer)
        # Save per-layer features
        for l_idx, feat in enumerate(all_features):
            save(f"spk_wavlm_layer{l_idx+1}", feat)
        # Average the specified layers
        selected = [all_features[l - 1] for l in global_ssl_layers]  # 1-based index
        ssl_features = torch.stack(selected).mean(dim=0)  # (B, T, 768)
    save("spk_ssl_features_avg", ssl_features)
    print(f"  SSL features: {ssl_features.shape}")

    # Free WavLM
    del ssl_extractor

    # Load GlobalEncoder
    ge_cfg = cfg.get("global_encoder", {}).get("init_args", {})
    global_encoder = GlobalEncoder(**ge_cfg).to(device)
    ge_sd = {k[len("global_encoder."):]: v for k, v in state_dict.items() if k.startswith("global_encoder.")}
    global_encoder.load_state_dict(ge_sd, strict=True)
    global_encoder.eval()

    with torch.no_grad():
        speaker_emb = global_encoder(ssl_features)  # (1, 128)
    save("spk_speaker_emb_full", speaker_emb)
    speaker_emb = speaker_emb.squeeze(0)  # (128,)
    print(f"  Speaker embedding: {speaker_emb.shape}, rms={speaker_emb.norm().item():.4f}")

    del global_encoder
    return speaker_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", type=str, default=None,
                        help="Path to reference WAV for real speaker embedding")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    state_dict = load_file(str(SAFETENSORS), device=str(device))
    cfg = load_config()
    model_cfg = cfg["config"]
    print(f"Loaded {len(state_dict)} tensors")

    # --- Fixed test inputs ---
    codec_indices = torch.tensor(
        [4083, 4528, 3750, 3779, 6985, 9756, 10048, 10130,
         11382, 7325, 7064, 7080, 7118, 8282, 10099, 10316,
         5927, 6427, 2660, 639, 5163, 12512, 5046, 0],
        dtype=torch.long, device=device
    )

    # Speaker embedding: from WAV or random
    if args.wav:
        speaker_emb = compute_speaker_embedding(args.wav, state_dict, cfg, device)
    else:
        print("\n(Using random speaker embedding — pass --wav for real voice)")
        torch.manual_seed(42)
        speaker_emb = torch.randn(128, device=device)

    save("input_codec_indices", codec_indices)
    save("input_speaker_emb", speaker_emb)

    # ===== Stage 1: FSQ Decode =====
    print("\n=== Stage 1: FSQ Decode ===")
    from miocodec.module.fsq import FSQ
    fsq_levels = cfg["local_quantizer"]["init_args"]["levels"]
    print(f"  levels: {fsq_levels}")

    fsq = FSQ(fsq_levels).to(device)
    codes = fsq.indices_to_codes(codec_indices)  # (24, 5)
    save("fsq_codes", codes)

    proj_out_w = state_dict["local_quantizer.proj_out.weight"]
    proj_out_b = state_dict["local_quantizer.proj_out.bias"]
    content_emb = codes @ proj_out_w.t() + proj_out_b  # (24, 768)
    save("fsq_decode_output", content_emb)

    # ===== Stage 2: wave_prenet =====
    print("\n=== Stage 2: wave_prenet ===")
    from miocodec.module.transformer import Transformer

    wp_args = cfg["wave_prenet"]["init_args"]
    wave_prenet = Transformer(**wp_args).to(device)
    load_submodule(state_dict, "wave_prenet.", wave_prenet)

    x = content_emb.unsqueeze(0)  # (1, 24, 768)
    with torch.no_grad():
        prenet_out = wave_prenet(x)  # (1, 24, 512)
    save("wave_prenet_output", prenet_out)

    # ===== Stage 3: wave_conv_upsample =====
    print("\n=== Stage 3: wave_conv_upsample ===")
    upsample_factor = model_cfg["wave_upsample_factor"]
    conv_up = torch.nn.ConvTranspose1d(512, 512, kernel_size=upsample_factor, stride=upsample_factor).to(device)
    load_submodule(state_dict, "wave_conv_upsample.", conv_up)

    x = prenet_out.transpose(1, 2)  # (1, 512, 24)
    with torch.no_grad():
        conv_up_out = conv_up(x)  # (1, 512, 48)
    save("wave_conv_upsample_output", conv_up_out)

    # ===== Stage 4: Interpolate =====
    print("\n=== Stage 4: Interpolate ===")
    stft_length = len(codec_indices) * upsample_factor
    interp_mode = model_cfg.get("wave_interpolation_mode", "linear")
    print(f"  stft_length={stft_length}, mode={interp_mode}")
    interp_out = F.interpolate(conv_up_out, size=stft_length, mode=interp_mode)
    save("interpolate_output", interp_out)

    # ===== Stage 5: wave_prior_net =====
    print("\n=== Stage 5: wave_prior_net ===")
    from miocodec.module.istft_head import ResNetStack
    wave_prior_net = ResNetStack(
        channels=512,
        num_blocks=model_cfg["wave_resnet_num_blocks"],
        kernel_size=model_cfg["wave_resnet_kernel_size"],
        num_groups=model_cfg["wave_resnet_num_groups"],
        dropout=model_cfg.get("wave_resnet_dropout", 0.1),
    ).to(device)
    load_submodule(state_dict, "wave_prior_net.", wave_prior_net)
    with torch.no_grad():
        prior_out = wave_prior_net(interp_out)
    save("wave_prior_net_output", prior_out)

    # ===== Stage 6: wave_decoder =====
    print("\n=== Stage 6: wave_decoder ===")
    wd_args = cfg["wave_decoder"]["init_args"]
    wave_decoder = Transformer(**wd_args).to(device)
    load_submodule(state_dict, "wave_decoder.", wave_decoder)

    x_dec = prior_out.transpose(1, 2)  # (1, 48, 512)
    cond = speaker_emb.unsqueeze(0).unsqueeze(1)  # (1, 1, 128)
    with torch.no_grad():
        decoder_out = wave_decoder(x_dec, condition=cond)
    save("wave_decoder_output", decoder_out)

    # ===== Stage 7: wave_post_net =====
    print("\n=== Stage 7: wave_post_net ===")
    wave_post_net = ResNetStack(
        channels=512,
        num_blocks=model_cfg["wave_resnet_num_blocks"],
        kernel_size=model_cfg["wave_resnet_kernel_size"],
        num_groups=model_cfg["wave_resnet_num_groups"],
        dropout=model_cfg.get("wave_resnet_dropout", 0.1),
    ).to(device)
    load_submodule(state_dict, "wave_post_net.", wave_post_net)
    x_post = decoder_out.transpose(1, 2)  # (1, 512, 48)
    with torch.no_grad():
        post_out = wave_post_net(x_post)
    save("wave_post_net_output", post_out)

    # ===== Stage 8: UpSamplerBlock =====
    print("\n=== Stage 8: wave_upsampler ===")
    from miocodec.module.istft_head import UpSamplerBlock
    wave_upsampler = UpSamplerBlock(
        in_channels=512,
        upsample_factors=model_cfg["wave_upsampler_factors"],
        kernel_sizes=model_cfg["wave_upsampler_kernel_sizes"],
    ).to(device)
    load_submodule(state_dict, "wave_upsampler.", wave_upsampler)
    with torch.no_grad():
        up_out = wave_upsampler(post_out)  # (1, 432, 512) = (B, L', C)
    save("wave_upsampler_output", up_out)

    # ===== Stage 9: ISTFTHead =====
    print("\n=== Stage 9: istft_head ===")
    from miocodec.module.istft_head import ISTFTHead
    n_fft = model_cfg["n_fft"]
    hop_length = model_cfg["hop_length"]
    istft_padding = model_cfg.get("istft_padding", "same")
    print(f"  n_fft={n_fft}, hop={hop_length}, padding={istft_padding}")

    istft_head = ISTFTHead(dim=512, n_fft=n_fft, hop_length=hop_length, padding=istft_padding).to(device)
    # strict=False because istft.window is a buffer not in safetensors
    ih_sd = {k[len("istft_head."):]: v for k, v in state_dict.items() if k.startswith("istft_head.")}
    istft_head.load_state_dict(ih_sd, strict=False)

    with torch.no_grad():
        # Save intermediate: projected
        projected = istft_head.out(up_out)  # (1, 432, 394)
        save("istft_projected", projected)

        projected_t = projected.transpose(1, 2)  # (1, 394, 432)
        mag_raw, phase_raw = projected_t.chunk(2, dim=1)
        save("istft_mag_raw", mag_raw)

        mag = torch.exp(mag_raw)
        mag_clamped = torch.clamp(mag, max=1e2)
        save("istft_mag_exp", mag)
        save("istft_mag_clamped", mag_clamped)

        # Full forward
        audio = istft_head(up_out)
    save("istft_audio_output", audio)

    # Save as WAV for easy listening
    import wave as wave_mod
    audio_np = audio.squeeze(0).cpu().numpy()
    peak = np.max(np.abs(audio_np))
    gain = 0.9 / peak if peak > 1e-6 else 1.0
    audio16 = (audio_np * gain * 32767).astype(np.int16)
    wav_path = OUT_DIR / "reference_output.wav"
    with wave_mod.open(str(wav_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(audio16.tobytes())
    print(f"  WAV: {wav_path}")

    print(f"\n=== Done! {OUT_DIR} ===")
    print(f"Audio: {audio.shape[1]} samples ({audio.shape[1]/44100:.3f}s)")


if __name__ == "__main__":
    main()
