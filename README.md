<div align="center">

# Fluffy

**A 3D desktop mascot with a Speech-to-Speech AI brain**

[![Rust](https://img.shields.io/badge/Rust-2021-orange?logo=rust)](https://www.rust-lang.org/)
[![Bevy](https://img.shields.io/badge/Bevy-0.18-blue?logo=bevy)](https://bevyengine.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Talk to a 3D character that lives on your desktop. Powered by Claude, Whisper, and MioTTS — all running locally in pure Rust.*

<!-- ![demo](assets/demo.gif) -->

</div>

---

## What is this?

Fluffy is an always-on-top 3D desktop mascot that you can actually talk to.

You speak → it listens (Whisper STT) → thinks (Claude) → responds in voice (MioTTS) — all while a 3D character reacts with expressions and lip sync.

```
Mic → VAD → Whisper STT → Claude LLM (streaming) → MioTTS → Speaker
                                  ├─► Text → chat panel
                                  └─► Lip sync + expressions → 3D mascot

TTS (fully local, no Python):
  text → LFM2 2.6B (codec tokens) → MioCodec (44.1kHz waveform)
  reference.wav → WavLM → speaker embedding (once at startup)
```

## Controls

| Key / Action | Effect |
|---|---|
| `Space` | Start / stop voice pipeline |
| `T` | Toggle chat panel |
| `C` | Toggle click-through |
| Left-click drag | Move window |

## Features

- **Transparent window** — floats over your desktop, always on top
- **Speech-to-Speech** — full voice conversation loop
- **Local TTS** — MioTTS 2.6B runs entirely in Rust via candle + CUDA, no Python server needed
- **Voice cloning** — provide a reference WAV and MioTTS clones the speaker's voice
- **Streaming LLM** — tokens appear in real time as Claude responds
- **Lip sync** — mouth moves with TTS audio amplitude
- **Procedural idle animation** — breathing and subtle head sway
- **Chat overlay** — conversation history on the right panel
- **Click-through** — transparent areas let clicks pass to windows below

## Requirements

| Tool | Purpose |
|---|---|
| [Rust](https://rustup.rs/) 1.80+ | Build toolchain |
| CUDA toolkit | GPU inference for MioTTS (`pacman -S cuda` on Arch) |
| [Whisper GGML model](https://huggingface.co/ggerganov/whisper.cpp) | Speech recognition |
| `ANTHROPIC_API_KEY` | Claude API |
| Reference WAV | Voice for TTS (place at `assets/voice/`) |

TTS models are **downloaded automatically** from HuggingFace on first run:
- [Aratako/MioTTS-2.6B](https://huggingface.co/Aratako/MioTTS-2.6B) (~5.2 GB, BF16)
- [Aratako/MioCodec-25Hz-44.1kHz-v2](https://huggingface.co/Aratako/MioCodec-25Hz-44.1kHz-v2) (~0.5 GB)
- [microsoft/wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus) (~0.4 GB, freed after init)

### System libraries (Linux)

```bash
# Arch
sudo pacman -S alsa-lib wayland cuda

# Ubuntu / Debian
sudo apt install libasound2-dev libwayland-dev nvidia-cuda-toolkit
```

## Getting Started

### 1. Clone

```bash
git clone https://github.com/m96-chan/fluffy.git
cd fluffy
```

### 2. Place a 3D model

```bash
mkdir -p assets/models
cp /path/to/your/model.glb assets/models/mascot.glb
```

### 3. Place a reference voice

```bash
mkdir -p assets/voice
cp /path/to/reference.wav assets/voice/reference.wav
```

This WAV is used for voice cloning — MioTTS will speak in this voice.

### 4. Get a Whisper model

```bash
mkdir -p ~/.local/share/fluffy/models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin \
     -O ~/.local/share/fluffy/models/ggml-base.bin
```

### 5. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 6. Run

```bash
cargo run --release --features cuda
```

> First build takes a while — Bevy is big. Subsequent builds are fast.
> Without `--features cuda`, TTS falls back to CPU (much slower).

## TTS Performance

Benchmarked on RTX 5090 with BF16, synthesizing 3 seconds of Japanese speech:

| Stage | Time |
|---|---|
| Prefill (19 tokens) | 18 ms |
| Decode (76 tokens) | 610 ms (125 tok/s) |
| MioCodec decode | 7 ms |
| **Total** | **0.63 s** |
| **RTF** | **0.21** |

Faster than the equivalent Python/PyTorch pipeline (RTF 0.22).

## Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Claude API key |
| `FLUFFY_MODEL_DIR` | `~/.local/share/fluffy/models` | Whisper model directory |

### Window config

Persisted automatically to `~/.config/fluffy/window.toml`:

```toml
x = 100
y = 100
click_through = false
```

## Architecture

```
fluffy/
├── src/
│   ├── main.rs              # Bevy App setup
│   ├── events.rs            # PipelineMessage (Bevy 0.18 Message system)
│   ├── state.rs             # AppConfig, PipelineState (Bevy Resources)
│   │
│   ├── pipeline/            # Speech-to-Speech pipeline (async tokio)
│   │   ├── coordinator.rs   #   VAD → STT → LLM → TTS orchestration
│   │   ├── relay.rs         #   async → Bevy message bridge
│   │   └── plugin.rs        #   Space key start/stop
│   │
│   ├── audio/               # cpal capture, rodio playback, VAD
│   ├── stt/                 # whisper-rs + model downloader
│   ├── llm/                 # Claude streaming API + tool use
│   ├── tts/
│   │   ├── engine.rs        #   TtsEngine: model init, warmup, synthesis
│   │   ├── client.rs        #   synthesize() API
│   │   ├── sentence.rs      #   Sentence boundary detection
│   │   └── models/
│   │       ├── lfm2/        #   LFM2 2.6B (30-layer conv+attention LM)
│   │       ├── miocodec/    #   MioCodec (FSQ → Transformer → iSTFT)
│   │       ├── wavlm/       #   WavLM + GlobalEncoder → speaker embedding
│   │       ├── download.rs  #   HuggingFace auto-download
│   │       └── config.rs    #   Model configs
│   │
│   ├── mascot/              # 3D model rendering, expression + lip sync
│   ├── animation/           # Procedural breathing + head sway
│   ├── chat/                # Chat overlay UI (Bevy UI)
│   └── window/              # Position, click-through, drag
│
├── patches/
│   ├── bevy_mesh/           # MAX_MORPH_WEIGHTS: 256 → 512
│   └── bevy_pbr/            # WGSL shader morph target limit patch
│
└── assets/
    ├── models/              # 3D model (.glb)
    └── voice/               # Reference WAV for voice cloning
```

### Key design decisions

**Pure Rust TTS**
MioTTS runs entirely in Rust using [candle](https://github.com/huggingface/candle) for GPU inference. No Python, no external TTS server, no ONNX runtime. The three-model pipeline (LFM2 → MioCodec → WavLM) is implemented from scratch against the original Python reference.

A [known candle issue](https://github.com/huggingface/candle/issues/3389) makes depthwise conv1d extremely slow with large groups. Fluffy works around this by expanding the convolution into element-wise operations, achieving a 38x prefill speedup.

**Pipeline ↔ Bevy bridge**
The voice pipeline runs as a `tokio` async task. It sends `PipelineMessage` values through an `mpsc` channel. A Bevy system drains this channel every frame and forwards messages into Bevy's message system, keeping the async world and ECS world cleanly separated.

## Running tests

```bash
cargo test
```

## License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built with [Bevy](https://bevyengine.org/) · [whisper-rs](https://github.com/tazz4843/whisper-rs) · [Claude](https://anthropic.com/) · [candle](https://github.com/huggingface/candle) · [MioTTS](https://huggingface.co/Aratako/MioTTS-2.6B)

</div>
