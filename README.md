<div align="center">

# 🐾 Fluffy

**A 3D desktop mascot with a Speech-to-Speech AI brain**

[![Rust](https://img.shields.io/badge/Rust-2021-orange?logo=rust)](https://www.rust-lang.org/)
[![Bevy](https://img.shields.io/badge/Bevy-0.18-blue?logo=bevy)](https://bevyengine.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-109%20passing-brightgreen)]()

*Talk to a VRM character that lives on your desktop. Powered by Claude, Whisper, and Fish Speech.*

<!-- ![demo](assets/demo.gif) -->

</div>

---

## ✨ What is this?

Fluffy is an always-on-top 3D desktop mascot that you can actually talk to.

You speak → it listens (Whisper STT) → thinks (Claude) → responds in voice (Fish Speech TTS) — all while a cute VRM character reacts with expressions and animations.

```
🎤 You speak
   └─► VAD detects speech
         └─► Whisper transcribes
               └─► Claude responds (streaming)
                     ├─► Text → chat panel
                     └─► TTS → character speaks + lip sync
```

## 🎮 Controls

| Key / Action | Effect |
|---|---|
| `Space` | Start / stop voice pipeline |
| `T` | Toggle chat panel |
| `C` | Toggle click-through (window becomes non-interactive) |
| Left-click drag | Move window |

## 🖼️ Features

- **Any VRM model** — supports VRM 0.x and VRM 1.0
- **Transparent window** — floats over your desktop, always on top
- **Speech-to-Speech** — full voice conversation loop
- **Streaming LLM** — tokens appear in real time as Claude responds
- **Lip sync** — mouth moves with TTS audio amplitude
- **Facial expressions** — `[happy]`, `[sad]`, `[surprised]`... extracted from LLM output
- **Procedural idle animation** — breathing and subtle head sway
- **Chat overlay** — conversation history on the right panel
- **VRMA support** — play `.vrma` animation files (VRM 1.0 retargeting)
- **Click-through** — transparent areas let clicks pass to windows below

## 🛠️ Requirements

| Tool | Purpose |
|---|---|
| [Rust](https://rustup.rs/) 1.80+ | Build toolchain |
| [Whisper GGML model](https://huggingface.co/ggerganov/whisper.cpp) | Speech recognition |
| `ANTHROPIC_API_KEY` | Claude API |
| [Fish Speech](https://github.com/fishaudio/fish-speech) (optional) | TTS server at `localhost:7860` |

### System libraries (Linux)

```bash
# Arch
sudo pacman -S alsa-lib wayland

# Ubuntu / Debian
sudo apt install libasound2-dev libwayland-dev
```

## 🚀 Getting Started

### 1. Clone

```bash
git clone https://github.com/m96-chan/fluffy.git
cd fluffy
```

### 2. Place a VRM model

```bash
mkdir -p assets/models
cp /path/to/your/model.vrm assets/models/mascot.vrm
```

### 3. Get a Whisper model

```bash
mkdir -p ~/.local/share/fluffy/models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin \
     -O ~/.local/share/fluffy/models/ggml-base.bin
```

Or set a custom path:

```bash
export FLUFFY_MODEL_DIR=/path/to/models
```

### 4. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 5. Run

```bash
cargo run --release
```

> **Note:** First build takes a while — Bevy is big. Subsequent builds are fast thanks to `dynamic_linking`.

## ⚙️ Configuration

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(required)* | Claude API key |
| `FLUFFY_TTS_URL` | `http://localhost:7860` | Fish Speech server URL |
| `FLUFFY_MODEL_DIR` | `~/.local/share/fluffy/models` | Whisper model directory |

### Window config

Persisted automatically to `~/.config/fluffy/window.toml`:

```toml
x = 100
y = 100
click_through = false
```

## 🏗️ Architecture

```
fluffy/
├── src/
│   ├── main.rs              # Bevy App setup
│   ├── events.rs            # PipelineMessage (Bevy 0.18 Message system)
│   ├── state.rs             # AppConfig, PipelineState (Bevy Resources)
│   │
│   ├── vrm/                 # VRM loader (VRM 0.x + 1.x, self-implemented)
│   │   ├── loader.rs        #   .vrm → GltfLoader delegate
│   │   ├── expressions.rs   #   Blend shape / morph target parser
│   │   └── vrma/            #   VRMA animation support
│   │       ├── loader.rs
│   │       ├── retarget.rs  #   Bone name retargeting (camelCase → PascalCase)
│   │       └── player.rs
│   │
│   ├── pipeline/            # Speech-to-Speech pipeline (async tokio)
│   │   ├── coordinator.rs   #   VAD → STT → LLM → TTS orchestration
│   │   ├── relay.rs         #   async → Bevy message bridge
│   │   └── plugin.rs        #   Space key start/stop
│   │
│   ├── audio/               # cpal capture, rodio playback, VAD
│   ├── stt/                 # whisper-rs + model downloader
│   ├── llm/                 # Claude streaming API + tool use
│   ├── tts/                 # Fish Speech client + sentence splitter
│   │
│   ├── mascot/              # VRM rendering, expression + lip sync
│   ├── animation/           # Procedural breathing + head sway
│   ├── chat/                # Chat overlay UI (Bevy UI)
│   └── window/              # Position, click-through, drag
│
├── patches/
│   ├── bevy_mesh/           # MAX_MORPH_WEIGHTS: 256 → 512
│   └── bevy_pbr/            # WGSL shader: array<vec4<f32>, 64u> → 128u
│
└── assets/
    └── models/
        └── mascot.vrm
```

### Key design decisions

**Why not `bevy_vrm1`?**
`bevy_vrm1` only supports VRM 1.0. Most models from VRoid Studio and Booth are VRM 0.x. Fluffy implements its own thin VRM loader on top of Bevy's GLTF support, handling both formats.

**Why patch Bevy?**
Bevy 0.18 hard-codes a 256 morph target limit in both the Rust code and WGSL shader. Real-world VRoid models often have 400–500+ morph targets. The patch raises this to 512. An upstream PR to make this configurable is planned ([#7](https://github.com/m96-chan/fluffy/issues/7)).

**Pipeline ↔ Bevy bridge**
The voice pipeline runs as a `tokio` async task. It sends `PipelineMessage` values through an `mpsc` channel. A Bevy system (`PipelineRelayPlugin`) drains this channel every frame and forwards messages into Bevy's message system, keeping the async world and ECS world cleanly separated.

## 🤝 Contributing

PRs welcome! See [open issues](https://github.com/m96-chan/fluffy/issues) for ideas.

### Running tests

```bash
cargo test
```

109 unit tests covering pipeline state, VRM expression parsing, VRMA retargeting, procedural animation math, window drag calculation, and more.

### Known issues / roadmap

- [ ] Whisper download progress UI (#2)
- [ ] Idle → talking animation transition polish (#4)
- [ ] VRMA playback on VRM 0.x models (bone name mapping) (#5)
- [ ] Upstream Bevy PR: raise morph target limit (#7)
- [ ] macOS / Windows support (currently Linux / X11 / Wayland)

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">

Built with [Bevy](https://bevyengine.org/) · [whisper-rs](https://github.com/tazz4843/whisper-rs) · [Claude](https://anthropic.com/) · [Fish Speech](https://github.com/fishaudio/fish-speech)

</div>
