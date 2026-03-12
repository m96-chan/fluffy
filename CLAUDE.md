# Fluffy - 3Dデスクトップマスコット + AI Agent

## ビルド & 実行
```bash
cargo build --features cuda    # CUDA有効ビルド
cargo run --features cuda      # 実行
```

## 環境要件
- `.cargo/config.toml` の `PKG_CONFIG_PATH` 設定必須 (alsa, wayland-client)
- 環境変数: `ANTHROPIC_API_KEY`
- CUDA 13.1+, RTX 5090

## アーキテクチャ
- Bevy 0.18 (Rust only, 透過ウィンドウ `WindowLevel::AlwaysOnTop`)
- Speech2Speech: マイク → VAD → Whisper STT → Claude LLM → MioTTS → スピーカー
- 3Dモデル: GLTF/GLB形式 (`assets/models/mascot.glb`)

## Bevy 0.18 API注意点
- `Event` → `Message`, `EventReader` → `MessageReader`, `EventWriter` → `MessageWriter`
- `app.add_event::<T>()` → `app.add_message::<T>()`
- `AmbientLight` はCameraエンティティのComponent (`affects_lightmapped_meshes`フィールド必要)
- `WindowResolution::new(u32, u32)` (f32ではない)
- `Query::get_single()` → `Query::single()` (Result返却)

## Bevy patches
- `patches/bevy_mesh/`, `patches/bevy_pbr/`: モーフターゲット上限 256→512
- `[patch.crates-io]` で適用

## アニメーション / モデル再ベイク
Mixamo FBXアニメーションをChocolatスケルトンにリターゲットするには `/tmp/rebake3.py` を使用:

```bash
blender --background --python /tmp/rebake3.py
```

### リターゲット方式: bone-space correction via conjugation
1. Chocolat GLB (`mascot.glb.bak`) と XBot T-pose FBX を両方インポート
2. 各ボーンペアでワールド空間レスト姿勢の補正クォータニオンを計算:
   - `correction = tgt_rest_world⁻¹ @ src_rest_world`
3. 各フレームで共役変換: `tgt_delta = correction × src_delta × correction⁻¹`
4. NLAトラックに配置して `export_nla_strips=True` でGLBエクスポート

### 注意点
- **Blender 5.0**: `bpy.ops.import_scene.fbx()` は使えない。`from io_scene_fbx.import_fbx import load` で直接呼ぶ
- **アクション検出**: インポート前後の `bpy.data.actions` の差分で新規アクションを特定する（名前マッチングは重複する）
- **FBX座標系**: FBXインポートでarmatureに90° X回転が付く。bone-space補正がこの差異を吸収する
- **ソースファイル**: `~/ダウンロード/X Bot@*.fbx` (Mixamo), `~/ダウンロード/X Bot.fbx` (T-pose)
- **検証**: `/tmp/gltf_viewer.html` をThree.jsビューアーとして使用可能（ドラッグ&ドロップ）

### 現在のアニメーション (6本)
Idle, Standing Greeting, Thinking, Yawn, Sitting, Falling Down

## ウィンドウ構成
- 750×600px固定
- 左400px: Camera3d (キャラクター)
- 右350px: Camera2d (チャットUI)

## 設定ファイル
- `~/.config/fluffy/config.toml` — AppConfig
- `~/.config/fluffy/window.toml` — WindowConfig (位置, click-through)
- `~/.config/fluffy/system_prompt.txt` — システムプロンプト (オプション)
