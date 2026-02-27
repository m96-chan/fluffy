#!/usr/bin/env bash
# Usage: ./tools/convert_anim.sh ~/ダウンロード/idle.fbx assets/anims/idle.glb

set -e

INPUT="${1:?Usage: $0 input.fbx output.glb}"
OUTPUT="${2:?Usage: $0 input.fbx output.glb}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Converting: $INPUT → $OUTPUT"

blender --background \
        --python "$SCRIPT_DIR/mixamo_to_glb.py" \
        -- "$INPUT" "$OUTPUT"

echo "Done!"
