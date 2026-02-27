#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?Usage: $0 input.fbx output.vrma target.vrm}"
OUTPUT="${2:?Usage: $0 input.fbx output.vrma target.vrm}"
VRM="${3:?Usage: $0 input.fbx output.vrma target.vrm}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Converting: $INPUT -> $OUTPUT"
echo "Target VRM: $VRM"

blender --background \
  --python "$SCRIPT_DIR/mixamo_to_vrma.py" \
  -- "$INPUT" "$OUTPUT" "$VRM"

echo "Done!"
