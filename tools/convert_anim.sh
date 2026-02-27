#!/usr/bin/env bash
# Mixamo FBX → GLB converter with optional VRM-based retargeting.
#
# Usage:
#   ./tools/convert_anim.sh input.fbx output.glb [target.vrm]
#
# With VRM (recommended):
#   ./tools/convert_anim.sh ~/Downloads/Idle.fbx assets/anims/idle.glb assets/models/mascot.vrm
#
# Without VRM (source armature as-is):
#   ./tools/convert_anim.sh ~/Downloads/Idle.fbx assets/anims/idle.glb

set -e

INPUT="${1:?Usage: $0 input.fbx output.glb [target.vrm]}"
OUTPUT="${2:?Usage: $0 input.fbx output.glb [target.vrm]}"
VRM="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Converting: $INPUT → $OUTPUT"
if [ -n "$VRM" ]; then
    echo "Target VRM: $VRM"
fi

blender --background \
        --python "$SCRIPT_DIR/mixamo_to_glb.py" \
        -- "$INPUT" "$OUTPUT" ${VRM:+"$VRM"}

echo "Done!"
