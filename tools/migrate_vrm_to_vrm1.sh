#!/usr/bin/env bash
set -euo pipefail

INPUT="${1:?Usage: $0 input.vrm output.vrm [backup.vrm]}"
OUTPUT="${2:?Usage: $0 input.vrm output.vrm [backup.vrm]}"
BACKUP="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Input:  $INPUT"
echo "Output: $OUTPUT"
if [ -n "$BACKUP" ]; then
  echo "Backup: $BACKUP"
else
  echo "Backup: auto"
fi

if [ -n "$BACKUP" ]; then
  blender --background \
    --python "$SCRIPT_DIR/migrate_vrm_to_vrm1.py" \
    -- "$INPUT" "$OUTPUT" "$BACKUP"
else
  blender --background \
    --python "$SCRIPT_DIR/migrate_vrm_to_vrm1.py" \
    -- "$INPUT" "$OUTPUT"
fi

echo "Done!"
