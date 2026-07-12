#!/bin/bash
# 02 — Concatenate per-subject 3D images into a 4D input.nii for each PALM variant.
#
# Each concatenated 4D file is multi-GB, so we concatenate once
# per unique input_paths.txt (keyed by md5 of its contents) 
# and symlink the duplicates.
#
# Run AFTER 01_prepare_palm.ipynb. Existing/symlinked input.nii are skipped.

set -euo pipefail

CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname "${CODE_DIR}")
PALM_DIR="$ROOT_DIR/output/voxel_statistics/palm"
CONCAT_SCRIPT="$CODE_DIR/utils/concatenate_niftis.py"

# --- CHOOSE NUMBER OF CPU CORES (-1 = all) ---
N_JOBS=-1
# ---------------------------------------------

[ -f "$CONCAT_SCRIPT" ] || { echo "Error: concat script not found at $CONCAT_SCRIPT"; exit 1; }

declare -A CANONICAL  # md5 -> canonical input.nii path (first one concatenated)

# deterministic order so the canonical (real file) is stable across reruns
mapfile -t PATH_FILES < <(find "$PALM_DIR" -name input_paths.txt | sort)

echo "Found ${#PATH_FILES[@]} variants."

for INPUT_PATHS_FILE in "${PATH_FILES[@]}"; do
    DIR=$(dirname "$INPUT_PATHS_FILE")
    OUTPUT_FILE="$DIR/input.nii"
    REL=${DIR#"$PALM_DIR/"}

    if [ -e "$OUTPUT_FILE" ]; then
        echo "[skip] $REL (input.nii already exists)"
        continue
    fi

    MD5=$(md5sum "$INPUT_PATHS_FILE" | awk '{print $1}')

    if [ -n "${CANONICAL[$MD5]:-}" ]; then
        ln -s "${CANONICAL[$MD5]}" "$OUTPUT_FILE"
        echo "[link] $REL -> ${CANONICAL[$MD5]#"$PALM_DIR/"}"
    else
        echo "[calc] $REL ..."
        python "$CONCAT_SCRIPT" "$INPUT_PATHS_FILE" "$OUTPUT_FILE" --n_jobs "$N_JOBS"
        CANONICAL[$MD5]="$OUTPUT_FILE"
    fi
done