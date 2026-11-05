#!/bin/bash
# 02 — Stack the per-patient images of every PALM variant into input.nii.
# Variants with identical subject lists share one file (symlink). Run after 01_prepare_palm.ipynb.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="${LNM_PROJECT_ROOT:-$(python -c 'import sys; sys.path.insert(0, "utils"); import config; print(config.PROJECT_ROOT)')}"
PALM_DIR="$ROOT/output/voxel_statistics/palm"
declare -A DONE   # md5(input_paths.txt) -> input.nii
while IFS= read -r LIST; do
    DIR=$(dirname "$LIST")
    [ -e "$DIR/input.nii" ] && continue
    MD5=$(md5sum "$LIST" | awk '{print $1}')
    if [ -n "${DONE[$MD5]:-}" ]; then
        ln -s "${DONE[$MD5]}" "$DIR/input.nii"
    else
        python utils/concatenate_niftis.py "$LIST" "$DIR/input.nii"
        DONE[$MD5]="$DIR/input.nii"
    fi
done < <(find "$PALM_DIR" -name input_paths.txt | sort)
