#!/bin/bash
# 03 — Run PALM for every variant written by 01 (TFCE, 1000 permutations, FDR, two-tailed).
#   case-control designs: label permutation within cohort (-eb eb.csv -within)
#   one-sample designs:   sign flipping (-ise)
# Existing results are skipped. Optional argument: substring filter on the variant path.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="${LNM_PROJECT_ROOT:-$(python -c 'import sys; sys.path.insert(0, "utils"); import config; print(config.PROJECT_ROOT)')}"
PALM_DIR="$ROOT/output/voxel_statistics/palm"
MASK="$ROOT/data/templates/MNI152_T1_2mm_Brain_Mask.nii"
PALM_BIN="${PALM_BIN:-palm}"
FILTER="${1:-}"
while IFS= read -r DESIGN; do
    DIR=$(dirname "$DESIGN"); REL=${DIR#"$PALM_DIR/"}
    [[ -n "$FILTER" && "$REL" != *"$FILTER"* ]] && continue
    [ -e "$DIR/results_tfce_tstat_fdrp.nii" ] && continue
    EXTRA=()
    [ -f "$DIR/eb.csv" ] && EXTRA+=(-eb "$DIR/eb.csv" -within)
    [[ "$REL" == onesample_* ]] && EXTRA+=(-ise)
    echo "=== $REL ${EXTRA[*]}"
    "$PALM_BIN" -i "$DIR/input.nii" -d "$DESIGN" -t "$DIR/contrast.csv" -m "$MASK" \
        -n 1000 -twotail -saveglm -saveparametric -logp -T -fdr "${EXTRA[@]}" -o "$DIR/results"
done < <(find "$PALM_DIR" -name design.csv | sort)
