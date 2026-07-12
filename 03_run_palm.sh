#!/bin/bash
# 03 — Run PALM for every prepared variant.
#
# Flags: -n 1000 -twotail -saveglm -saveparametric -logp -T -fdr
#   * GROUP comparisons (lnm_*, vlsm_*): LABEL PERMUTATION, restricted to WITHIN each cohort
#     via the per-variant eb.csv  (-eb eb.csv -within); designs also carry cohort fixed effects.
#   * ONE-SAMPLE (onesample/*): intercept-only design.
# A variant gets -eb automatically iff its directory contains eb.csv (written by 01 for groups).
#
# Idempotent: a variant whose results_*_tstat.nii already exists is skipped.
#
# Optional 1st arg = a path filter (substring) to run a subset, e.g. for parallelising across
# terminals:   ./03_run_palm.sh lnm_nocov      ./03_run_palm.sh onesample
#
# Run AFTER 02_concatenate_niis.sh.

set -uo pipefail

CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname "${CODE_DIR}")
PALM_DIR="$ROOT_DIR/output/voxel_statistics/palm"
TEMPLATE_DIR="$ROOT_DIR/data/templates"
MASK="$TEMPLATE_DIR/MNI152_T1_2mm_Brain_Mask.nii"

PALM_BIN="${PALM_BIN:-palm}"          # override with PALM_BIN=/path/to/palm if not on PATH
FILTER="${1:-}"                        # optional substring filter on the variant path

COMMON_FLAGS=(-n 1000 -twotail -saveglm -saveparametric -logp -T -fdr)

mapfile -t DESIGN_FILES < <(find "$PALM_DIR" -name design.csv | sort)

for DESIGN in "${DESIGN_FILES[@]}"; do
    DIR=$(dirname "$DESIGN")
    REL=${DIR#"$PALM_DIR/"}

    [ -n "$FILTER" ] && [[ "$REL" != *"$FILTER"* ]] && continue

    INPUT="$DIR/input.nii"
    CONTRAST="$DIR/contrast.csv"

    if [ ! -e "$INPUT" ]; then
        echo "[skip] $REL (no input.nii — run 02 first)"; continue
    fi
    if ls "$DIR"/results_*_tstat.nii >/dev/null 2>&1; then
        echo "[skip] $REL (results already present)"; continue
    fi

    # group comparisons use within-cohort label permutation (per-variant eb.csv);
    # the one-sample arm is intercept-only and reported with parametric inference.
    EXTRA=()
    [ -f "$DIR/eb.csv" ] && EXTRA+=(-eb "$DIR/eb.csv" -within)

    echo "=== PALM: $REL ${EXTRA[*]} ==="
    "$PALM_BIN" -i "$INPUT" -d "$DESIGN" -t "$CONTRAST" -m "$MASK" \
        "${COMMON_FLAGS[@]}" "${EXTRA[@]}" -o "$DIR/results"
done

echo "All requested PALM runs complete."
