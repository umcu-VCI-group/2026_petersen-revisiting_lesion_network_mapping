#!/bin/bash

# Concatenates 3D NIfTI images into a 4D NIfTI file to use with PALM.

# -----------------------------------------
CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname $CODE_DIR)
PALM_DIR=$ROOT_DIR/outputs/voxel_statistics/palm

N_JOBS=-1

# -----------------------------------------
echo $(ls $PALM_DIR)

CONCAT_SCRIPT="$CODE_DIR/utils/concatenate_niftis.py"

for DIR in $(ls $PALM_DIR/* -d);do 
    
    echo "--- Processing $DIR ---"

    INPUT_PATHS_FILE="$DIR/input_paths.txt"
    OUTPUT_FILE="$DIR/input.nii"

    if [ -f "$INPUT_PATHS_FILE" ]; then
        echo "Running concatenation for $OUTPUT_FILE..."
    python "$CONCAT_SCRIPT" "$INPUT_PATHS_FILE" "$OUTPUT_FILE" --n_jobs $N_JOBS
    else
        echo "Warning: $INPUT_PATHS_FILE not found."
    fi

done

