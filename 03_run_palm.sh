#!/bin/bash

CODE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname $CODE_DIR)
PALM_DIR=$ROOT_DIR/outputs/voxel_statistics/palm
TEMPLATE_DIR=$ROOT_DIR/data/templates

for DIR in $(ls $PALM_DIR/* -d); do

	palm -i $DIR/input.nii -d $DIR/design.csv -t $DIR/contrast.csv -m $TEMPLATE_DIR/MNI152_T1_2mm_Brain_Mask.nii -n 1000 -twotail -saveglm -saveparametric -logp -T -fdr -o $DIR/results

done

