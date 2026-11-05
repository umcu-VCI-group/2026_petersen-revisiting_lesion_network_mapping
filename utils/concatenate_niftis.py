"""Stack the 3D images listed in input_paths.txt into one 4D NIfTI (parallel loading).

Usage: python utils/concatenate_niftis.py input_paths.txt input.nii [--n_jobs -1]
"""
import argparse

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_paths_file"); ap.add_argument("output_file")
    ap.add_argument("--n_jobs", type=int, default=-1)
    a = ap.parse_args()
    paths = [l.strip() for l in open(a.input_paths_file) if l.strip()]
    ref = nib.load(paths[0])
    data = Parallel(n_jobs=a.n_jobs)(delayed(lambda p: nib.load(p).get_fdata())(p) for p in paths)
    assert all(d.shape == ref.shape for d in data), "image shapes differ"
    nib.save(nib.Nifti1Image(np.stack(data, axis=-1), ref.affine, ref.header), a.output_file)
    print(f"{len(paths)} images -> {a.output_file}")


if __name__ == "__main__":
    main()
