"""Concatenates 3D NIfTI images into a 4D NIfTI file using parallel loading.

This script is a Python-based replacement for the `mrcat` command
for the specific purpose of concatenating 3D volumes into a 4D file.
File loading is parallelized across multiple CPU cores.
"""

import argparse
import os
import sys

import nibabel as nib
import numpy as np
from joblib import Parallel, delayed


def _load_nifti_worker(path: str, reference_shape: tuple) -> np.ndarray:
    """Worker function to load a single NIfTI, check its shape, and return its data."""
    try:
        img = nib.load(path)
        data = img.get_fdata()
        if data.shape != reference_shape:
            # We raise an error here, which will be caught by the main process
            raise ValueError(
                f"Shape mismatch! {path} has shape {data.shape}, "
                f"but expected {reference_shape}"
            )
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"NIfTI file not found at {path}")
    except Exception as e:
        # Re-raise other exceptions to be caught by the main process
        raise e


def concatenate_niftis(
    input_paths_file: str, output_file: str, n_jobs: int = -1
):
    """Loads NIfTI files in parallel, concatenates, and saves.

    Args:
        input_paths_file: Path to a .txt file containing a list of
                          paths to 3D NIfTI files (one per line).
        output_file: Path to save the resulting 4D NIfTI file.
        n_jobs: Number of CPU cores to use (-1 for all available cores).

    """
    print("--- Parallel NIfTI Concatenation ---")
    print(f"Reading image list from: {input_paths_file}")

    # 1. Read and clean file paths from the input text file
    try:
        with open(input_paths_file) as f:
            paths = [line.strip() for line in f.readlines() if line.strip()]
    except FileNotFoundError:
        print(
            f"Error: Input file not found at {input_paths_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    if not paths:
        print(
            f"Error: No file paths found in {input_paths_file}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(paths)} images to concatenate. Using {n_jobs} core(s).")

    # 2. Load the FIRST image serially to establish reference properties
    try:
        first_img = nib.load(paths[0])
        first_data = first_img.get_fdata()
        affine = first_img.affine
        header = first_img.header
        reference_shape = first_data.shape
        print(f"Reference 3D shape set to: {reference_shape}")
    except Exception as e:
        print(f"Error loading first file {paths[0]}: {e}", file=sys.stderr)
        sys.exit(1)

    image_data_list = [first_data]
    remaining_paths = paths[1:]

    # 3. Load the REST of the images in parallel
    if remaining_paths:
        try:
            print(
                f"Loading remaining {len(remaining_paths)} images in parallel..."
            )
            parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
                delayed(_load_nifti_worker)(path, reference_shape)
                for path in remaining_paths
            )
            image_data_list.extend(parallel_results)
        except Exception as e:
            print(f"Error during parallel file loading: {e}", file=sys.stderr)
            sys.exit(1)

    # 4. Stack data into a 4D array
    try:
        concatenated_data = np.stack(image_data_list, axis=-1)
    except Exception as e:
        print(f"Error during numpy stacking: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Final 4D data shape: {concatenated_data.shape}")

    # 5. Create and save the new 4D NIfTI image
    try:
        output_img = nib.Nifti1Image(concatenated_data, affine, header)
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        nib.save(output_img, output_file)
        print(f"✅ Successfully concatenated and saved to: {output_file}")

    except Exception as e:
        print(
            f"Error saving NIfTI file to {output_file}: {e}", file=sys.stderr
        )
        sys.exit(1)


def main():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Concatenate 3D NIfTI files into a 4D file in parallel.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "input_paths_file",
        type=str,
        help="Path to the .txt file listing the 3D NIfTI files to concatenate.",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to save the final 4D NIfTI file.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of CPU cores to use. -1 means all available (default: -1).",
    )
    args = parser.parse_args()

    concatenate_niftis(args.input_paths_file, args.output_file, args.n_jobs)


if __name__ == "__main__":
    main()
