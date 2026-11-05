"""Paths, analysis families, PALM file conventions and shared map statistics.

Usage from a notebook in the repository root:
    import sys; sys.path.insert(0, "utils")
    from config import *
    from mapstats import *
"""
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# --- layout ----------------------------------------------------------------------------------
# PROJECT_ROOT holds data/ and output/: set LNM_PROJECT_ROOT, otherwise the first ancestor of
# this repository that contains both folders, otherwise the repository itself.
_REPO = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(os.environ.get("LNM_PROJECT_ROOT") or next(
    (p for p in [_REPO, *_REPO.parents] if (p / "data").is_dir() and (p / "output").is_dir()), _REPO))
DATA_DIR = PROJECT_ROOT / "data"
PALM_DIR = PROJECT_ROOT / "output" / "voxel_statistics" / "palm"
AGG_DIR = PROJECT_ROOT / "output" / "voxel_statistics" / "aggregate_mean"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
TAB_DIR = PROJECT_ROOT / "output" / "tables"
for _d in (FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLINICAL_CSV = DATA_DIR / "clean" / "clinical_data.csv"
FLNM_DIR = DATA_DIR / "processed" / "fLNM_2mm_leaddbs"              # lesion-derived connectivity maps
LESION_DIR = DATA_DIR / "raw" / "lesion_masks" / "acuteinfarct_2mm"  # binary lesion masks (MNI, 2 mm)
DEGREE_MAP = DATA_DIR / "degree_maps" / "GSP1000_degree_fisher_z.nii.gz"
BRAIN_MASK = DATA_DIR / "templates" / "MNI152_T1_2mm_Brain_Mask.nii.gz"
BG_IMG = DATA_DIR / "templates" / "MNI152_T1_2mm_Brain.nii.gz"
# ROI-level inputs of the null simulation (Schaefer-1000 + Tian-54 = 1,054 regions)
CONNECTOME_GLOB = (DATA_DIR / "connectomes", "*parcels1054*desc-groupconnectivity_connmatrix.tsv")
CONNECTOME_MAT = DATA_DIR / "GSP1000_FC_Schaefer1000Melbourne54.mat"                   # legacy format (key "FC")
LESION_ROI_CSV = DATA_DIR / "processed" / "lesion_overlap_schaefer1000_roi_data.csv"   # patients x ROIs, binary

# --- cognitive domains (fixed display order) -------------------------------------------------
DOMAINS = {
    "cognition-aef_domain_score": "Attention / Executive function",
    "cognition-ps_domain_score": "Processing speed",
    "cognition-language_domain_score": "Language",
    "cognition-verbalmemory_domain_score": "Verbal memory",
    "cognition-vsfunctions_domain_score": "Visuospatial functions",
    "cognition-vsmemory_domain_score": "Visuospatial memory",
}
KEYS = list(DOMAINS)
NAMES = list(DOMAINS.values())
SH = dict(zip(KEYS, ["Attn/Exec", "Proc. speed", "Language", "Verbal mem.", "VS func.", "VS mem."]))
XDOM_KEYS = [KEYS[0], KEYS[2], KEYS[3]]   # domains adjusted for each other's impairment (S23)

# --- PALM analysis families --------------------------------------------------------------------
# Case-control (impaired vs unimpaired; cohort fixed effects + within-cohort exchangeability blocks)
LNM = "lnm_nocov"                # primary analysis: cohort only
LNM_FULL = "lnm_volnihss"        # + lesion volume + NIHSS
VLSM = "vlsm_nocov"              # voxel-based lesion-symptom mapping, cohort only
VLSM_FULL = "vlsm_volnihss"      # + lesion volume + NIHSS
LNM_XDOM = "lnm_xdomcov"         # + impairment labels of the two other strongly overlapping domains
# One-sample t-tests in impaired patients (sign flipping)
OS = "onesample_cohort"          # cohort only
OS_FULL = "onesample_volnihss"   # + lesion volume + NIHSS

# --- PALM output files (p-maps are -log10 p) -------------------------------------------------
F_TSTAT = "results_vox_tstat.nii"
F_COPE = "results_vox_cope.nii"
F_PERM_FDR = "results_tfce_tstat_fdrp.nii"        # label permutation / sign flipping, TFCE, FDR
F_PERM_FWE = "results_tfce_tstat_fwep.nii"
F_PARAM_FDR = "results_vox_tstat_fdrparap.nii"    # parametric, voxelwise FDR
THR = -np.log10(0.05)


def result_path(family, key, fname):
    return PALM_DIR / family / key / fname


def has_results(family, key):
    """True once PALM has finished a variant (the TFCE-FDR map is written last)."""
    return result_path(family, key, F_PERM_FDR).exists()


def load_mask():
    """(brain_bool_3d, affine, background_path)."""
    m = nib.load(BRAIN_MASK)
    return m.get_fdata() > 0, m.affine, str(BG_IMG)


def load_map(family, key, fname, brain=None):
    arr = nib.load(result_path(family, key, fname)).get_fdata()
    return np.where(brain, arr, 0.0) if brain is not None else arr


def tmap(family, key, brain):
    """In-brain signed T (NaN outside the brain)."""
    return np.where(brain, nib.load(result_path(family, key, F_TSTAT)).get_fdata(), np.nan)


def sigmap(family, key, brain, pmap=F_PERM_FDR):
    """In-brain significance mask, p < 0.05."""
    return (load_map(family, key, pmap, brain=brain) >= THR) & brain


def sample_ids(family, key):
    """Subject IDs of one PALM variant, in design order."""
    lines = (PALM_DIR / family / key / "input_paths.txt").read_text().split()
    return [Path(l).name.split("_infarct")[0] for l in lines if l]
