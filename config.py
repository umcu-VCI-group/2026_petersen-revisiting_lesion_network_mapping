"""Shared configuration for the plotting/analysis notebooks.

Encodes the analysis directory layout and the PALM output-file conventions so every
notebook resolves paths and significance maps identically.

Usage from a notebook in `analysis/`:
    import sys; sys.path.insert(0, "utils")
    from config import *
"""
from pathlib import Path
import numpy as np
import nibabel as nib

# --- project layout -------------------------------------------------------------------------
# Notebooks live in <ROOT>/analysis/. ROOT is the first ancestor holding both data/ and output/.
PROJECT_ROOT = next(
    p for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]
    if (p / "data").is_dir() and (p / "output").is_dir()
)
DATA_DIR = PROJECT_ROOT / "data"
PALM_DIR = PROJECT_ROOT / "output" / "voxel_statistics" / "palm"
FIG_DIR = PROJECT_ROOT / "output" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- cognitive domains (key -> pretty name), fixed display order ----------------------------
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

# --- PALM variant families --------------------------------------------------------------------
# group comparisons (impaired vs not-impaired; all carry an explicit intercept and retain
# cohort fixed effects + within-cohort exchangeability blocks):
FAM_LNM_NOCOV = "lnm_nocov"      # PRIMARY (main manuscript): LNM group comparison, NO volume cov
FAM_LNM_NIHSS = "lnm_volnihss"   # supplement: LNM group comparison, + NIHSS sensitivity
FAM_VLSM = "vlsm_volcov"         # supplement: VLSM group comparison (volume-adjusted)
FAM_ONESAMPLE = "onesample"      # impaired-cases one-sample LNM (symptom-blind benchmark)

# Manuscript ordering: the no-covariate LNM is the MAIN-TEXT primary; the rest are supplement.
CORE_FAMILY = FAM_LNM_NOCOV                       # main-manuscript analysis
SUPPLEMENT_FAMILIES = [FAM_LNM_NIHSS, FAM_VLSM]   # supplement variants
# LNM covariate layers, main-first (nocov primary, then volume+NIHSS sensitivity):
LNM_FAMILIES = [FAM_LNM_NOCOV, FAM_LNM_NIHSS]
# All group comparisons, main-first:
GROUP_FAMILIES = [FAM_LNM_NOCOV, FAM_LNM_NIHSS, FAM_VLSM]

# --- PALM output filenames (all p-maps are -log10(p) because PALM was run with -logp) --------
F_TSTAT = "results_vox_tstat.nii"            # signed voxelwise T-statistic
F_COPE = "results_vox_cope.nii"              # contrast estimate (beta)
F_PARAM_FDR = "results_vox_tstat_fdrparap.nii"   # PARAMETRIC voxelwise FDR  (-log10 p)
F_PARAM_UNC = "results_vox_tstat_uncparap.nii"   # parametric voxelwise uncorrected (-log10 p)
F_PERM_FDR = "results_tfce_tstat_fdrp.nii"       # PERMUTATION TFCE FDR      (-log10 p)
F_PERM_FWE = "results_tfce_tstat_fwep.nii"       # PERMUTATION TFCE FWE      (-log10 p)
F_PARAM_FWE = "results_vox_tstat_fwep.nii"       # (permutation vox FWE; parametric FWE n/a in PALM)

THR = -np.log10(0.05)            # significance threshold on the -log10(p) maps

# --- path + image helpers ---------------------------------------------------------------------
def result_path(family, domain_key, fname):
    """Absolute path to a PALM output file for one variant."""
    return PALM_DIR / family / domain_key / fname

def has_results(family, domain_key=None):
    """True only if PALM has FULLY finished a variant.
    """
    if domain_key is None:
        return all(result_path(family, k, F_PERM_FDR).exists() for k in KEYS)
    return result_path(family, domain_key, F_PERM_FDR).exists()

# alias with a clearer name; identical semantics
is_complete = has_results

def load_mask():
    """(brain_bool_3d, affine, background_path_str)."""
    mimg = nib.load(DATA_DIR / "templates" / "MNI152_T1_2mm_Brain_Mask.nii.gz")
    brain = mimg.get_fdata() > 0
    bg = str(DATA_DIR / "templates" / "MNI152_T1_2mm_Brain.nii.gz")
    return brain, mimg.affine, bg

def load_map(family, domain_key, fname, brain=None):
    """Load a PALM map, zeroing out-of-brain voxels when a brain mask is given."""
    arr = nib.load(result_path(family, domain_key, fname)).get_fdata()
    return np.where(brain, arr, 0.0) if brain is not None else arr

# degree map from https://github.com/dutchconnectomelab/lesionnetworkmapping/
DEGREE_MAP = DATA_DIR / "degree_maps" / "GSP1000_degree_fisher_z.nii.gz"
