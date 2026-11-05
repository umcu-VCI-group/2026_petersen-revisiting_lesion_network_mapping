# Analysis code for: "[Ensuring anatomical specificity in lesion network mapping: a multicentre study of 2,950 stroke patients](https://www.medrxiv.org/content/10.64898/2026.02.16.26346377)"

> [!TIP]
> We are currently developing [Lacuna](https://lacuna-py.readthedocs.io/en/latest/), an open-source Python package that implements the lesion network mapping workflow recommended in the manuscript, from lesion masks to lesion network maps, including group-level inference based on label permutation testing, within a single package. Lacuna was not used for the analyses reported here, which were performed with Lead-DBS and PALM (see below).

## Data layout

The code reads from and writes to a project root that holds `data/` and `output/`. It uses the first
parent folder of this repository that contains both, or the folder given by `LNM_PROJECT_ROOT`.

```
data/clean/clinical_data.csv                 clinical and cognitive data (one row per patient)
data/raw/lesion_masks/acuteinfarct_2mm/      binary lesion masks, MNI152NLin6Asym, 2 mm
data/processed/fLNM_2mm_leaddbs/             lesion-derived connectivity maps (Lead-DBS, GSP1000)
data/processed/lesion_overlap_schaefer1000_roi_data.csv   lesions x ROIs (simulation, real-lesion arm)
data/connectomes/*parcels1054*desc-groupconnectivity_connmatrix.tsv   ROI connectome (simulation)
data/degree_maps/GSP1000_degree_fisher_z.nii.gz
data/templates/MNI152_T1_2mm_Brain(_Mask).nii.gz   (+ uncompressed mask .nii for PALM)
data/atlases/                                Schaefer-1000, Tian S4, Buckner-7 (peak labels)
```

The clinical data and lesion data belong to the contributing cohorts and are not distributed. Please refer to the respective references for information on templates and atlases. For the degree map and GSP1000 connecitivity matrix please see [here](https://github.com/dutchconnectomelab/lesionnetworkmapping/tree/main/data/).

Lesion-based connectivity maps were derived using [Lead-DBS](https://www.lead-dbs.org/). 

| step | file | output |
|---|---|---|
| 1 | `01_prepare_palm.ipynb` | PALM designs for the seven analysis families |
| 2 | `02_concatenate_inputs.sh` | 4D inputs per design |
| 3 | `03_run_palm.sh` | PALM results (TFCE, 1,000 permutations, FDR) |
| 4 | `04_descriptives.ipynb` | Table 1, Supplementary figures S1, S2 |
| 5 | `05_brain_maps.ipynb` | Figure 1, Supplementary figures S3–S5, S8, S10, S12, S13, S17, S18, S23 |
| 6 | `06_spatial_correspondence.ipynb` | Figure 2, Supplementary figures S6, S7, S9, S15, S19, S20 |
| 7 | `07_peaks.ipynb` | Figure 3 |
| 8 | `08_onesample.ipynb` | Figure 4, Supplementary figures S11, S14 |
| 9 | `09_lnm_vs_vlsm.ipynb` | Figure 5, Supplementary figures S21, S22 |
| 10 | `10_simulation.ipynb` | Supplementary figure S16 |

Analysis families (`output/voxel_statistics/palm/<family>/<domain>/`):

| family | model |
|---|---|
| `lnm_nocov` | lesion network mapping, impaired vs unimpaired, cohort (primary analysis) |
| `lnm_volnihss` | + lesion volume + NIHSS |
| `lnm_xdomcov` | + impairment labels of the two other domains (attention/executive, language, verbal memory) |
| `vlsm_nocov`, `vlsm_volnihss` | voxel-based lesion-symptom mapping, same models |
| `onesample_cohort`, `onesample_volnihss` | one-sample t-test in impaired patients (sign flipping) |

Case-control designs permute impairment labels within cohort (exchangeability blocks).

## Environment

`conda env create -f environment.yml`. PALM (https://github.com/andersonwinkler/PALM) must be on the
path or set through `PALM_BIN`.

## Code structure

```
utils/config.py      paths, domains, analysis families, PALM file names, loaders
utils/mapstats.py    Dice, Pearson, cross-domain matrices, exact Mantel test, phi, mean maps
utils/brainplot.py   axial slice and cortical surface renders
utils/plotting.py    cross-domain comparison figure, overlap figure, correlation matrices
utils/figstyle.py    figure style, text conventions, panel layout
```
