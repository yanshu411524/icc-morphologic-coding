# Morphologic coding and intrahepatic cholangiocarcinoma incidence, 1998–2017

Analysis code for a study of how much of the apparent 1998–2017 rise in intrahepatic cholangiocarcinoma (ICC) incidence is produced by improving morphologic specificity in cancer registries.

Registries that leave fewer liver cancers coded as unspecified carcinoma or unspecified morphology over time inflate the apparent trend of every specified subtype, including ICC. The scripts reallocate those unspecified cases within registry–year–sex–age strata, recompute world-standardised rates, and estimate how much of the observed rise that reallocation removes.

## Scripts

Run from the repository root, in this order:

```bash
python3 analysis/24_ci5_morphology_panel.py
python3 analysis/25_morphology_redistribution.py
python3 analysis/26_morph_trends.py
python3 analysis/27_seer_validation.py
python3 analysis/39_revision_sensitivity.py
FIGURE_FONT=arial python3 analysis/30_figures.py
```

| Script | Role |
|---|---|
| `24_ci5_morphology_panel.py` | Build the CI5plus liver-cancer morphology panel |
| `25_morphology_redistribution.py` | Reallocate unspecified cases (primary model and bounds) |
| `26_morph_trends.py` | EAPC, joinpoint regression, and bootstrap uncertainty |
| `27_seer_validation.py` | SEER confirmation and death-certificate-only checks |
| `39_revision_sensitivity.py` | Reallocate only part of unspecified morphology |
| `30_figures.py` | Figures from the tables above |

`39` reads `tables/revision_seer_dco.json`, which `27` writes. Death-certificate-only cases are identified from SEER **Type of Reporting Source**, not from Diagnostic Confirmation.

## Data

Registry data are not in this repository.

- **CI5plus detailed** files are available from the International Agency for Research on Cancer: https://ci5.iarc.fr  
  After extraction, place `data.csv` and `id_dict.csv` in `data/raw/ci5/_extracted/CI5plus_Detailed/`.
- **SEER research data** are available to registered users through SEER*Stat: https://seer.cancer.gov/seerstat/  
  Export case listings as:
  - `data/raw/seer/seer8_1975-2023_hepatobiliary.csv`
  - `data/raw/seer/seer17_2000-2023_hepatobiliary.csv`

  The listings used here are malignant behaviour, known age, and primary sites C22.0, C22.1, C24.0, C24.1, C24.8, and C24.9, with Diagnostic Confirmation, ICD-O-3 histology, and Type of Reporting Source. Do not redistribute SEER case listings.

## Requirements

Python 3.9 or later, plus:

```bash
python3 -m pip install -r requirements.txt
```

`numpy` and `matplotlib` are required. The redistribution and panel scripts use only the Python standard library.

## Authors

Lei Feng and Zhen You, West China Hospital, Sichuan University.
