# MIMIC-IV data preprocess: in-hospital mortality

Extracts in-hospital mortality cohorts from MIMIC-IV, one row per patient, with a patient-level train/val/test split and features that are fitted on the training split only.

This repository contains code only. MIMIC-IV is credentialed data: each user needs their own PhysioNet access and must follow the PhysioNet data use agreement. Never commit raw tables or extracted outputs. The extractor refuses to write its output inside this repository, and the `.gitignore` also excludes `outputs/`, `*.csv`, `*.csv.gz` and `*.parquet`.

## Install

```bash
pip install -r requirements.txt      # numpy, pandas, pyarrow, scikit-learn, pytest
```

Extraction runs on CPU with pandas, takes a few seconds, and needs about 0.5 GB of RAM.

## Run

```bash
python -m mimic_mortality --cohort icu  --mimic-dir /path/to/physionet.org/files/mimiciv/3.0 --out /path/outside/repo/icu_3.0
python -m mimic_mortality --cohort hosp --mimic-dir /path/to/physionet.org/files/mimiciv/3.0 --out /path/outside/repo/hosp_3.0
```

`--mimic-dir` is the MIMIC-IV version folder: the one that contains `hosp/` and `icu/` (MIMIC-IV 1.0 has `core/` instead of `hosp/`; both layouts are found). You can set `MIMIC_IV_DIR` instead of passing it. Both `.csv` and `.csv.gz` tables are read.

Options:

| Option | Default | Meaning |
|---|---|---|
| `--out DIR` | required | new output folder; must not exist and must lie outside this repository |
| `--seed` | 42 | split seed |
| `--test-size` | 0.2 | fraction of patients in the test split |
| `--val-size` | 0.1 | fraction of all patients in the validation split; 0 turns it off |
| `--missing-as-category` | off | give missing categorical values their own one-hot level (see below) |

The code was run on MIMIC-IV 3.0 and 2.2 for both cohorts, on 2.0 for the icu cohort, and on 1.0 for the hosp cohort. The local 1.0 copy had no ICU tables, so the icu cohort on 1.0 is covered by the tests only.

## Cohorts

Both cohorts use the label `hospital_expire_flag` of the selected admission (1 = died before hospital discharge). Age at admission is `anchor_age + year(admittime) - anchor_year`, capped at 91. Only adult admissions (18 or older) are considered, and each patient's earliest adult admission is kept, with ties on `admittime` broken by the lower `hadm_id`.

| | `icu` | `hosp` |
|---|---|---|
| Unit | first adult hospital admission that has an ICU stay; within it, the first ICU stay by `intime` | first adult hospital admission, ICU or not |
| Tables | admissions, patients, icustays | admissions, patients |
| Features | `age` (standardised); one-hot `admission_type`, `admission_location`, `insurance`, `marital_status`, `language`, `first_careunit` | one-hot `admission_type_group` (EMERGENCY, URGENT, ELECTIVE, OBSERVATION, OTHER), `insurance_group` (Medicare, Medicaid, Private, Other), `marital_status_group` (MARRIED, SINGLE, WIDOWED, DIVORCED, OTHER) |
| One-hot levels | learned from the training rows, so they differ between MIMIC-IV versions | fixed: always 14 columns, some of which can be all zero |
| Protected (`s_j`) | `s_0` sex, `s_1` race group | `s_0` sex, `s_1` race group, `s_2` age band (18-44=0, 45-64=1, 65-74=2, 75+=3) |
| Origin | ICLR27 `intersectional_extraction/code/build_mimic_cache.py` | synthetic-privacy `psyn/data/loaders.py` |

Codes:

- Sex is M=0 and F=1. The synthetic-privacy loader used F=0, M=1, so flip it when comparing with that project.
- Race groups are White=0, Black=1, Hispanic=2, Asian=3, Other=4, assigned by the prefix of the MIMIC race string. SOUTH AMERICAN, PORTUGUESE, multiple race, unknown and declined all map to Other.
- Protected attributes are never used as features.

Version notes:

- MIMIC-IV 1.0 and 2.x have no `Private` insurance value, so on those versions `insurance_group=Other` includes privately insured patients.
- In 1.0 and 2.x, `language` is only `ENGLISH` or `?`, and `?` counts as missing.
- `mimic_version` in `metadata.json` is the name of the `--mimic-dir` folder.

## Leakage rules

- Columns recorded at or after discharge are never read: `dischtime`, `deathtime`, `discharge_location`, `edouttime`, `outtime`, `los`, `last_careunit`, `dod`.
- Each patient has exactly one row, so the label-stratified split is a patient-level split. The code refuses to split a table with repeated `subject_id`.
- The split is drawn before any fitting. The imputers, scaler and one-hot encoder are fitted on training rows only. A category that appears only in validation or test rows becomes an all-zero one-hot block.
- **Missing values are imputed, not encoded.** MIMIC-IV does not record when marital status, language or insurance were entered, and they are missing far more often for patients who died. In MIMIC-IV 3.0 admissions the base in-hospital mortality is 2.2%, but it is 12.0% when marital status is missing and 15.6% when language is missing. A "missing" one-hot level would therefore carry outcome information.
  - By default, missing values and the placeholders `?` and `INFORMATION NOT AVAILABLE` are filled with the training median (numeric) or the training mode (categorical).
  - `--missing-as-category` restores the older behaviour, where missing values get their own level. Use it only to reproduce earlier results.
- `metadata.json` reports, for the training split, the share of missing values per feature for each label, and the single feature most correlated with the label. Check both before trusting a model.

## Output

Each run reserves the `--out` name, writes into a temporary folder next to it, and moves that folder into place only when every file is written. A failed run removes both (parent folders created for `--out` stay). An existing folder or file is never overwritten. Split sizes and the `--out` location are checked before any table is read.

| File | Contents |
|---|---|
| `cohort.parquet` | one row per patient: ids (`subject_id`, `hadm_id`, and `stay_id` for icu), admission time, raw and grouped columns (missing values kept as missing), protected codes, `hospital_expire_flag`, `split` |
| `train.parquet`, `val.parquet`, `test.parquet` | model-ready `f_0..f_k` features (float32), `label`, `s_0..` protected codes. Rows follow the order of `cohort.parquet` within each split |
| `metadata.json` | MIMIC-IV version, cohort definition, split sizes, mortality rate per split, feature names, protected-code maps, training cell sizes, missingness by label |
| `manifest.json` | command, arguments, sha256 of each input table, git commit, package versions, and sha256 and row count of every output file. Input hashes are compared with PhysioNet's `SHA256SUMS.txt`, which lists the `.csv.gz` files, so a decompressed `.csv` copy shows `null` (not checked) |

Load a split:

```python
import pandas as pd
train = pd.read_parquet("/path/outside/repo/icu_3.0/train.parquet")
X = train.filter(regex=r"^f_").to_numpy()
y = train["label"].to_numpy()
S = train.filter(regex=r"^s_").to_numpy()
```

## Tests

```bash
python -m pytest
```

The tests use a small made-up MIMIC-shaped folder, in both the 2.x/3.x and the 1.0 layout. They check:

- admission and ICU-stay selection, including ties;
- the age rules;
- race and admission-type grouping;
- missing-value handling;
- split disjointness and train-only fitting;
- alignment of labels, protected codes and one-hot features with the cohort table;
- the no-overwrite and failed-run cleanup behaviour;
- the `--out` and split-size checks.

## License

The code is released under the MIT License (see `LICENSE`). The license covers this code only, not MIMIC-IV, which remains under the PhysioNet Credentialed Health Data License and data use agreement.
