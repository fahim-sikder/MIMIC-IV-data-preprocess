"""MIMIC-IV in-hospital mortality cohorts, one row per patient.

Two cohorts, both labelled with `hospital_expire_flag` (death before hospital
discharge) of the selected admission:

  icu   First adult hospital admission with an ICU stay, per patient
        (admissions x patients x icustays). Within that admission the first
        ICU stay is used. Adapted from ICLR27/intersectional_extraction.

  hosp  First adult hospital admission per patient, ICU or not. Categories
        are collapsed to small fixed lists. Adapted from synthetic-privacy.

Leakage rules. Columns recorded at or after discharge are never read:
dischtime, deathtime, discharge_location, edouttime, last_careunit, outtime,
los, dod. The label column is used for the label only. Each patient
contributes exactly one row, so any row-level split is a patient-level split.

Missing values are kept as missing (NaN) in the cohort table. MIMIC-IV does
not record when marital status, language or insurance were entered, and these
fields are missing far more often for patients who died, so a "missing"
category would carry outcome information. Placeholder values that mean
"unknown" ('?' and 'INFORMATION NOT AVAILABLE') are also set to missing.
How missing values become features is decided in features.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .io import read_table

LABEL = "hospital_expire_flag"
ADULT_AGE = 18
MAX_AGE = 91  # MIMIC-IV reports every age above 89 as 91

RACE_GROUPS = ("White", "Black", "Hispanic", "Asian", "Other")
SEX_CODES = {"M": 0, "F": 1}
AGE_BANDS = ("18-44", "45-64", "65-74", "75+")

ADMISSION_TYPE_GROUPS = ("EMERGENCY", "URGENT", "ELECTIVE", "OBSERVATION", "OTHER")
INSURANCE_GROUPS = ("Medicare", "Medicaid", "Private", "Other")
MARITAL_GROUPS = ("MARRIED", "SINGLE", "WIDOWED", "DIVORCED", "OTHER")

MISSING_MARKERS = ("?", "INFORMATION NOT AVAILABLE", "")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def race_group(raw) -> str:
    """Collapse MIMIC-IV's race strings to five groups by prefix.

    'WHITE - RUSSIAN' -> White, 'BLACK/CAPE VERDEAN' -> Black,
    'HISPANIC/LATINO - CUBAN' and 'HISPANIC OR LATINO' -> Hispanic,
    'ASIAN - CHINESE' -> Asian. Everything else is Other, including
    SOUTH AMERICAN, PORTUGUESE, MULTIPLE RACE/ETHNICITY, UNKNOWN,
    UNABLE TO OBTAIN and PATIENT DECLINED TO ANSWER.
    """
    up = str(raw).strip().upper()
    if up.startswith("WHITE"):
        return "White"
    if up.startswith("BLACK"):
        return "Black"
    if up.startswith("HISPANIC") or up.startswith("LATINO"):
        return "Hispanic"
    if up.startswith("ASIAN"):
        return "Asian"
    return "Other"


def age_at_admission(df: pd.DataFrame) -> pd.Series:
    """Age in whole years at `admittime`.

    MIMIC-IV gives `anchor_age` in the shifted `anchor_year`, so the age at an
    admission is anchor_age + (admission year - anchor_year), capped at 91.
    """
    years = df["admittime"].dt.year - df["anchor_year"]
    return (df["anchor_age"] + years).clip(upper=MAX_AGE).astype(float)


def age_band(age: pd.Series) -> pd.Series:
    out = np.select([age < 45, age < 65, age < 75], list(AGE_BANDS[:3]), default=AGE_BANDS[3])
    return pd.Series(out, index=age.index)


def as_missing(values: pd.Series) -> pd.Series:
    """Strings with object dtype; placeholder values for 'unknown' become NaN."""
    s = values.astype(object).where(values.notna(), np.nan)
    stripped = s.map(lambda v: v.strip() if isinstance(v, str) else v)
    return stripped.where(~stripped.isin(MISSING_MARKERS), np.nan)


def _group(values: pd.Series, keep, other: str) -> pd.Series:
    """Keep the values in `keep`, map other present values to `other`, leave NaN as NaN."""
    out = values.where(values.isin(keep), other)
    return out.where(values.notna(), np.nan)


def _codes(values: pd.Series, categories) -> pd.Series:
    cat = pd.Categorical(values, categories=list(categories))
    if (cat.codes < 0).any():
        bad = sorted(set(values[cat.codes < 0].astype(str)))[:10]
        raise ValueError(f"values outside {list(categories)}: {bad}")
    return pd.Series(cat.codes.astype(np.int64), index=values.index)


def _check_categories(values: pd.Series, categories) -> None:
    present = values.dropna()
    bad = sorted(set(present[~present.isin(categories)].astype(str)))
    if bad:
        raise ValueError(f"values outside {list(categories)}: {bad[:10]}")


def _sex_code(gender: pd.Series) -> pd.Series:
    codes = gender.astype(str).str.upper().map(SEX_CODES)
    if codes.isna().any():
        bad = sorted(set(gender[codes.isna()].astype(str)))[:10]
        raise ValueError(f"unexpected gender values: {bad}")
    return codes.astype(np.int64)


def _first_row_per_patient(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Keep the earliest row per subject_id by `order`, with a stable sort.

    Row-wise selection on purpose: groupby().first() would take the first
    non-null value of each column separately, mixing admissions.
    """
    df = df.sort_values(["subject_id", *order], kind="mergesort")
    return df.drop_duplicates("subject_id", keep="first").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cohort: first adult ICU-linked admission (ICLR27)
# ---------------------------------------------------------------------------

def build_icu_cohort(mimic_dir: str | Path) -> pd.DataFrame:
    adm = read_table(
        mimic_dir, "hosp", "admissions",
        ["subject_id", "hadm_id", "admittime", "admission_type", "admission_location",
         "insurance", "language", "marital_status", "race", LABEL],
        parse_dates=["admittime"],
    )
    pat = read_table(mimic_dir, "hosp", "patients",
                     ["subject_id", "gender", "anchor_age", "anchor_year"])
    icu = read_table(mimic_dir, "icu", "icustays",
                     ["subject_id", "hadm_id", "stay_id", "intime", "first_careunit"],
                     parse_dates=["intime"])

    df = adm.merge(icu, on=["subject_id", "hadm_id"], how="inner", validate="one_to_many")
    df = df.merge(pat, on="subject_id", how="inner", validate="many_to_one")
    df["age"] = age_at_admission(df)
    df = df[df["age"] >= ADULT_AGE]
    # earliest admission, then the earliest ICU stay within it
    df = _first_row_per_patient(df, ["admittime", "hadm_id", "intime", "stay_id"])

    for col in ("admission_type", "admission_location", "insurance",
                "marital_status", "language", "first_careunit"):
        df[col] = as_missing(df[col])
    df["race_group"] = df["race"].map(race_group)
    df["sex_code"] = _sex_code(df["gender"])
    df["race_code"] = _codes(df["race_group"], RACE_GROUPS)
    df[LABEL] = df[LABEL].astype(np.int64)

    cols = ["subject_id", "hadm_id", "stay_id", "admittime", "intime", "age", "gender",
            "race", "race_group", "sex_code", "race_code", "admission_type",
            "admission_location", "insurance", "marital_status", "language",
            "first_careunit", LABEL]
    return df[cols].sort_values("subject_id", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Cohort: first adult hospital admission (synthetic-privacy)
# ---------------------------------------------------------------------------

def admission_type_group(raw) -> str:
    """Collapse MIMIC-IV's nine admission types to five groups.

    SURGICAL SAME DAY ADMISSION is a scheduled admission, so it is ELECTIVE.
    """
    up = str(raw).upper()
    if "EMER" in up:
        return "EMERGENCY"
    if "URGENT" in up:
        return "URGENT"
    if "ELECTIVE" in up or "SURGICAL SAME DAY" in up:
        return "ELECTIVE"
    if "OBSERVATION" in up:
        return "OBSERVATION"
    return "OTHER"


def build_hosp_cohort(mimic_dir: str | Path) -> pd.DataFrame:
    adm = read_table(
        mimic_dir, "hosp", "admissions",
        ["subject_id", "hadm_id", "admittime", "admission_type", "insurance",
         "marital_status", "race", LABEL],
        parse_dates=["admittime"],
    )
    pat = read_table(mimic_dir, "hosp", "patients",
                     ["subject_id", "gender", "anchor_age", "anchor_year"])

    df = adm.merge(pat, on="subject_id", how="inner", validate="many_to_one")
    df["age"] = age_at_admission(df)
    df = df[df["age"] >= ADULT_AGE]
    df = _first_row_per_patient(df, ["admittime", "hadm_id"])

    for col in ("admission_type", "insurance", "marital_status"):
        df[col] = as_missing(df[col])
    df["race_group"] = df["race"].map(race_group)
    df["age_band"] = age_band(df["age"])
    df["admission_type_group"] = df["admission_type"].map(
        admission_type_group, na_action="ignore")
    df["insurance_group"] = _group(df["insurance"], INSURANCE_GROUPS[:3], "Other")
    df["marital_status_group"] = _group(df["marital_status"], MARITAL_GROUPS[:4], "OTHER")
    df["sex_code"] = _sex_code(df["gender"])
    df["race_code"] = _codes(df["race_group"], RACE_GROUPS)
    df["age_band_code"] = _codes(df["age_band"], AGE_BANDS)
    _check_categories(df["admission_type_group"], ADMISSION_TYPE_GROUPS)
    _check_categories(df["insurance_group"], INSURANCE_GROUPS)
    _check_categories(df["marital_status_group"], MARITAL_GROUPS)
    df[LABEL] = df[LABEL].astype(np.int64)

    cols = ["subject_id", "hadm_id", "admittime", "age", "age_band", "gender", "race",
            "race_group", "sex_code", "race_code", "age_band_code", "admission_type",
            "admission_type_group", "insurance", "insurance_group", "marital_status",
            "marital_status_group", LABEL]
    return df[cols].sort_values("subject_id", kind="mergesort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CohortSpec:
    name: str
    description: str
    build: Callable[[str | Path], pd.DataFrame]
    num_features: tuple[str, ...]
    cat_features: tuple[str, ...]
    protected: tuple[str, ...]          # integer-coded columns, written as s_0, s_1, ...
    protected_maps: dict = field(default_factory=dict)
    # Fixed one-hot levels per categorical feature. None: levels are learned
    # from the training rows (they then depend on the MIMIC-IV version).
    categories: dict | None = None
    # Level used for missing values with --missing-as-category.
    missing_levels: dict = field(default_factory=dict)


COHORTS = {
    "icu": CohortSpec(
        name="icu",
        description="First adult ICU-linked hospital admission per patient; "
                    "label = in-hospital death",
        build=build_icu_cohort,
        num_features=("age",),
        cat_features=("admission_type", "admission_location", "insurance",
                      "marital_status", "language", "first_careunit"),
        protected=("sex_code", "race_code"),
        protected_maps={"sex_code": SEX_CODES,
                        "race_code": {g: i for i, g in enumerate(RACE_GROUPS)}},
        categories=None,
        missing_levels={c: "UNKNOWN" for c in ("admission_type", "admission_location",
                                                "insurance", "marital_status", "language",
                                                "first_careunit")},
    ),
    "hosp": CohortSpec(
        name="hosp",
        description="First adult hospital admission per patient; "
                    "label = in-hospital death",
        build=build_hosp_cohort,
        num_features=(),
        cat_features=("admission_type_group", "insurance_group", "marital_status_group"),
        protected=("sex_code", "race_code", "age_band_code"),
        protected_maps={"sex_code": SEX_CODES,
                        "race_code": {g: i for i, g in enumerate(RACE_GROUPS)},
                        "age_band_code": {b: i for i, b in enumerate(AGE_BANDS)}},
        categories={"admission_type_group": ADMISSION_TYPE_GROUPS,
                    "insurance_group": INSURANCE_GROUPS,
                    "marital_status_group": MARITAL_GROUPS},
        missing_levels={"admission_type_group": "OTHER", "insurance_group": "Other",
                        "marital_status_group": "OTHER"},
    ),
}
