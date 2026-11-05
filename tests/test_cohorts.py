import numpy as np
import pandas as pd
import pytest

from mimic_mortality.cohorts import (admission_type_group, as_missing, build_hosp_cohort,
                                     build_icu_cohort, race_group)


def _by_subject(df):
    return df.set_index("subject_id")


def test_icu_first_admission_and_first_stay(small_mimic):
    df = _by_subject(build_icu_cohort(small_mimic))
    assert df.index.is_unique
    # earliest admission, then the earliest ICU stay by intime (not the lowest stay_id)
    assert df.loc[1, "hadm_id"] == 101 and df.loc[1, "stay_id"] == 1002
    assert df.loc[1, "first_careunit"] == "Medical Intensive Care Unit (MICU)"
    assert df.loc[1, "hospital_expire_flag"] == 0
    # admissions without an ICU stay are skipped
    assert df.loc[2, "hadm_id"] == 202 and df.loc[2, "hospital_expire_flag"] == 1
    # the child admission is dropped and the first adult one is used
    assert df.loc[3, "hadm_id"] == 302 and df.loc[3, "age"] == 23
    # ages above 89 stay capped at 91
    assert df.loc[4, "age"] == 91
    assert df.loc[4, "race_group"] == "Other"
    # equal admittime: the lower hadm_id wins, whatever the file order
    assert df.loc[5, "hadm_id"] == 501 and df.loc[5, "stay_id"] == 5001
    assert df.loc[5, "hospital_expire_flag"] == 0


def test_hosp_first_admission(small_mimic):
    df = _by_subject(build_hosp_cohort(small_mimic))
    assert df.index.is_unique
    assert df.loc[1, "hadm_id"] == 101
    assert df.loc[2, "hadm_id"] == 201          # no ICU stay needed
    assert df.loc[2, "admission_type_group"] == "ELECTIVE"
    assert df.loc[2, "insurance_group"] == "Other"         # 'No charge'
    assert pd.isna(df.loc[2, "marital_status_group"])      # missing stays missing
    assert df.loc[2, "hospital_expire_flag"] == 0
    assert df.loc[3, "hadm_id"] == 302 and df.loc[3, "age_band"] == "18-44"
    assert df.loc[4, "age_band"] == "75+"
    assert df.loc[5, "hadm_id"] == 501


def test_mimic_iv_1_0_layout(v1_mimic):
    df = _by_subject(build_hosp_cohort(v1_mimic))
    assert df.loc[2, "race_group"] == "Black" and df.loc[2, "hadm_id"] == 201
    icu = _by_subject(build_icu_cohort(v1_mimic))
    assert icu.loc[1, "stay_id"] == 1002


@pytest.mark.parametrize("build", [build_icu_cohort, build_hosp_cohort])
def test_no_outcome_columns(small_mimic, build):
    cols = set(build(small_mimic).columns)
    for leaky in ("dischtime", "deathtime", "discharge_location", "outtime", "los",
                  "last_careunit", "dod", "edouttime"):
        assert leaky not in cols


def test_missing_markers_become_nan():
    s = pd.Series(["MARRIED", "?", None, "INFORMATION NOT AVAILABLE", " ", np.nan, "English "])
    out = as_missing(s)
    assert out.isna().tolist() == [False, True, True, True, True, True, False]
    assert out.iloc[6] == "English"


@pytest.mark.parametrize("raw, group", [
    ("WHITE", "White"), ("WHITE - EASTERN EUROPEAN", "White"),
    ("BLACK/CAPE VERDEAN", "Black"), ("HISPANIC OR LATINO", "Hispanic"),
    ("HISPANIC/LATINO - PUERTO RICAN", "Hispanic"), ("ASIAN - ASIAN INDIAN", "Asian"),
    ("SOUTH AMERICAN", "Other"), ("PORTUGUESE", "Other"),
    ("MULTIPLE RACE/ETHNICITY", "Other"), ("UNKNOWN", "Other"), (None, "Other"),
])
def test_race_group(raw, group):
    assert race_group(raw) == group


@pytest.mark.parametrize("raw, group", [
    ("EW EMER.", "EMERGENCY"), ("DIRECT EMER.", "EMERGENCY"), ("URGENT", "URGENT"),
    ("ELECTIVE", "ELECTIVE"), ("SURGICAL SAME DAY ADMISSION", "ELECTIVE"),
    ("AMBULATORY OBSERVATION", "OBSERVATION"), ("EU OBSERVATION", "OBSERVATION"),
    ("OBSERVATION ADMIT", "OBSERVATION"), ("DIRECT OBSERVATION", "OBSERVATION"),
])
def test_admission_type_group(raw, group):
    assert admission_type_group(raw) == group


def test_missing_folder_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="hosp/ and icu/"):
        build_hosp_cohort(tmp_path)
