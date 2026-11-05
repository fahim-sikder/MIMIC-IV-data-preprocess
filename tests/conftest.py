"""A tiny MIMIC-IV-shaped folder for tests. All values are made up."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACES = ["WHITE", "WHITE - RUSSIAN", "BLACK/AFRICAN AMERICAN", "HISPANIC OR LATINO",
         "HISPANIC/LATINO - CUBAN", "ASIAN - CHINESE", "SOUTH AMERICAN", "UNKNOWN"]


def _hand_written():
    """Patients 1-4 cover the selection rules checked in test_cohorts.py."""
    adm = [
        # s1: first admission has two ICU stays listed out of time order
        dict(subject_id=1, hadm_id=101, admittime="2150-01-01 08:00", race="WHITE",
             admission_type="EW EMER.", insurance="Medicare", marital_status="MARRIED",
             hospital_expire_flag=0),
        dict(subject_id=1, hadm_id=102, admittime="2152-06-01 08:00", race="WHITE",
             admission_type="ELECTIVE", insurance="Medicare", marital_status="MARRIED",
             hospital_expire_flag=1),
        # s2: first admission has no ICU stay; the second one does
        dict(subject_id=2, hadm_id=201, admittime="2140-03-01 10:00", race="BLACK/AFRICAN",
             admission_type="SURGICAL SAME DAY ADMISSION", insurance="No charge",
             marital_status=None, hospital_expire_flag=0),
        dict(subject_id=2, hadm_id=202, admittime="2141-03-01 10:00", race="BLACK/AFRICAN",
             admission_type="URGENT", insurance="Private", marital_status="SINGLE",
             hospital_expire_flag=1),
        # s3: first admission is before age 18; the later adult one is used
        dict(subject_id=3, hadm_id=301, admittime="2158-01-01 00:00", race="ASIAN",
             admission_type="EU OBSERVATION", insurance=None, marital_status=None,
             hospital_expire_flag=0),
        dict(subject_id=3, hadm_id=302, admittime="2165-01-01 00:00", race="ASIAN",
             admission_type="DIRECT EMER.", insurance="Medicaid", marital_status="SINGLE",
             hospital_expire_flag=0),
        # s4: anchor_age 91 group, admitted 5 years after anchor_year -> capped at 91
        dict(subject_id=4, hadm_id=401, admittime="2185-01-01 00:00", race="PORTUGUESE",
             admission_type="EW EMER.", insurance="Other", marital_status="WIDOWED",
             hospital_expire_flag=1),
        # s5: two admissions with the same admittime -> the lower hadm_id is used
        dict(subject_id=5, hadm_id=502, admittime="2170-02-02 02:00", race="WHITE",
             admission_type="URGENT", insurance="Medicare", marital_status="?",
             hospital_expire_flag=1),
        dict(subject_id=5, hadm_id=501, admittime="2170-02-02 02:00", race="WHITE",
             admission_type="ELECTIVE", insurance="Medicare", marital_status="SINGLE",
             hospital_expire_flag=0),
    ]
    pat = [
        dict(subject_id=1, gender="F", anchor_age=60, anchor_year=2150),
        dict(subject_id=2, gender="M", anchor_age=40, anchor_year=2140),
        dict(subject_id=3, gender="F", anchor_age=18, anchor_year=2160),
        dict(subject_id=4, gender="M", anchor_age=91, anchor_year=2180),
        dict(subject_id=5, gender="F", anchor_age=50, anchor_year=2170),
    ]
    icu = [
        dict(subject_id=1, hadm_id=101, stay_id=1001, intime="2150-01-03 08:00",
             first_careunit="Coronary Care Unit (CCU)"),
        dict(subject_id=1, hadm_id=101, stay_id=1002, intime="2150-01-01 09:00",
             first_careunit="Medical Intensive Care Unit (MICU)"),
        dict(subject_id=1, hadm_id=102, stay_id=1003, intime="2152-06-01 09:00",
             first_careunit="Surgical Intensive Care Unit (SICU)"),
        dict(subject_id=2, hadm_id=202, stay_id=2001, intime="2141-03-01 12:00",
             first_careunit="Trauma SICU (TSICU)"),
        dict(subject_id=3, hadm_id=301, stay_id=3001, intime="2158-01-01 02:00",
             first_careunit="Neuro Stepdown"),
        dict(subject_id=3, hadm_id=302, stay_id=3002, intime="2165-01-01 02:00",
             first_careunit="Neuro Stepdown"),
        dict(subject_id=4, hadm_id=401, stay_id=4001, intime="2185-01-01 03:00",
             first_careunit="Medical Intensive Care Unit (MICU)"),
        dict(subject_id=5, hadm_id=502, stay_id=5002, intime="2170-02-02 03:00",
             first_careunit="Neuro Stepdown"),
        dict(subject_id=5, hadm_id=501, stay_id=5001, intime="2170-02-02 04:00",
             first_careunit="Coronary Care Unit (CCU)"),
    ]
    return adm, pat, icu


def _bulk(n: int, start: int = 1000, seed: int = 0):
    """Extra patients, one or two admissions each, so splits can be stratified."""
    rng = np.random.default_rng(seed)
    adm, pat, icu = [], [], []
    for k in range(n):
        sid = start + k
        pat.append(dict(subject_id=sid, gender="F" if k % 2 else "M",
                        anchor_age=int(rng.integers(18, 92)), anchor_year=2150))
        for j in range(1 + k % 2):
            hid = sid * 10 + j
            adm.append(dict(subject_id=sid, hadm_id=hid,
                            admittime=f"{2150 + j}-0{1 + k % 9}-01 08:00",
                            race=RACES[k % len(RACES)], admission_type="EW EMER.",
                            insurance=["Medicare", "Private", None][k % 3],
                            marital_status=["MARRIED", None][k % 2],
                            hospital_expire_flag=int(k % 5 == 0)))
            icu.append(dict(subject_id=sid, hadm_id=hid, stay_id=hid * 10,
                            intime=f"{2150 + j}-0{1 + k % 9}-01 10:00",
                            first_careunit="Medical Intensive Care Unit (MICU)"))
    return adm, pat, icu


def write_mimic(root: Path, adm, pat, icu, gz: bool = True, v1_layout: bool = False) -> Path:
    """v1_layout: MIMIC-IV 1.0 keeps admissions/patients in core/ and names
    the race column `ethnicity`."""
    ext = ".csv.gz" if gz else ".csv"
    hosp = root / ("core" if v1_layout else "hosp")
    hosp.mkdir(parents=True)
    (root / "icu").mkdir()
    a = pd.DataFrame(adm)
    if v1_layout:
        a = a.rename(columns={"race": "ethnicity"})
    for col, val in [("admission_location", "EMERGENCY ROOM"), ("language", "English"),
                     ("dischtime", "2200-01-01 00:00"), ("deathtime", None),
                     ("discharge_location", "HOME")]:
        a[col] = val
    a.to_csv(hosp / f"admissions{ext}", index=False)
    p = pd.DataFrame(pat)
    p["anchor_year_group"] = "2008 - 2010"
    p["dod"] = None
    p.to_csv(hosp / f"patients{ext}", index=False)
    i = pd.DataFrame(icu)
    i["outtime"] = "2200-01-01 00:00"
    i["los"] = 1.0
    i["last_careunit"] = i["first_careunit"]
    i.to_csv(root / "icu" / f"icustays{ext}", index=False)
    return root


@pytest.fixture
def small_mimic(tmp_path) -> Path:
    return write_mimic(tmp_path / "mimiciv" / "3.0", *_hand_written())


@pytest.fixture
def v1_mimic(tmp_path) -> Path:
    return write_mimic(tmp_path / "mimiciv" / "1.0", *_hand_written(), v1_layout=True)


@pytest.fixture
def bulk_mimic(tmp_path) -> Path:
    adm, pat, icu = _hand_written()
    a2, p2, i2 = _bulk(300)
    return write_mimic(tmp_path / "mimiciv" / "3.0", adm + a2, pat + p2, icu + i2, gz=False)
