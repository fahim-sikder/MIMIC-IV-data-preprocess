import dataclasses
import json
import os
import warnings

import numpy as np
import pandas as pd
import pytest

from mimic_mortality.__main__ import main
from mimic_mortality.cohorts import COHORTS
from mimic_mortality.features import assign_splits, featurize
from mimic_mortality.io import sha256, write_new


def _toy(n=200):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "subject_id": np.arange(n),
        "x": rng.normal(10.0, 2.0, n),
        "c": rng.choice(["A", "B"], n, p=[0.7, 0.3]).astype(object),
        "s": rng.integers(0, 2, n),
        "y": (np.arange(n) % 4 == 0).astype(int),
    })


# ---------------------------------------------------------------- splits

def test_splits_are_disjoint_and_stratified():
    df = _toy()
    split = assign_splits(df, "y", test_size=0.2, val_size=0.1, seed=0)
    assert set(split) == {"train", "val", "test"}
    assert split.value_counts().to_dict() == {"train": 140, "test": 40, "val": 20}
    for s in ("train", "val", "test"):
        assert df.loc[split == s, "y"].mean() == pytest.approx(0.25, abs=0.03)


def test_split_rejects_repeated_patients():
    df = _toy()
    df.loc[1, "subject_id"] = 0
    with pytest.raises(ValueError, match="more than one row"):
        assign_splits(df, "y")


# ---------------------------------------------------------------- featurisation

def test_transforms_fit_on_train_only():
    df = _toy()
    df["split"] = assign_splits(df, "y", seed=0)
    df.loc[df["split"] == "test", "c"] = "C"          # category never seen in train
    out, names = featurize(df, ("x",), ("c",), ("s",), "y")
    train, test = out["train"], out["test"]
    assert train["f_0"].mean() == pytest.approx(0.0, abs=1e-5)
    assert train["f_0"].std(ddof=0) == pytest.approx(1.0, abs=1e-4)
    assert len(names) == 3                            # x, c=A, c=B
    assert (test[["f_1", "f_2"]].to_numpy() == 0).all()

    # changing val/test values must not change the training features
    df2 = df.copy()
    df2.loc[df2["split"] != "train", "x"] += 1000.0
    out2, _ = featurize(df2, ("x",), ("c",), ("s",), "y")
    pd.testing.assert_frame_equal(out["train"], out2["train"])


def test_missing_values_imputed_by_default():
    df = _toy()
    df["split"] = assign_splits(df, "y", seed=0)
    df.loc[df.index % 7 == 0, "c"] = np.nan
    out, names = featurize(df, (), ("c",), ("s",), "y")
    assert names == ["cat__c_A", "cat__c_B"]          # no level for "missing"
    miss = df["c"].isna().to_numpy()
    for split, part in out.items():
        rows = (df["split"] == split).to_numpy()
        # every missing value gets the training mode, A
        assert (part.loc[miss[rows], "f_0"] == 1).all()


def test_missing_as_category_reproduces_old_behaviour():
    df = _toy()
    df["split"] = assign_splits(df, "y", seed=0)
    df.loc[df.index % 7 == 0, "c"] = np.nan
    out, names = featurize(df, (), ("c",), ("s",), "y", missing="category",
                           missing_levels={"c": "UNKNOWN"})
    assert names == ["cat__c_A", "cat__c_B", "cat__c_UNKNOWN"]


def test_fixed_levels():
    df = _toy()
    df["split"] = assign_splits(df, "y", seed=0)
    with pytest.warns(UserWarning, match="no training rows"):
        out, names = featurize(df, (), ("c",), ("s",), "y", categories={"c": ("A", "B", "Z")})
    assert names == ["cat__c_A", "cat__c_B", "cat__c_Z"]
    assert (out["train"]["f_2"] == 0).all()
    with pytest.raises(ValueError, match="not in the fixed levels"):
        featurize(df, (), ("c",), ("s",), "y", categories={"c": ("A",)})


def test_feature_label_overlap_rejected():
    df = _toy()
    df["split"] = assign_splits(df, "y", seed=0)
    with pytest.raises(ValueError, match="both as features"):
        featurize(df, ("x", "y"), (), ("s",), "y")


# ---------------------------------------------------------------- writing

def test_write_new_never_overwrites(tmp_path):
    p = tmp_path / "a.txt"
    write_new(p, lambda t: t.write_text("first"))
    with pytest.raises(FileExistsError):
        write_new(p, lambda t: t.write_text("second"))
    assert p.read_text() == "first"
    assert [f.name for f in tmp_path.iterdir()] == ["a.txt"]


def test_write_new_cleans_up_after_failed_writer(tmp_path):
    def boom(tmp):
        tmp.write_text("partial")
        raise RuntimeError("disk full")
    with pytest.raises(RuntimeError):
        write_new(tmp_path / "b.txt", boom)
    assert list(tmp_path.iterdir()) == []


def test_write_new_without_hard_links(tmp_path, monkeypatch):
    def no_link(src, dst):
        raise PermissionError("hard links not supported")
    monkeypatch.setattr(os, "link", no_link)
    p = tmp_path / "c.txt"
    write_new(p, lambda t: t.write_text("first"))
    assert p.read_text() == "first"
    with pytest.raises(FileExistsError):
        write_new(p, lambda t: t.write_text("second"))
    assert p.read_text() == "first"
    assert [f.name for f in tmp_path.iterdir()] == ["c.txt"]


# ---------------------------------------------------------------- end to end

def _check_alignment(out, cohort_df, meta, spec):
    """label, s_* and the one-hot f_* columns must match the cohort rows."""
    names = list(meta["feature_names"].values())
    for s in meta["n_rows"]:
        part = pd.read_parquet(out / f"{s}.parquet")
        rows = cohort_df[cohort_df["split"] == s].reset_index(drop=True)
        assert len(part) == len(rows) == meta["n_rows"][s]
        assert (part["label"].to_numpy() == rows["hospital_expire_flag"].to_numpy()).all()
        for j, col in enumerate(spec.protected):
            assert (part[f"s_{j}"].to_numpy() == rows[col].to_numpy()).all()
        for col in spec.cat_features:
            present = rows[col].notna().to_numpy()
            block = [f"f_{i}" for i, n in enumerate(names) if n.startswith(f"cat__{col}_")]
            for value in rows.loc[present, col].unique():
                hit = (rows[col] == value).to_numpy()
                name = f"cat__{col}_{value}"
                if name not in names:
                    # a level never seen in training: no column, all-zero block
                    train_values = cohort_df.loc[cohort_df["split"] == "train", col]
                    assert value not in set(train_values.dropna())
                    assert (part.loc[hit, block].to_numpy() == 0).all()
                    continue
                idx = names.index(name)
                assert (part[f"f_{idx}"].to_numpy()[hit] == 1).all()


@pytest.mark.parametrize("cohort, n_protected", [("icu", 2), ("hosp", 3)])
def test_cli_end_to_end(bulk_mimic, tmp_path, cohort, n_protected):
    out = tmp_path / f"run_{cohort}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        main(["--cohort", cohort, "--mimic-dir", str(bulk_mimic), "--out", str(out)])

    assert {p.name for p in tmp_path.iterdir()} == {out.name, "mimiciv"}  # no partial folder
    files = {p.name for p in out.iterdir()}
    assert files == {"cohort.parquet", "train.parquet", "val.parquet", "test.parquet",
                     "metadata.json", "manifest.json"}
    cohort_df = pd.read_parquet(out / "cohort.parquet")
    assert cohort_df["subject_id"].is_unique
    ids = {s: set(cohort_df.loc[cohort_df["split"] == s, "subject_id"])
           for s in ("train", "val", "test")}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])

    meta = json.loads((out / "metadata.json").read_text())
    spec = COHORTS[cohort]
    assert len(meta["protected"]) == n_protected
    _check_alignment(out, cohort_df, meta, spec)
    names = " ".join(meta["feature_names"].values())
    assert "UNKNOWN" not in names                      # missing values are imputed
    for leaky in ("expire", "disch", "death", "outtime", "race", "gender", "sex"):
        assert leaky not in names
    if cohort == "hosp":
        assert meta["feature_dim"] == 14               # fixed levels 5 + 4 + 5

    manifest = json.loads((out / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        assert sha256(out / name) == info["sha256"]
    for info in manifest["inputs"].values():
        assert len(info["sha256"]) == 64

    with pytest.raises(FileExistsError):
        main(["--cohort", cohort, "--mimic-dir", str(bulk_mimic), "--out", str(out)])


def test_cli_missing_as_category(bulk_mimic, tmp_path):
    out = tmp_path / "run"
    main(["--cohort", "icu", "--mimic-dir", str(bulk_mimic), "--out", str(out),
          "--missing-as-category"])
    meta = json.loads((out / "metadata.json").read_text())
    assert "cat__marital_status_UNKNOWN" in meta["feature_names"].values()


def test_failed_run_leaves_nothing(bulk_mimic, tmp_path, monkeypatch):
    def boom(_):
        raise RuntimeError("bad table")
    monkeypatch.setitem(COHORTS, "icu", dataclasses.replace(COHORTS["icu"], build=boom))
    out = tmp_path / "runs" / "x"
    with pytest.raises(RuntimeError):
        main(["--cohort", "icu", "--mimic-dir", str(bulk_mimic), "--out", str(out)])
    assert list((tmp_path / "runs").iterdir()) == []


def test_out_inside_repo_refused(bulk_mimic):
    from mimic_mortality.__main__ import REPO
    with pytest.raises(SystemExit):
        main(["--cohort", "hosp", "--mimic-dir", str(bulk_mimic), "--out", str(REPO / "outputs" / "x")])
    assert not (REPO / "outputs" / "x").exists()


def test_bad_split_sizes_rejected_before_reading(tmp_path):
    with pytest.raises(SystemExit):
        main(["--cohort", "hosp", "--mimic-dir", str(tmp_path / "nowhere"),
              "--out", str(tmp_path / "x"), "--test-size", "0.7", "--val-size", "0.4"])


def test_feature_with_no_training_values_rejected():
    df = _toy()
    df["split"] = assign_splits(df, "y", seed=0)
    df.loc[df["split"] == "train", "c"] = np.nan
    with pytest.raises(ValueError, match="no observed value"):
        featurize(df, (), ("c",), ("s",), "y", categories={"c": ("A", "B")})


def test_write_new_fallback_cleans_reserved_name(tmp_path, monkeypatch):
    def no_link(src, dst):
        raise PermissionError("hard links not supported")

    def bad_replace(src, dst):
        raise OSError("device busy")
    monkeypatch.setattr(os, "link", no_link)
    monkeypatch.setattr(os, "replace", bad_replace)
    with pytest.raises(OSError, match="device busy"):
        write_new(tmp_path / "d.txt", lambda t: t.write_text("data"))
    assert list(tmp_path.iterdir()) == []
