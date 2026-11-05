"""Patient-level train/val/test split and leak-free featurisation.

The split is drawn first. Every fitted transform (imputers, standard scaler,
one-hot encoder) is fitted on the training rows only and then applied to all
three splits, so no statistic of the validation or test rows reaches the
features.

Missing values (missing="impute", the default) are replaced by the median
(numeric) or the most frequent value (categorical) of the training rows, so
no feature says "this value was missing". missing="category" instead gives
missing values their own one-hot level, as the older scripts did; use it only
to reproduce them, because in MIMIC-IV missingness of marital status,
language and insurance is strongly associated with in-hospital death.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SPLITS = ("train", "val", "test")
MISSING_MODES = ("impute", "category")


def assign_splits(
    df: pd.DataFrame,
    label: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """Label-stratified split. Requires one row per subject_id, so it is
    also a patient-level split. `val_size` is a fraction of all rows."""
    if df["subject_id"].duplicated().any():
        raise ValueError("cohort has more than one row per subject_id; "
                         "a row split would leak patients across splits")
    if not 0 < test_size < 1 or not 0 <= val_size < 1 or test_size + val_size >= 1:
        raise ValueError(f"bad split sizes: test={test_size}, val={val_size}")
    y = df[label].to_numpy()
    pos = np.arange(len(df))
    train, test = train_test_split(pos, test_size=test_size, random_state=seed, stratify=y)
    val = np.array([], dtype=int)
    if val_size > 0:
        train, val = train_test_split(train, test_size=val_size / (1.0 - test_size),
                                      random_state=seed, stratify=y[train])
    split = np.empty(len(df), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    return pd.Series(split, index=df.index, name="split")


def featurize(
    df: pd.DataFrame,
    num_features: tuple[str, ...],
    cat_features: tuple[str, ...],
    protected: tuple[str, ...],
    label: str,
    categories: dict | None = None,
    missing: str = "impute",
    missing_levels: dict | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return {split: frame with f_0.. features, label, s_0.. protected} and the
    feature names. Transforms are fitted on the rows with split == 'train'.

    categories: fixed one-hot levels per categorical feature, or None to learn
    them from the training rows. missing: 'impute' or 'category' (see module
    docstring); missing_levels gives the level name per column for 'category'.
    """
    if "split" not in df.columns:
        raise ValueError("assign splits before featurising")
    if missing not in MISSING_MODES:
        raise ValueError(f"missing must be one of {MISSING_MODES}")
    overlap = set(num_features + cat_features) & set(protected + (label,))
    if overlap:
        raise ValueError(f"columns used both as features and as label/protected: {overlap}")
    missing_levels = missing_levels or {}

    def _inputs(rows: pd.DataFrame) -> pd.DataFrame:
        x = rows[list(num_features + cat_features)].copy()
        for c in cat_features:
            x[c] = x[c].astype(object).where(x[c].notna(), np.nan)
            if missing == "category":
                x[c] = x[c].fillna(missing_levels.get(c, "UNKNOWN"))
        return x

    train = df[df["split"] == "train"]
    if train.empty:
        raise ValueError("no training rows")
    x_train = _inputs(train)
    empty = [c for c in num_features + cat_features if x_train[c].isna().all()]
    if empty:
        raise ValueError(f"features with no observed value in the training rows: {empty}; "
                         "drop them from the cohort spec or check the input tables")

    steps = []
    if num_features:
        steps.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), list(num_features)))
    if cat_features:
        if categories is not None:
            levels = [list(categories[c]) for c in cat_features]
            for c, lv in zip(cat_features, levels):
                seen = set(x_train[c].dropna())
                unseen = sorted(seen - set(lv))
                if unseen:
                    raise ValueError(f"{c}: values {unseen[:5]} are not in the fixed levels {lv}")
                empty = [v for v in lv if v not in seen]
                if empty:
                    warnings.warn(f"{c}: no training rows in levels {empty}; "
                                  "their one-hot columns are all zero", stacklevel=2)
        else:
            levels = "auto"
        steps.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(categories=levels,
                                                                handle_unknown="ignore",
                                                                sparse_output=False))]),
                      list(cat_features)))
    transform = ColumnTransformer(steps, remainder="drop", sparse_threshold=0.0,
                                  verbose_feature_names_out=True)
    transform.fit(x_train)
    names = [str(n) for n in transform.get_feature_names_out()]

    out = {}
    for split in SPLITS:
        rows = df[df["split"] == split]
        if rows.empty:
            continue
        X = np.asarray(transform.transform(_inputs(rows)), dtype=np.float32)
        cols = {f"f_{i}": X[:, i] for i in range(X.shape[1])}
        cols["label"] = rows[label].to_numpy(np.int64)
        for j, c in enumerate(protected):
            cols[f"s_{j}"] = rows[c].to_numpy(np.int64)
        out[split] = pd.DataFrame(cols)
    return out, names


def missingness_by_label(df: pd.DataFrame, columns, label: str) -> dict:
    """Share of missing values per column among training rows, by label value.
    A large gap between the two labels means the missingness carries outcome
    information."""
    train = df[df["split"] == "train"]
    out = {}
    for c in columns:
        miss = train[c].isna()
        out[c] = {f"label_{int(v)}": float(miss[train[label] == v].mean())
                  for v in sorted(train[label].unique())}
    return out
