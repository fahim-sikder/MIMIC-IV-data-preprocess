"""Extract a MIMIC-IV in-hospital mortality cohort.

    python -m mimic_mortality --cohort icu  --mimic-dir /path/to/mimiciv/3.0 --out /data/icu_3.0
    python -m mimic_mortality --cohort hosp --mimic-dir /path/to/mimiciv/3.0 --out /data/hosp_3.0

Writes a new folder `--out` with cohort.parquet, train/val/test.parquet,
metadata.json and manifest.json. The output holds row-level MIMIC data, so
`--out` is required and may not lie inside this repository. The name is
reserved at the start, the files are written under a temporary name, and the
folder is moved into place only when every file is written; a failed run
removes both. An existing folder is never reused.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .cohorts import COHORTS, LABEL
from .features import SPLITS, assign_splits, featurize, missingness_by_label
from .io import find_table, physionet_checksums, sha256, write_new


REPO = Path(__file__).resolve().parents[1]


def _git_state() -> dict:
    repo = REPO

    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain", "--", "mimic_mortality") if commit else None
    return {"commit": commit, "package_modified": bool(status) if commit else None}


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    return write_new(path, lambda tmp: df.to_parquet(tmp, index=False))


def _write_json(obj: dict, path: Path) -> Path:
    def _dump(tmp: Path) -> None:
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=str)
            f.write("\n")
    return write_new(path, _dump)


def _inputs(mimic_dir: Path, paths: list[Path]) -> dict:
    sums = physionet_checksums(mimic_dir)
    out = {}
    for p in paths:
        digest = sha256(p)
        rel = p.relative_to(mimic_dir).as_posix()
        expected = sums.get(rel)
        out[rel] = {"bytes": p.stat().st_size, "sha256": digest,
                    "matches_physionet_sha256sums": None if expected is None
                    else digest == expected}
    return out


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m mimic_mortality", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cohort", choices=sorted(COHORTS), required=True)
    ap.add_argument("--mimic-dir", default=os.environ.get("MIMIC_IV_DIR"),
                    help="MIMIC-IV version folder holding hosp/ and icu/ "
                         "(or set MIMIC_IV_DIR)")
    ap.add_argument("--out", required=True,
                    help="new output folder, outside this repository (must not exist)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.1,
                    help="fraction of all rows; 0 disables the validation split")
    ap.add_argument("--missing-as-category", action="store_true",
                    help="give missing categorical values their own one-hot level, as the "
                         "older scripts did. Off by default: in MIMIC-IV this missingness "
                         "is strongly associated with in-hospital death")
    args = ap.parse_args(argv)
    if not args.mimic_dir:
        ap.error("--mimic-dir is required (or set MIMIC_IV_DIR)")
    if not 0 < args.test_size < 1 or not 0 <= args.val_size < 1 \
            or args.test_size + args.val_size >= 1:
        ap.error(f"bad split sizes: --test-size {args.test_size}, --val-size {args.val_size}")
    out = Path(args.out).expanduser().resolve()
    if out == REPO or REPO in out.parents:
        ap.error(f"--out {out} is inside the repository {REPO}; outputs hold row-level "
                 "MIMIC data, so write them somewhere that is never shared with the code")
    return args


def _build(args, argv, spec, mimic_dir: Path, work: Path, input_paths: list[Path]) -> dict:
    missing = "category" if args.missing_as_category else "impute"
    cohort = spec.build(mimic_dir)
    cohort["split"] = assign_splits(cohort, LABEL, args.test_size, args.val_size, args.seed)
    splits, feature_names = featurize(
        cohort, spec.num_features, spec.cat_features, spec.protected, LABEL,
        categories=spec.categories, missing=missing, missing_levels=spec.missing_levels)

    written = [_write_parquet(cohort, work / "cohort.parquet")]
    for name, frame in splits.items():
        written.append(_write_parquet(frame, work / f"{name}.parquet"))

    train = splits["train"]
    f_cols = [c for c in train.columns if c.startswith("f_")]
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = [np.corrcoef(train[c], train["label"])[0, 1] for c in f_cols]
    corr = np.abs(np.nan_to_num(np.asarray(corr, dtype=float), nan=0.0))
    top = int(np.argmax(corr)) if len(corr) else None
    cells = (train.groupby([f"s_{j}" for j in range(len(spec.protected))]).size()
             .rename("n").reset_index())

    metadata = {
        "cohort": spec.name,
        "description": spec.description,
        "mimic_version": mimic_dir.name,
        "task": "in-hospital mortality (binary)",
        "label": f"label = {LABEL} of the selected admission",
        "unit": "one row per patient (subject_id)",
        "split": {"method": "label-stratified, one row per patient", "seed": args.seed,
                  "test_size": args.test_size, "val_size": args.val_size},
        "n_rows": {s: int(len(f)) for s, f in splits.items()},
        "mortality_rate": {s: float(f["label"].mean()) for s, f in splits.items()},
        "features": {"num": list(spec.num_features), "cat": list(spec.cat_features),
                     "levels": "fixed" if spec.categories else "learned from train",
                     "missing_values": ("own one-hot level" if missing == "category" else
                                        "imputed with train median / most frequent value"),
                     "fitted_on": "train only"},
        "feature_dim": len(feature_names),
        "feature_names": {f"f_{i}": n for i, n in enumerate(feature_names)},
        "protected": {f"s_{j}": {"column": c, "codes": spec.protected_maps.get(c)}
                      for j, c in enumerate(spec.protected)},
        "train_cell_sizes": cells.to_dict(orient="records"),
        "train_missing_share_by_label": missingness_by_label(
            cohort, spec.num_features + spec.cat_features, LABEL),
        "max_abs_corr_feature_label_train": (
            {"feature": feature_names[top], "abs_corr": float(corr[top])}
            if top is not None else None),
        "row_order": "rows of <split>.parquet follow cohort.parquet rows with that split",
    }
    written.append(_write_json(metadata, work / "metadata.json"))

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": ["python", "-m", "mimic_mortality",
                    *(argv if argv is not None else sys.argv[1:])],
        "args": vars(args) | {"mimic_dir": str(mimic_dir)},
        "inputs": _inputs(mimic_dir, input_paths),
        "code": _git_state(),
        "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                     "numpy": np.__version__, "scikit-learn": sklearn.__version__},
        "files": {p.name: {"sha256": sha256(p),
                           "rows": int(len(cohort)) if p.name == "cohort.parquet"
                           else metadata["n_rows"].get(p.stem)}
                  for p in written},
    }
    _write_json(manifest, work / "manifest.json")
    return metadata


def main(argv=None) -> Path:
    args = parse_args(argv)
    spec = COHORTS[args.cohort]
    mimic_dir = Path(args.mimic_dir).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()

    input_paths = [find_table(mimic_dir, "hosp", "admissions"),
                   find_table(mimic_dir, "hosp", "patients")]
    if spec.name == "icu":
        input_paths.append(find_table(mimic_dir, "icu", "icustays"))

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        out.mkdir(exist_ok=False)  # reserve the name for this run
    except FileExistsError:
        raise FileExistsError(f"output folder {out} already exists; choose a new --out "
                              "so earlier results are never overwritten") from None
    work = out.parent / f".{out.name}.partial-{os.getpid()}"
    print(f"[{spec.name}] reading {mimic_dir}")
    try:
        work.mkdir(exist_ok=False)
        metadata = _build(args, argv, spec, mimic_dir, work, input_paths)
    except BaseException:
        # remove only what this run created: its temporary folder and its reserved name
        shutil.rmtree(work, ignore_errors=True)
        try:
            out.rmdir()  # succeeds only if still empty
        except OSError:
            pass
        raise
    _move_into_place(work, out)

    print(f"[{spec.name}] MIMIC-IV {mimic_dir.name}: patients={sum(metadata['n_rows'].values())}  "
          + "  ".join(f"{s}={metadata['n_rows'][s]} (mortality {metadata['mortality_rate'][s]:.3f})"
                      for s in SPLITS if s in metadata["n_rows"]))
    top = metadata["max_abs_corr_feature_label_train"]
    print(f"[{spec.name}] feature_dim={metadata['feature_dim']}  missing values: "
          f"{metadata['features']['missing_values']}")
    if top:
        print(f"[{spec.name}] most label-correlated feature on train: "
              f"{top['feature']} (|r|={top['abs_corr']:.3f})")
    print(f"[{spec.name}] wrote {out}")
    return out


def _move_into_place(work: Path, out: Path) -> None:
    """Move the finished folder onto the reserved, empty `out`.

    On POSIX, rename replaces an empty directory and fails on a non-empty one,
    so files that appeared in `out` meanwhile are never replaced.
    """
    try:
        os.rename(work, out)
    except OSError:
        if any(out.iterdir()):
            raise FileExistsError(f"{out} is no longer empty; results left in {work}") from None
        out.rmdir()  # e.g. Windows, where rename cannot replace a directory
        os.rename(work, out)


if __name__ == "__main__":
    main()
