"""Reading MIMIC-IV tables, and writing outputs without ever overwriting a file."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

# MIMIC-IV 1.0 keeps admissions and patients in core/; 2.0 and later in hosp/.
MODULE_FALLBACKS = {"hosp": ("hosp", "core"), "icu": ("icu",)}


def find_table(mimic_dir: str | Path, module: str, name: str) -> Path:
    """Return the path of `<mimic_dir>/<module>/<name>.csv.gz` or `.csv`."""
    tried = []
    for mod in MODULE_FALLBACKS.get(module, (module,)):
        base = Path(mimic_dir) / mod
        for ext in (".csv.gz", ".csv"):
            path = base / f"{name}{ext}"
            if path.is_file():
                return path
            tried.append(str(path))
    raise FileNotFoundError(
        f"{name} table not found; tried {tried}. Pass the MIMIC-IV version folder "
        "(the one containing hosp/ and icu/, or core/ for 1.0), "
        "e.g. .../physionet.org/files/mimiciv/3.0"
    )


def read_table(
    mimic_dir: str | Path,
    module: str,
    name: str,
    usecols: Iterable[str],
    parse_dates: Iterable[str] = (),
) -> pd.DataFrame:
    """Read only the requested columns of one MIMIC-IV table.

    MIMIC-IV 1.0 calls the race column `ethnicity`; it is renamed to `race` so
    the same code runs on 1.0 and on 2.x/3.x.
    """
    path = find_table(mimic_dir, module, name)
    usecols = list(usecols)
    header = set(pd.read_csv(path, nrows=0).columns)
    rename = {}
    if "race" in usecols and "race" not in header and "ethnicity" in header:
        usecols = ["ethnicity" if c == "race" else c for c in usecols]
        rename = {"ethnicity": "race"}
    missing = [c for c in usecols if c not in header]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    df = pd.read_csv(path, usecols=usecols, parse_dates=list(parse_dates))
    return df.rename(columns=rename)


def _exists_error(path: Path) -> FileExistsError:
    return FileExistsError(f"{path} already exists; refusing to overwrite")


def write_new(path: str | Path, write_fn: Callable[[Path], None]) -> Path:
    """Write a file atomically and never replace an existing one.

    `write_fn` writes to a temporary file in the same folder, which is then
    hard-linked into place. `os.link` fails if the target exists, so a file
    is never overwritten and readers never see a half-written file. The
    temporary file is removed whether or not writing succeeds. On file
    systems without hard links the name is first reserved as an empty file,
    so a reader may briefly see it empty.
    """
    path = Path(path)
    if path.exists():
        raise _exists_error(path)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        write_fn(tmp)
        try:
            os.link(tmp, path)
        except FileExistsError:
            raise _exists_error(path) from None
        except OSError:
            # File systems without hard links: reserve the name exclusively,
            # then move the finished file onto the reserved (empty) name.
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                raise _exists_error(path) from None
            os.close(fd)
            try:
                os.replace(tmp, path)
            except BaseException:
                path.unlink(missing_ok=True)  # the empty file this call reserved
                raise
    finally:
        tmp.unlink(missing_ok=True)
    return path


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def physionet_checksums(mimic_dir: str | Path) -> dict[str, str]:
    """Read SHA256SUMS.txt shipped with a PhysioNet download, if present."""
    path = Path(mimic_dir) / "SHA256SUMS.txt"
    if not path.is_file():
        return {}
    sums = {}
    for line in path.read_text().splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            sums[parts[1].strip().lstrip("*")] = parts[0].lower()
    return sums
