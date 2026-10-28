"""Download + extract the UCI HAR Dataset into data/raw/.

Usage (from backend/):
    python scripts/download_uci_har.py
"""

from __future__ import annotations

import io
import pathlib
import shutil
import sys
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
EXPECTED_DIR = "UCI HAR Dataset"


def main() -> int:
    backend = pathlib.Path(__file__).resolve().parents[1]
    raw_common = backend / "data" / "raw"
    raw_common.mkdir(parents=True, exist_ok=True)
    final_root = raw_common / EXPECTED_DIR

    if (final_root / "train" / "X_train.txt").exists():
        print(f"already present: {final_root}")
        return 0
    if not (raw_common / "ucihar.zip").exists() and not (raw_common / "raw_data").exists():
        print("downloading", URL)
        with urllib.request.urlopen(URL) as resp:  # noqa: S310 (dataset host)
            blob = resp.read()
        (raw_common / "ucihar.zip").write_bytes(blob)
        print(f"saved {(len(blob) / 1e6):.1f} MB -> {raw_common / 'ucihar.zip'}")

    zip_path = raw_common / "ucihar.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
            inner = [n for n in names if n.endswith("Dataset.zip")]
            if inner:
                nested = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
                nested.extractall(raw_common)
            else:
                zf.extractall(raw_common)

    nested_dir = raw_common / "raw_data" / "UCI HAR Dataset"
    if nested_dir.exists() and not final_root.exists():
        shutil.move(str(nested_dir), final_root)
        shutil.rmtree(raw_common / "raw_data", ignore_errors=True)
    elif not final_root.exists():
        for cand in raw_common.iterdir():
            if cand.is_dir() and "dataset" in cand.name.lower():
                shutil.move(str(cand), final_root)
                break

    ok = (final_root / "train" / "X_train.txt").exists()
    print("OK:" if ok else "MISSING:", final_root / "train" / "X_train.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
