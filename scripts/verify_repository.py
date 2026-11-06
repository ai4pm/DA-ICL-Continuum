#!/usr/bin/env python3
"""Lightweight integrity checks for packaged documentation and assets."""

from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    required = [
        ROOT / "README.md",
        ROOT / "LICENSE",
        ROOT / "CITATION.cff",
        ROOT / "configs/synthetic_paper.yaml",
        ROOT / "src/bibm_replication/estimators.py",
        ROOT / "docs/figures.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required repository files: {missing}")
    manifest = ROOT / "figures/manuscript/SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, raw_path = line.split(maxsplit=1)
        path = manifest.parent / raw_path.lstrip("*")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise SystemExit(f"Checksum mismatch: {path}")
    print("Repository integrity checks passed.")


if __name__ == "__main__":
    main()
