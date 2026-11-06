#!/usr/bin/env python3
"""Lightweight integrity checks for packaged documentation and assets."""

from __future__ import annotations

import hashlib
import re
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
    manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
    if len(manifest_lines) != 7:
        raise SystemExit(f"Expected 7 Figure 1--4 assets, found {len(manifest_lines)} manifest entries.")
    manifest_paths = set()
    for line in manifest_lines:
        expected, raw_path = line.split(maxsplit=1)
        raw_path = raw_path.lstrip("*")
        manifest_paths.add(raw_path)
        path = manifest.parent / raw_path
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != expected:
            raise SystemExit(f"Checksum mismatch: {path}")
    image_paths = {
        str(path.relative_to(manifest.parent))
        for path in manifest.parent.rglob("*.png")
    }
    if image_paths != manifest_paths:
        raise SystemExit("Manuscript PNG files do not match the Figure 1--4 checksum manifest.")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    embedded = set(re.findall(r'<img src="figures/manuscript/([^"]+)"', readme))
    if embedded != manifest_paths:
        raise SystemExit("README image references do not match the seven packaged manuscript panels.")
    print("Repository integrity checks passed.")


if __name__ == "__main__":
    main()
