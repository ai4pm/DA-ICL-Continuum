"""TabPFN-2.6 checkpoint acquisition and integrity verification."""

from __future__ import annotations

import hashlib
from pathlib import Path


TABPFN26_REPOSITORY = "Prior-Labs/tabpfn_2_6"
TABPFN26_FILENAME = "tabpfn-v2.6-classifier-v2.6_default.ckpt"
TABPFN26_SHA256 = "0578fa56f97e11024e31735aaec2c4e7332584b7730242fbaf6c0bbd0299206a"


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    observed = sha256(path)
    if observed != TABPFN26_SHA256:
        raise ValueError(
            f"TabPFN-2.6 checkpoint SHA-256 mismatch: expected {TABPFN26_SHA256}, observed {observed}."
        )
    return path


def resolve_checkpoint(path: str | Path | None = None) -> Path:
    """Resolve an explicit path or the standard TabPFN cache and verify it."""
    candidate = (
        Path(path).expanduser()
        if path is not None
        else Path.home() / ".cache" / "tabpfn" / TABPFN26_FILENAME
    )
    if not candidate.is_file():
        raise FileNotFoundError(
            f"TabPFN-2.6 checkpoint not found at {candidate}. Run "
            "'bibm-replicate fetch-checkpoint --destination checkpoints' and pass "
            "model_path='checkpoints/tabpfn-v2.6-classifier-v2.6_default.ckpt'."
        )
    return verify_checkpoint(candidate)


def fetch_checkpoint(destination: str | Path) -> Path:
    from huggingface_hub import hf_hub_download

    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=TABPFN26_REPOSITORY,
            filename=TABPFN26_FILENAME,
            local_dir=destination,
        )
    )
    return verify_checkpoint(downloaded)
