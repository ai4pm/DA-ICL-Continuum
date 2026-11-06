"""Command-line entry point for reproducible batch operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .checkpoint import fetch_checkpoint, verify_checkpoint
from .estimators import ContinuumICLClassifier, DAICLClassifier, DiscreteICLClassifier
from .figures import generate_synthetic_figures
from .metrics import binary_metrics
from .synthetic import analyze_synthetic, run_synthetic


METHODS = {
    "discrete": DiscreteICLClassifier,
    "continuum": ContinuumICLClassifier,
    "da-icl": DAICLClassifier,
}


def _require(archive, key):
    if key not in archive:
        raise ValueError(f"Input archive is missing required key {key!r}.")
    return archive[key]


def run_npz(args):
    with np.load(args.input, allow_pickle=False) as archive:
        common = dict(
            n_neighbors=args.n_neighbors,
            n_estimators=args.n_estimators,
            temperature=args.temperature,
            random_state=args.random_state,
            device=args.device,
            model_path=str(args.model_path) if args.model_path else None,
            backend=args.backend,
        )
        model = METHODS[args.method](**common)
        fit_kwargs = {}
        query_kwargs = {}
        if args.method in {"continuum", "da-icl"}:
            fit_kwargs["genotype_features"] = _require(archive, "genotype_dev")
            query_kwargs["genotype_features"] = _require(archive, "genotype_test")
        if args.method == "da-icl":
            fit_kwargs["dace_features"] = _require(archive, "ae_dev")
        if args.method == "discrete":
            fit_kwargs["ancestry"] = _require(archive, "ancestry_dev")
            query_kwargs["ancestry"] = _require(archive, "ancestry_test")
        if "sample_ids" in archive:
            fit_kwargs["sample_ids"] = archive["sample_ids"]
        model.fit(_require(archive, "X_dev"), _require(archive, "y_dev"), **fit_kwargs)
        probability = model.predict_proba(_require(archive, "X_test"), **query_kwargs)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination, probability=probability, prediction=(probability[:, 1] >= 0.5).astype(int))
        report_path = destination.with_suffix(".contexts.json")
        report_path.write_text(
            json.dumps(model.last_context_report_.to_dict(orient="records"), indent=2, default=str),
            encoding="utf-8",
        )
        if "y_test" in archive:
            truth = np.asarray(archive["y_test"]).reshape(-1)
            if len(truth) != len(probability):
                raise ValueError("y_test must align with X_test.")
            destination.with_suffix(".metrics.json").write_text(
                json.dumps(binary_metrics(truth, probability[:, 1]), indent=2),
                encoding="utf-8",
            )
    return destination


def parser():
    root = argparse.ArgumentParser(prog="bibm-replicate")
    commands = root.add_subparsers(dest="command", required=True)

    synthetic = commands.add_parser("run-synthetic", help="Run configured synthetic replicates and sweeps.")
    synthetic.add_argument("--config", required=True, type=Path)

    analyze = commands.add_parser("analyze-synthetic", help="Aggregate saved synthetic replicate metrics.")
    analyze.add_argument("--result-root", required=True, type=Path)

    figures = commands.add_parser("generate-figures", help="Regenerate figures from saved result tables.")
    figures.add_argument("--experiment", choices=["synthetic"], required=True)
    figures.add_argument("--result-root", type=Path, default=Path("results/synthetic"))

    batch = commands.add_parser("run-npz", help="Run one estimator on a convenient NPZ archive.")
    batch.add_argument("--input", required=True, type=Path)
    batch.add_argument("--output", required=True, type=Path)
    batch.add_argument("--method", choices=sorted(METHODS), required=True)
    batch.add_argument("--n-neighbors", type=int, default=100)
    batch.add_argument("--n-estimators", type=int, default=32)
    batch.add_argument("--temperature", type=float, default=0.9)
    batch.add_argument("--random-state", type=int, default=0)
    batch.add_argument("--device", default="cuda")
    batch.add_argument("--model-path", type=Path)
    batch.add_argument("--backend", choices=["tabpfn", "logistic"], default="tabpfn")

    verify = commands.add_parser("verify-checkpoint", help="Verify the exact TabPFN-2.6 checkpoint SHA-256.")
    verify.add_argument("path", type=Path)
    fetch = commands.add_parser("fetch-checkpoint", help="Download TabPFN-2.6 from Prior Labs and verify it.")
    fetch.add_argument("--destination", type=Path, default=Path("checkpoints"))
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "run-synthetic":
        output = run_synthetic(args.config)
    elif args.command == "analyze-synthetic":
        output = analyze_synthetic(args.result_root)
    elif args.command == "generate-figures":
        output = generate_synthetic_figures(args.result_root)
    elif args.command == "run-npz":
        output = run_npz(args)
    elif args.command == "verify-checkpoint":
        output = verify_checkpoint(args.path)
    elif args.command == "fetch-checkpoint":
        output = fetch_checkpoint(args.destination)
    else:  # pragma: no cover
        raise RuntimeError(args.command)
    print(output)


if __name__ == "__main__":
    main()
