"""Command-line entry point for checkpoint and synthetic operations."""

from __future__ import annotations

import argparse
from pathlib import Path

from .checkpoint import fetch_checkpoint, verify_checkpoint
from .figures import generate_synthetic_figures
from .synthetic import analyze_synthetic, run_synthetic


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
    elif args.command == "verify-checkpoint":
        output = verify_checkpoint(args.path)
    elif args.command == "fetch-checkpoint":
        output = fetch_checkpoint(args.destination)
    else:  # pragma: no cover
        raise RuntimeError(args.command)
    print(output)


if __name__ == "__main__":
    main()
