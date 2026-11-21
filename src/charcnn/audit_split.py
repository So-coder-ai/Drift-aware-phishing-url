"""Command-line entry point for the first Kaggle notebook."""

from __future__ import annotations

import argparse
import sys

from .data import CoverageError, audit_and_split


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit and chronologically split DeepURLBench.")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args(argv)
    try:
        manifest = audit_and_split(args.config)
    except CoverageError as exc:
        print(f"STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print("Created leakage-safe chronological splits:")
    for name, details in manifest["splits"].items():
        print(f"  {name}: {details['rows']} rows -> {details['path']}")


if __name__ == "__main__":
    main()
