#!/usr/bin/env python3
"""Validate a paired release run before upload or Pod termination."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.paired_validate import (  # noqa: E402
    validate_paired_release_run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--min-valid-pairs-per-task",
        type=int,
        default=None,
        help="Optional stricter per-task threshold; cannot weaken the manifest",
    )
    parser.add_argument(
        "--min-primary-pairs-per-task",
        type=int,
        default=None,
        help="Optional stricter primary-cohort threshold; cannot weaken the manifest",
    )
    args = parser.parse_args()
    summary = validate_paired_release_run(
        args.run_dir,
        min_valid_pairs_per_task=args.min_valid_pairs_per_task,
        min_primary_pairs_per_task=args.min_primary_pairs_per_task,
    )
    print(json.dumps(summary, indent=2))
    print(f"PAIRED_RELEASE_DATASET_OK: {Path(args.run_dir).resolve()}")


if __name__ == "__main__":
    main()
