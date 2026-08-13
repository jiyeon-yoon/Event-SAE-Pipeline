#!/usr/bin/env python3
"""Validate an extended LIBERO run before uploading it or deleting the Pod."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.validate import (
    validate_extended_run,
)  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    summary = validate_extended_run(args.run_dir)
    print(json.dumps(summary, indent=2))
    print(f"EXTENDED_DATASET_OK: {Path(args.run_dir).resolve()}")


if __name__ == "__main__":
    main()
