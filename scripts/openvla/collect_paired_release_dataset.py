#!/usr/bin/env python3
"""Collect matched normal/forced-release LIBERO rollouts with rich telemetry."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.config import (  # noqa: E402
    load_paired_release_config,
    parse_overrides,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override key=value (repeatable), e.g. env.task_ids=0,1",
    )
    args = parser.parse_args()
    cfg = load_paired_release_config(
        args.config,
        parse_overrides(args.override),
    )

    # Keep config parsing and --help usable without loading LIBERO/OpenVLA.
    from event_sae.openvla.extended_collection.paired_runner import (
        collect_paired_release_libero,
    )

    collect_paired_release_libero(cfg)


if __name__ == "__main__":
    main()
