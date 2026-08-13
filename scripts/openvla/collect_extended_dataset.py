#!/usr/bin/env python3
"""Collect the new rich LIBERO dataset without changing Event-SAE's collector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.config import (  # noqa: E402
    load_extended_config,
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
    cfg = load_extended_config(args.config, parse_overrides(args.override))
    # Keep --help and config parsing usable on a non-LIBERO machine.  The
    # heavyweight simulator/model imports are needed only when collection starts.
    from event_sae.openvla.extended_collection.runner import collect_extended_libero

    collect_extended_libero(cfg)


if __name__ == "__main__":
    main()
