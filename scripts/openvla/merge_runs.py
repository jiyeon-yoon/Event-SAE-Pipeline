"""Merge parallel OpenVLA collection runs into one Event-SAE run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from event_sae.openvla.merge_runs import MergeConfig, merge_openvla_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--expected-shards", type=int, default=None)
    parser.add_argument(
        "--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink"
    )
    args = parser.parse_args()
    summary = merge_openvla_runs(
        MergeConfig(
            input_dirs=tuple(args.input_dir),
            output_dir=args.output_dir,
            trials_per_task=args.trials_per_task,
            expected_tasks=args.expected_tasks,
            expected_shards=args.expected_shards,
            link_mode=args.link_mode,
        )
    )
    print(json.dumps(summary, indent=2))
    print("MERGE_OK")


if __name__ == "__main__":
    main()
