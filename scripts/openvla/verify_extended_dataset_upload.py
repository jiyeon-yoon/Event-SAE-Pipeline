#!/usr/bin/env python3
"""Verify that every local extended-dataset file reached Hugging Face."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def local_inventory(run_dir: str | Path) -> dict[str, int]:
    root = Path(run_dir).expanduser().resolve()
    inventory: dict[str, int] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if not path.is_file() or ".cache" in relative.parts:
            continue
        inventory[relative.as_posix()] = path.stat().st_size
    return inventory


def compare_inventories(
    local: dict[str, int], remote: dict[str, int | None]
) -> tuple[list[str], list[tuple[str, int, int | None]]]:
    missing = sorted(set(local) - set(remote))
    wrong_size = sorted(
        (name, size, remote[name])
        for name, size in local.items()
        if name in remote and remote[name] is not None and remote[name] != size
    )
    return missing, wrong_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    local = local_inventory(args.run_dir)
    info = HfApi().dataset_info(args.repo_id, files_metadata=True)
    remote = {item.rfilename: item.size for item in info.siblings}
    missing, wrong_size = compare_inventories(local, remote)
    if missing or wrong_size:
        raise RuntimeError(
            f"Incomplete upload: missing={missing[:10]} wrong_size={wrong_size[:10]}"
        )
    required = {
        "manifest.json",
        "summary.json",
        "prompt_records.jsonl",
        "episode_results.jsonl",
        "trajectory_records.jsonl",
        "action_records.jsonl",
        "policy_uncertainty.jsonl",
        "sae_activations/post_mlp_residual/activation_index.jsonl",
    }
    absent_required = sorted(required - set(local))
    if absent_required:
        raise RuntimeError(f"Required local files are absent: {absent_required}")
    print(
        f"REMOTE_EXTENDED_DATASET_OK: {args.repo_id} "
        f"files={len(local)} bytes={sum(local.values())}"
    )


if __name__ == "__main__":
    main()
