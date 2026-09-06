#!/usr/bin/env python3
"""Verify that every local extended-dataset file reached Hugging Face."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ALLOWED_REMOTE_ONLY_FILES = frozenset({".gitattributes", "README.md"})


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


def unexpected_remote_files(
    local: dict[str, int], remote: dict[str, int | None]
) -> list[str]:
    """Return stale remote files, excluding metadata created by the Hub."""
    return sorted(set(remote) - set(local) - ALLOWED_REMOTE_ONLY_FILES)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    args = parser.parse_args()

    from huggingface_hub import HfApi

    root = Path(args.run_dir).expanduser().resolve()
    local = local_inventory(args.run_dir)
    info = HfApi().dataset_info(args.repo_id, files_metadata=True)
    remote = {item.rfilename: item.size for item in info.siblings}
    missing, wrong_size = compare_inventories(local, remote)
    unexpected_remote = unexpected_remote_files(local, remote)
    if missing or wrong_size or unexpected_remote:
        raise RuntimeError(
            "Upload inventory mismatch: "
            f"missing={missing[:10]} "
            f"wrong_size={wrong_size[:10]} "
            f"unexpected_remote={unexpected_remote[:10]}"
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
    manifest = json.loads(
        (root / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    schema = manifest.get("schema_version")
    if schema == "extended_openvla_libero_paired_release_v2":
        if manifest.get("collection_status") != "complete":
            raise RuntimeError("Paired collection is not marked complete")
        from event_sae.openvla.extended_collection.paired_validate import (
            validate_paired_release_run,
        )

        validate_paired_release_run(root)
        required.update({"pair_results.jsonl", "COLLECTION_COMPLETE"})
    elif schema != "extended_openvla_libero_v1":
        raise RuntimeError(f"Unsupported extended dataset schema: {schema!r}")
    absent_required = sorted(required - set(local))
    if absent_required:
        raise RuntimeError(f"Required local files are absent: {absent_required}")
    print(
        f"REMOTE_EXTENDED_DATASET_OK: {args.repo_id} "
        f"files={len(local)} bytes={sum(local.values())}"
    )


if __name__ == "__main__":
    main()
