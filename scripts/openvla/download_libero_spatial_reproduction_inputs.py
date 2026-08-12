"""Download and verify the five Spatial-500 dataset parts and layer-31 SAE."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download


DATASETS = {
    "0-1": (
        "jiyeony/event-sae-libero-spatial-tasks-0-1",
        "3babb75de5cedf5bed93daf192c955cd43cdff36",
        66,
    ),
    "2-3": (
        "jiyeony/event-sae-libero-spatial-tasks-2-3",
        "9899ee0d47f505287cad3273f473a270406f1d56",
        64,
    ),
    "4-5": (
        "jiyeony/event-sae-libero-spatial-tasks-4-5",
        "b3fcc95b47d46ea6977d59c6036ae8ff36ed4057",
        80,
    ),
    "6-7": (
        "jiyeony/event-sae-libero-spatial-tasks-6-7",
        "159119ccf6b45e65f1e68bbf7eb8080f445baba1",
        77,
    ),
    "8-9": (
        "jiyeony/event-sae-libero-spatial-tasks-8-9",
        "0c3318880daf9e4ea67558bebc41ed734d942f53",
        82,
    ),
}
SAE_REPO = "jiyeony/event-sae-openvla-libero-spatial-layer31-paper"
SAE_REVISION = "adb776b08f5b8ec4ea67556cf3e3bc60d91d607a"
SAE_SHA256 = "18443083d320d2ad3c607431f307697639966ad92075bdfe031398557f3ba9d6"
VISION_MODEL_REPO = "google/siglip-base-patch16-224"
VISION_MODEL_REVISION = "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed"


def _jsonl_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=300.0,
        help="Fail before downloading unless this much disk space is free.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).expanduser().resolve()
    pair_root = output_root / "pairs"
    checkpoint_root = output_root / "checkpoint"
    pair_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(output_root)
    free_gib = disk.free / (1024**3)
    existing_gib = sum(
        path.stat().st_size
        for path in output_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    ) / (1024**3)
    reusable_capacity_gib = free_gib + existing_gib
    if reusable_capacity_gib < args.min_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free plus {existing_gib:.1f} GiB already "
            f"downloaded under {output_root}; at least {args.min_free_gib:.1f} GiB "
            "of reusable capacity is required"
        )

    pair_summaries = []
    for pair, (repo_id, revision, expected_shards) in DATASETS.items():
        local_dir = pair_root / f"tasks-{pair}"
        print(f"Downloading {repo_id} -> {local_dir}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            local_dir=local_dir,
        )
        prompts = _jsonl_count(local_dir / "prompt_records.jsonl")
        videos = len(list((local_dir / "videos").glob("*.mp4")))
        shards = len(
            list(
                (local_dir / "sae_activations" / "post_mlp_residual").glob(
                    "layer_31_shard_*.pt"
                )
            )
        )
        if (prompts, videos, shards) != (100, 100, expected_shards):
            raise RuntimeError(
                f"Invalid {repo_id}: prompts={prompts}, videos={videos}, "
                f"shards={shards}; expected 100, 100, {expected_shards}"
            )
        pair_summaries.append(
            {
                "pair": pair,
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": str(local_dir),
                "prompts": prompts,
                "videos": videos,
                "shards": shards,
            }
        )

    print(f"Downloading {SAE_REPO} -> {checkpoint_root}", flush=True)
    snapshot_download(
        repo_id=SAE_REPO,
        revision=SAE_REVISION,
        local_dir=checkpoint_root,
        allow_patterns=["trainer_0/ae.pt", "trainer_0/config.json", "train-layer31-paper.log"],
    )
    checkpoint_path = checkpoint_root / "trainer_0" / "ae.pt"
    checkpoint_hash = _sha256(checkpoint_path)
    if checkpoint_hash != SAE_SHA256:
        raise RuntimeError(
            f"SAE checksum mismatch: {checkpoint_hash}; expected {SAE_SHA256}"
        )

    summary = {
        "schema_version": "event_sae_spatial_inputs_v1",
        "pairs": pair_summaries,
        "totals": {
            "episodes": sum(item["prompts"] for item in pair_summaries),
            "videos": sum(item["videos"] for item in pair_summaries),
            "activation_shards": sum(item["shards"] for item in pair_summaries),
        },
        "sae": {
            "repo_id": SAE_REPO,
            "revision": SAE_REVISION,
            "checkpoint_path": str(checkpoint_path),
            "sha256": checkpoint_hash,
        },
        "vision_model": {
            "repo_id": VISION_MODEL_REPO,
            "revision": VISION_MODEL_REVISION,
        },
        "disk_preflight": {
            "total_gib": disk.total / (1024**3),
            "free_gib": free_gib,
            "existing_download_gib": existing_gib,
            "reusable_capacity_gib": reusable_capacity_gib,
            "required_free_gib": args.min_free_gib,
        },
    }
    if summary["totals"] != {
        "episodes": 500,
        "videos": 500,
        "activation_shards": 369,
    }:
        raise RuntimeError(f"Invalid aggregate counts: {summary['totals']}")
    summary_path = output_root / "input_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["totals"], indent=2))
    print(f"INPUTS_OK: {summary_path}")


if __name__ == "__main__":
    main()
