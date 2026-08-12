"""CLI: build event-feature score matrix.

Joins per-event SAE top-k activations (either online sbatch output or the
offline `scripts/extract_topk.py` output, both of which produce
`token_topk_sparse_v1` shards) with VLM-annotated event clusters, and
writes a single `.pt` payload with the `(num_clusters, dict_size)` score
matrix and per-row top-N feature summaries.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.scoring import score_cluster_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-feature score matrix.")
    parser.add_argument(
        "--topk-run-dir",
        required=True,
        help="Directory containing manifest.json + token_topk_sparse_v1 shards "
        "(either an online EVAL run dir or the offline extract_topk output dir).",
    )
    parser.add_argument("--event-features-path", required=True, help="Path to event_features.jsonl")
    parser.add_argument("--cluster-assignments-path", required=True, help="Path to cluster_assignments.jsonl")
    parser.add_argument(
        "--clusters-path",
        default=None,
        help=(
            "Path to authoritative clusters.jsonl. Recommended: this prevents a failed "
            "descriptive VLM annotation from removing a cluster from ranking."
        ),
    )
    parser.add_argument(
        "--cluster-annotations-path",
        default=None,
        help="Optional descriptive cluster_annotations.jsonl (phrase/phase labels).",
    )
    parser.add_argument(
        "--prompt-records-path",
        default=None,
        help=(
            "Path to prompt_records.jsonl from the dense collection run. Used to map every "
            "episode to its task_id for matrix_task_mean (paper-faithful behavior: per-task "
            "mean is over all rollout timesteps in the task, not only event-window timesteps). "
            "If omitted, matrix_task_mean only averages over episodes that produced events."
        ),
    )
    parser.add_argument("--output-path", required=True, help="Where to save the score matrix .pt payload")
    parser.add_argument("--window-size", type=int, default=5, help="Half-window size in env steps (default: 5)")
    parser.add_argument("--top-n", type=int, default=20, help="Top-N features to summarize per row (default: 20)")
    parser.add_argument(
        "--step-mapping",
        choices=("auto", "action_executed", "chunk_executed", "inference_step"),
        default="auto",
        help=(
            "How shard rows map to env timesteps. action_executed uses chunk_start + token_idx "
            "(OpenPI AE default); chunk_executed broadcasts each row to all executed env steps "
            "of its chunk (OpenPI PG default); inference_step uses step_in_episode directly "
            "(OpenVLA legacy). 'auto' picks per manifest.capture_target."
        ),
    )
    args = parser.parse_args()
    if args.clusters_path is None and args.cluster_annotations_path is None:
        parser.error("provide --clusters-path and/or --cluster-annotations-path")

    summary = score_cluster_features(
        topk_run_dir=Path(args.topk_run_dir),
        event_features_path=Path(args.event_features_path),
        cluster_assignments_path=Path(args.cluster_assignments_path),
        cluster_annotations_path=(
            Path(args.cluster_annotations_path) if args.cluster_annotations_path else None
        ),
        clusters_path=Path(args.clusters_path) if args.clusters_path else None,
        output_path=Path(args.output_path),
        window_size=args.window_size,
        top_n=args.top_n,
        step_mapping=args.step_mapping,
        prompt_records_path=Path(args.prompt_records_path) if args.prompt_records_path else None,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
