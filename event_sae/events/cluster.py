"""Task-local agglomerative clustering of event features.

For each unique `task_description`, builds a single concatenated feature
vector per sample from [normalized vision embedding, z-scored state vector,
z-scored progress percent] and runs agglomerative clustering with cosine
distance threshold. Exemplars are the members closest to the cluster
centroid (preferring unique episodes). Clusters at or above `min_coverage`
of unique task episodes are marked canonical.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from event_sae.events.io import load_jsonl, write_jsonl


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "task"


def _l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return matrix / norms


def _zscore(matrix: np.ndarray) -> np.ndarray:
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std > 1e-12, std, 1.0)
    return (matrix - mean) / std


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 1.0 - np.clip(a @ b.T, -1.0, 1.0)


def build_task_vectors(
    task_records: list[dict],
    *,
    vision_weight: float = 1.0,
    state_weight: float = 0.5,
    progress_weight: float = 0.4,
) -> np.ndarray:
    """Concatenate weighted normalized [vision, state, progress] per record, then L2-normalize rows."""
    vision = np.asarray([record["vision_embedding"] for record in task_records], dtype=np.float32)
    state = np.asarray([record["state_vector"] for record in task_records], dtype=np.float32)
    progress = np.asarray([[record["progress_percent"]] for record in task_records], dtype=np.float32)

    vision_norm = _l2_normalize_rows(vision)
    state_norm = _l2_normalize_rows(_zscore(state))
    progress_norm = _zscore(progress)

    combined = np.concatenate(
        [vision_weight * vision_norm, state_weight * state_norm, progress_weight * progress_norm],
        axis=1,
    )
    return _l2_normalize_rows(combined)


def select_exemplars(
    member_records: list[dict],
    member_vectors: np.ndarray,
    *,
    num_exemplars: int,
) -> list[dict]:
    """Select up to `num_exemplars` cluster members closest to the centroid,
    preferring unique source episodes."""
    centroid = _l2_normalize_rows(member_vectors.mean(axis=0, keepdims=True))
    distances = _cosine_distance(member_vectors, centroid).reshape(-1)
    order = np.argsort(distances)

    exemplars: list[dict] = []
    used_episodes: set[int] = set()
    for idx in order:
        record = member_records[int(idx)]
        episode_num = int(record["episode_num"])
        if episode_num in used_episodes:
            continue
        exemplars.append(record)
        used_episodes.add(episode_num)
        if len(exemplars) >= num_exemplars:
            return exemplars
    for idx in order:
        record = member_records[int(idx)]
        if record in exemplars:
            continue
        exemplars.append(record)
        if len(exemplars) >= num_exemplars:
            break
    return exemplars


def cluster_events(
    event_features_path: Path,
    output_dir: Path,
    *,
    prompt_records_path: Path | None = None,
    vision_weight: float = 1.0,
    state_weight: float = 0.5,
    progress_weight: float = 0.4,
    distance_threshold: float = 0.18,
    min_coverage: float = 0.5,
    num_exemplars: int = 5,
) -> dict:
    """Cluster event features task-locally and write assignments + summaries.

    Outputs (under output_dir):
      - cluster_assignments.jsonl  (one row per sample -> cluster_id)
      - clusters.jsonl             (one row per cluster + exemplars + coverage)
      - summary.json
    """
    event_features_path = Path(event_features_path).resolve()
    if not event_features_path.is_file():
        raise FileNotFoundError(f"event_features.jsonl not found: {event_features_path}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(event_features_path)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_task[record["task_description"]].append(record)

    # Paper coverage is relative to every attempted rollout for the task,
    # including a rollout where AWE yielded no usable event. Preserve the
    # historical event-only denominator when prompt_records is omitted.
    attempted_episodes_by_task: dict[str, set[int]] = defaultdict(set)
    if prompt_records_path is not None:
        prompt_records_path = Path(prompt_records_path).resolve()
        if not prompt_records_path.is_file():
            raise FileNotFoundError(f"prompt_records.jsonl not found: {prompt_records_path}")
        for prompt in load_jsonl(prompt_records_path):
            attempted_episodes_by_task[str(prompt["task_description"])].add(
                int(prompt["episode_num"])
            )

    assignments: list[dict] = []
    cluster_summaries: list[dict] = []
    for task_description, task_records in sorted(by_task.items()):
        task_records = sorted(
            task_records,
            key=lambda item: (
                int(item["episode_num"]),
                int(item["waypoint_step"]),
                int(item["waypoint_rank"]),
            ),
        )
        task_vectors = build_task_vectors(
            task_records,
            vision_weight=vision_weight,
            state_weight=state_weight,
            progress_weight=progress_weight,
        )
        if len(task_records) == 1:
            labels = np.asarray([0], dtype=np.int32)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=distance_threshold,
            )
            labels = clustering.fit_predict(task_vectors)

        task_slug = _slugify(task_description)
        event_episode_nums = {int(record["episode_num"]) for record in task_records}
        attempted_episode_nums = attempted_episodes_by_task.get(task_description)
        total_episodes = len(attempted_episode_nums or event_episode_nums)
        member_ids_by_label: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            member_ids_by_label[int(label)].append(idx)

        cluster_id_by_label: dict[int, str] = {}
        for local_cluster_idx, label in enumerate(sorted(member_ids_by_label)):
            member_indices = member_ids_by_label[label]
            member_records = [task_records[idx] for idx in member_indices]
            member_vectors = task_vectors[member_indices]
            exemplars = select_exemplars(member_records, member_vectors, num_exemplars=num_exemplars)
            episode_nums = sorted({int(record["episode_num"]) for record in member_records})
            coverage = len(episode_nums) / max(total_episodes, 1)
            cluster_id = f"{task_slug}_cluster_{local_cluster_idx:02d}"
            cluster_id_by_label[int(label)] = cluster_id
            cluster_summaries.append(
                {
                    "cluster_id": cluster_id,
                    "task_description": task_description,
                    "cluster_label": int(label),
                    "num_members": len(member_records),
                    "total_task_episodes": int(total_episodes),
                    "episode_coverage": float(coverage),
                    "is_canonical": bool(coverage >= min_coverage),
                    "member_sample_ids": [record["sample_id"] for record in member_records],
                    "member_episode_nums": episode_nums,
                    "representative_sample_ids": [record["sample_id"] for record in exemplars],
                    "representative_clip_paths": [record["clip_path"] for record in exemplars],
                    "representative_frame_paths": [record["frame_paths"] for record in exemplars],
                    "representative_waypoint_steps": [int(record["waypoint_step"]) for record in exemplars],
                    "representative_progress_percents": [
                        float(record["progress_percent"]) for record in exemplars
                    ],
                    "cluster_mean_progress_percent": float(
                        np.mean([record["progress_percent"] for record in member_records])
                    ),
                }
            )

        for record, label in zip(task_records, labels, strict=True):
            assignments.append(
                {
                    "sample_id": record["sample_id"],
                    "task_description": task_description,
                    "episode_num": int(record["episode_num"]),
                    "waypoint_step": int(record["waypoint_step"]),
                    "cluster_label": int(label),
                    "cluster_id": cluster_id_by_label[int(label)],
                }
            )

    write_jsonl(output_dir / "cluster_assignments.jsonl", assignments)
    write_jsonl(output_dir / "clusters.jsonl", cluster_summaries)
    summary = {
        "event_features_path": str(event_features_path),
        "prompt_records_path": (
            str(prompt_records_path) if prompt_records_path is not None else None
        ),
        "coverage_denominator": (
            "all_attempted_episodes" if prompt_records_path is not None else "event_episodes_only"
        ),
        "num_events": len(records),
        "num_tasks": len(by_task),
        "num_clusters": len(cluster_summaries),
        "num_recurring_clusters": sum(
            1 for cluster in cluster_summaries if cluster["is_canonical"]
        ),
        "vision_weight": float(vision_weight),
        "state_weight": float(state_weight),
        "progress_weight": float(progress_weight),
        "distance_threshold": float(distance_threshold),
        "min_coverage": float(min_coverage),
        "num_exemplars": int(num_exemplars),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
