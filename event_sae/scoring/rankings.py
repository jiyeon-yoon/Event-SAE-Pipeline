"""Four feature-ranking strategies for SAE candidate selection.

Implements the rankings compared in paper Section 4.4:

- ``event_aligned`` — top-N features per cluster row from an existing
  event-feature score matrix (output of ``score_cluster_features.py``).
- ``window_mean`` — for each cluster row, mean SAE activation over the
  same event windows used by event-aligned, then top-N features.
- ``task_mean`` — for each task, mean SAE activation over every rollout
  timestep in the run, then top-N features.
- ``random_alive`` — uniform random sample from features that fire at
  least once in the run, excluding features already selected by any of
  the three informed rankings.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None



# ---------------------------------------------------------------------------
# Shared shard iteration
# ---------------------------------------------------------------------------


def _load_manifest(topk_run_dir: Path) -> dict:
    manifest_path = topk_run_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest.json: {manifest_path}")
    import json
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("format") != "token_topk_sparse_v1":
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')!r}")
    return manifest


def _iter_shards(topk_run_dir: Path, manifest: dict, *, desc: str | None = None):
    shards = manifest["shards"]
    it = shards
    if tqdm is not None and desc:
        it = tqdm(shards, desc=desc, unit="shard")
    for shard_meta in it:
        payload = torch.load(topk_run_dir / shard_meta["path"], map_location="cpu")
        yield payload


def _top_pairs(vec: torch.Tensor, top_n: int) -> list[dict]:
    k = min(top_n, int(vec.numel()))
    values, indices = torch.topk(vec, k=k, largest=True)
    return [{"feature_id": int(i), "score": float(v)} for i, v in zip(indices.tolist(), values.tolist())]




# ---------------------------------------------------------------------------
# Ranking implementations
# ---------------------------------------------------------------------------


def _load_matrix(scores_pt_path: Path, key: str) -> tuple[torch.Tensor, list[dict]]:
    """Load a named matrix + row_keys from a score artifact. Accepts both
    the OpenPI-style payload with `matrix_raw / matrix_window_mean /
    matrix_task_mean` and the legacy single-`matrix` payload."""
    payload = torch.load(Path(scores_pt_path).resolve(), map_location="cpu")
    if key in payload:
        return payload[key].to(dtype=torch.float32), list(payload["row_keys"])
    # Legacy: only `matrix` (= event_aligned matrix_raw). Used to be called
    # `matrix` before the openpi-mech refactor; remap the three keys to the
    # legacy single matrix so downstream still works on old artifacts.
    if "matrix" in payload:
        return payload["matrix"].to(dtype=torch.float32), list(payload["row_keys"])
    raise KeyError(f"Score artifact missing '{key}' (and no legacy 'matrix'): {scores_pt_path}")


def _load_event_aligned_matrix(scores_pt_path: Path) -> tuple[torch.Tensor, list[dict]]:
    return _load_matrix(scores_pt_path, "matrix_raw")


def event_aligned_top_features_per_row(scores_pt_path: Path, top_n: int) -> list[dict]:
    """Read the score matrix produced by ``score_cluster_features.py`` and
    return per-cluster-row top-N features."""
    matrix, row_keys = _load_event_aligned_matrix(scores_pt_path)
    out: list[dict] = []
    for row_idx, meta in enumerate(row_keys):
        out.append(
            {
                "ranking": "event_aligned",
                "task_description": str(meta["task_description"]),
                "cluster_id": str(meta["cluster_id"]),
                "phrase": str(meta.get("phrase", "")),
                "phase": str(meta.get("phase", "")),
                "top_features": _top_pairs(matrix[row_idx], top_n),
            }
        )
    return out


def event_aligned_suite_top_k(
    scores_pt_path: Path, top_k: int, *, min_coverage: float = 0.5
) -> list[dict]:
    """Suite-level top-K event-aligned features: mean of the score matrix
    over **canonical** rows (``episode_coverage >= min_coverage``), then
    top-K. Matches mechanistic-steering-vlas suite-config generator,
    which averages over the canonical-filtered matrix (default
    ``min_coverage=0.5``)."""
    matrix, row_keys = _load_event_aligned_matrix(scores_pt_path)
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    suite_vec = matrix[keep_idx].mean(dim=0)
    return _top_pairs(suite_vec, top_k)


def window_mean_top_features_per_row(
    *,
    scores_pt_path: Path,
    top_n: int,
) -> list[dict]:
    """Per cluster row, top-N features by mean SAE activation over the
    cluster's event windows. Reads the pre-computed ``matrix_window_mean``
    from the score artifact (built by
    ``event_sae.scoring.score_matrix.score_cluster_features``)."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_window_mean")
    out: list[dict] = []
    for row_idx, meta in enumerate(row_keys):
        if float(matrix[row_idx].abs().sum().item()) == 0.0:
            continue
        out.append(
            {
                "ranking": "window_mean",
                "task_description": str(meta["task_description"]),
                "cluster_id": str(meta["cluster_id"]),
                "phrase": str(meta.get("phrase", "")),
                "phase": str(meta.get("phase", "")),
                "top_features": _top_pairs(matrix[row_idx], top_n),
            }
        )
    return out


def window_mean_suite_top_k(
    *,
    scores_pt_path: Path,
    top_k: int,
    min_coverage: float = 0.5,
) -> list[dict]:
    """Suite-level top-K window-mean features: ``num_events``-weighted
    mean of per-row pre-computed ``matrix_window_mean`` vectors,
    restricted to canonical rows (``episode_coverage >= min_coverage``),
    then top-K."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_window_mean")
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    matrix = matrix[keep_idx]
    num_events_per_row = [int(row.get("num_events", 0)) for row in row_keys]
    weights = torch.tensor(
        [float(num_events_per_row[i]) for i in keep_idx], dtype=torch.float32
    )
    if float(weights.sum().item()) <= 0:
        raise RuntimeError("Total event-weight is zero for window-mean aggregation.")
    suite_vec = (matrix * weights[:, None]).sum(dim=0) / weights.sum()
    return _top_pairs(suite_vec, top_k)




def task_mean_top_features_per_task(
    *,
    scores_pt_path: Path,
    top_n: int,
) -> list[dict]:
    """Per task, top-N features by mean SAE activation across every rollout
    step. Reads pre-computed ``matrix_task_mean`` from the score artifact.
    Note: rows here are per-cluster (each cluster broadcasts its task's
    mean), so we dedupe by task_description for the per-task ranking."""
    matrix, row_keys = _load_matrix(scores_pt_path, "matrix_task_mean")
    seen: dict[str, int] = {}
    for i, meta in enumerate(row_keys):
        task = str(meta.get("task_description", ""))
        if task and task not in seen:
            seen[task] = i
    out: list[dict] = []
    for task, idx in sorted(seen.items()):
        out.append(
            {
                "ranking": "task_mean",
                "task_description": task,
                "top_features": _top_pairs(matrix[idx], top_n),
            }
        )
    return out


def task_mean_suite_top_k(
    *,
    scores_pt_path: Path,
    top_k: int,
    min_coverage: float = 0.5,
) -> list[dict]:
    """Suite-level top-K task-mean features. Mirrors openpi-mech's
    ``_task_mean_suite_vector``: dedupe canonical cluster rows by
    ``task_description``, then take a per-task-timestep-count-weighted
    mean across unique tasks."""
    payload = torch.load(Path(scores_pt_path).resolve(), map_location="cpu")
    matrix = payload.get("matrix_task_mean")
    if matrix is None:
        matrix = payload["matrix"]
    matrix = matrix.to(dtype=torch.float32)
    row_keys = list(payload["row_keys"])
    keep_idx = [
        i
        for i, row in enumerate(row_keys)
        if float(row.get("episode_coverage", 0.0)) >= min_coverage
    ]
    if not keep_idx:
        raise RuntimeError(
            f"No canonical rows with episode_coverage >= {min_coverage}; "
            f"total rows={len(row_keys)}."
        )
    task_timestep_counts = (
        payload.get("selection_counts", {}).get("task_timestep_counts", {}) or {}
    )

    seen_tasks: set[str] = set()
    vectors: list[torch.Tensor] = []
    weights: list[float] = []
    for i in keep_idx:
        meta = row_keys[i]
        task_desc = str(meta.get("task_description", ""))
        if task_desc in seen_tasks:
            continue
        seen_tasks.add(task_desc)
        vectors.append(matrix[i])
        task_id = meta.get("task_id")
        weight = task_timestep_counts.get(task_id)
        if weight is None and task_id is not None:
            weight = task_timestep_counts.get(str(task_id))
        weights.append(float(weight) if weight else 1.0)
    if not vectors:
        raise RuntimeError("No tasks remained for task_mean suite aggregation.")
    stacked = torch.stack(vectors, dim=0)
    weight_t = torch.tensor(weights, dtype=torch.float32)
    if float(weight_t.sum().item()) <= 0:
        raise RuntimeError("Total task-timestep weight is zero for task_mean.")
    suite_vec = (stacked * weight_t[:, None]).sum(dim=0) / weight_t.sum()
    return _top_pairs(suite_vec, top_k)


def alive_feature_ids(topk_run_dir: Path) -> set[int]:
    """Set of feature ids that fire at least once anywhere in the run."""
    topk_run_dir = Path(topk_run_dir).resolve()
    manifest = _load_manifest(topk_run_dir)
    alive: set[int] = set()
    for payload in _iter_shards(topk_run_dir, manifest, desc="alive scan"):
        values = payload["top_feature_vals"]
        ids = payload["top_feature_ids"].to(dtype=torch.int64)[values > 0].tolist()
        alive.update(ids)
    return alive


def random_alive_features(
    *,
    topk_run_dir: Path,
    num_features: int,
    exclude_feature_ids: set[int],
    seed: int = 0,
) -> list[int]:
    """Uniform-random sample of ``num_features`` alive features, excluding
    any feature already selected by the informed rankings. Paper Section
    4.4 random-alive control."""
    alive = alive_feature_ids(topk_run_dir)
    candidates = sorted(alive - set(int(x) for x in exclude_feature_ids))
    if len(candidates) < num_features:
        raise RuntimeError(
            f"Not enough alive features after exclusion: have {len(candidates)}, need {num_features}"
        )
    rng = random.Random(seed)
    return sorted(rng.sample(candidates, num_features))
