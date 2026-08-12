"""Event-feature score matrix.

Joins per-event SAE top-k activations (from `event_sae.openvla.activations`
or `event_sae.openpi.activations` either online or via
`scripts/extract_topk.py`) with VLM-annotated event clusters, builds
event-centered temporal windows around each event, and projects three
time templates (pulse, step-up, step-down) onto the per-feature
trajectory. The per-feature score is the maximum positive projection
across templates; per-event scores are averaged within each
`(cluster, episode)` group and then across episodes to give one row per
cluster.

Three matrices are produced per call, mirroring openpi-mech's
`build_openpi_feature_score_matrix.py`:

  - ``matrix_raw``         — `max(pulse, step_up, step_down)` over the
                             3 templates (event_aligned score)
  - ``matrix_window_mean`` — mean activation over each event's window
                             (window_mean score)
  - ``matrix_task_mean``   — per-task mean activation over every cached
                             timestep (task_mean score)

The ``step_mapping`` argument controls how shard rows map to env
timesteps (mirrors openpi-mech):

  - ``action_executed`` (OpenPI AE default): one env step per row at
    ``chunk_start + token_idx`` when the token is executed.
  - ``chunk_executed`` (OpenPI PG default): broadcast each row to every
    executed env step of its chunk
    (``chunk_start..chunk_start+executed_chunk_len-1``).
  - ``inference_step`` (OpenVLA legacy / fallback): use ``step_in_episode``
    directly (no chunk semantics).

Output payload (single torch.save .pt):

  - `matrix_raw` / `matrix_window_mean` / `matrix_task_mean`
  - `matrix`               : alias of `matrix_raw` for backward compat
  - `row_keys`             : per-row cluster metadata
  - `row_results`          : per-row top-N feature summaries
  - `templates`            : the three time templates used
  - `selection_counts`     : join + filter accounting
  - `selected_events`      : per-event provenance after scoring
  - `step_mapping`         : the mapping used
  - `source`               : input paths + manifest summary
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from event_sae.events.io import load_jsonl


# ---------------------------------------------------------------------------
# Templates + projections
# ---------------------------------------------------------------------------


def _normalize_template(template: torch.Tensor) -> torch.Tensor:
    template = template.to(dtype=torch.float32)
    template = template - torch.mean(template)
    norm = torch.linalg.vector_norm(template)
    if float(norm) == 0.0:
        raise ValueError("Template norm is zero after mean-centering.")
    return template / norm


def build_templates(window_size: int) -> dict[str, torch.Tensor]:
    """Build the three normalized time templates used for event scoring."""
    positions = torch.arange(-window_size, window_size + 1, dtype=torch.float32)
    pulse = 1.0 - torch.abs(positions) / float(window_size + 1)
    pulse = _normalize_template(pulse)
    step_up = torch.where(positions < 0, -torch.ones_like(positions), torch.ones_like(positions))
    step_up = _normalize_template(step_up)
    step_down = -step_up
    return {"pulse": pulse, "step_up": step_up, "step_down": step_down}


def _project_pattern_scores(
    centered_matrix: torch.Tensor,
    templates: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    centered_signal = centered_matrix - torch.mean(centered_matrix, dim=0, keepdim=True)
    return {
        name: torch.clamp(torch.matmul(centered_signal.transpose(0, 1), template), min=0.0)
        for name, template in templates.items()
    }


def _top_features(scores: torch.Tensor, top_n: int) -> tuple[list[int], list[float]]:
    k = min(top_n, int(scores.numel()))
    values, indices = torch.topk(scores, k=k, largest=True)
    return indices.tolist(), [float(x) for x in values.tolist()]


def _row_top_summary(scores: torch.Tensor, top_n: int) -> dict[str, list]:
    feature_ids, values = _top_features(scores, top_n)
    return {"feature_ids": feature_ids, "scores": values}


# ---------------------------------------------------------------------------
# Window placement + per-step action-token aggregation
# ---------------------------------------------------------------------------


def _fit_step_window(
    *,
    center_step: int,
    window_size: int,
    num_steps: int,
) -> tuple[list[int] | None, list[int], int | None]:
    """Shift a (2w+1)-step centered window to stay inside [0, num_steps)."""
    requested_steps = list(range(center_step - window_size, center_step + window_size + 1))
    if not requested_steps:
        return [], [], 0
    min_req, max_req = min(requested_steps), max(requested_steps)
    if (max_req - min_req) >= num_steps:
        return None, requested_steps, None
    shift = 0
    if max_req >= num_steps:
        shift -= max_req - (num_steps - 1)
    if min_req + shift < 0:
        shift += -(min_req + shift)
    return [s + shift for s in requested_steps], requested_steps, shift


def build_templates_at_event_idx(window_size: int, event_idx: int) -> dict[str, torch.Tensor]:
    """Boundary-aware templates: when the event window is shifted to stay
    inside ``[0, num_steps)``, the event position inside the window may not
    be the center any more. Templates are re-centered around
    ``event_idx`` so pulse / step-up / step-down stay anchored on the
    waypoint. Matches openpi-mech's ``_build_templates_at_event_idx``."""
    if event_idx < 0 or event_idx > 2 * window_size:
        raise ValueError(f"event_idx={event_idx} outside window length {2 * window_size + 1}.")
    positions = torch.arange(2 * window_size + 1, dtype=torch.float32) - float(event_idx)
    pulse = torch.clamp(1.0 - torch.abs(positions) / float(window_size + 1), min=0.0)
    pulse = _normalize_template(pulse)
    step_up = torch.where(positions < 0, -torch.ones_like(positions), torch.ones_like(positions))
    step_up = _normalize_template(step_up)
    step_down = -step_up
    return {"pulse": pulse, "step_up": step_up, "step_down": step_down}


def _effective_steps_for_row(
    *,
    step_mapping: str,
    step_in_episode: int,
    token_idx: int,
    chunk_start_step: int,
    executed_chunk_len: int,
) -> list[int]:
    """Translate a topk-shard row into the list of env steps it represents.

    Matches openpi-mech ``build_openpi_feature_score_matrix.py::_effective_steps``.
    ``chunk_start_step`` and ``executed_chunk_len`` may be ``-1`` sentinels
    on OpenVLA shards (no chunking); in that case only ``inference_step``
    mode is valid.
    """
    if step_mapping == "inference_step":
        return [step_in_episode] if step_in_episode >= 0 else []
    if step_mapping == "action_executed":
        if chunk_start_step < 0 or executed_chunk_len <= 0:
            return []
        if token_idx < 0 or token_idx >= executed_chunk_len:
            return []
        return [chunk_start_step + token_idx]
    if step_mapping == "chunk_executed":
        if chunk_start_step < 0 or executed_chunk_len <= 0:
            return []
        return [chunk_start_step + offset for offset in range(int(executed_chunk_len))]
    raise ValueError(f"Unsupported step_mapping={step_mapping!r}")


def _default_step_mapping(capture_target: str | None) -> str:
    """Backend default: AE → action_executed (per-token executed env step);
    PG → chunk_executed (prefix conditions whole executed chunk); anything
    else → inference_step (OpenVLA legacy)."""
    if capture_target == "action_expert":
        return "action_executed"
    if capture_target == "paligemma":
        return "chunk_executed"
    return "inference_step"


# ---------------------------------------------------------------------------
# Cluster / event join
# ---------------------------------------------------------------------------


@dataclass
class _JoinResult:
    selected_events: list[dict]
    cluster_metadata_by_id: dict[str, dict]
    counts: dict[str, int]


def join_cluster_events(
    *,
    event_features: list[dict],
    cluster_assignments: list[dict],
    cluster_annotations: list[dict] | None = None,
    clusters: list[dict] | None = None,
) -> _JoinResult:
    """Join events to clusters; VLM labels are descriptive, not membership.

    When ``clusters`` is supplied, its clustering output is authoritative and
    a missing/invalid Gemini annotation does not silently remove an event from
    feature ranking. Passing annotations alone retains the legacy behavior.
    """
    event_by_sample_id = {}
    for record in event_features:
        sample_id = str(record["sample_id"])
        if sample_id in event_by_sample_id:
            raise ValueError(f"Duplicate sample_id in event_features.jsonl: {sample_id}")
        event_by_sample_id[sample_id] = record

    counts = {
        "valid_clusters": 0,
        "joined_events": 0,
        "skipped_api_error": 0,
        "skipped_parse_error": 0,
        "skipped_empty_phrase": 0,
        "skipped_empty_phase": 0,
        "skipped_missing_cluster_annotation": 0,
        "skipped_missing_event_features": 0,
        "clusters_without_valid_annotation": 0,
    }

    cluster_metadata_by_id: dict[str, dict] = {}
    for cluster in clusters or []:
        cluster_id = str(cluster["cluster_id"])
        if cluster_id in cluster_metadata_by_id:
            raise ValueError(f"Duplicate cluster_id in clusters: {cluster_id}")
        cluster_metadata_by_id[cluster_id] = {
            "cluster_id": cluster_id,
            "task_description": str(cluster["task_description"]),
            "phrase": cluster_id,
            "phase": "unlabeled",
            "episode_coverage": float(cluster.get("episode_coverage", 0.0)),
            "model": "",
            "prompt_version": "",
            "annotation_valid": False,
            "representative_sample_ids": list(cluster.get("representative_sample_ids", [])),
            "representative_clip_paths": list(cluster.get("representative_clip_paths", [])),
            "representative_frame_paths": list(cluster.get("representative_frame_paths", [])),
            "representative_progress_percents": list(
                cluster.get("representative_progress_percents", [])
            ),
        }

    for annotation in cluster_annotations or []:
        cluster_id = str(annotation["cluster_id"])
        if annotation.get("api_error") is not None:
            counts["skipped_api_error"] += 1
            continue
        if annotation.get("parse_error") is not None:
            counts["skipped_parse_error"] += 1
            continue
        phrase = str(annotation.get("phrase", "")).strip()
        if not phrase:
            counts["skipped_empty_phrase"] += 1
            continue
        phase = str(annotation.get("phase", "")).strip()
        if not phase:
            counts["skipped_empty_phase"] += 1
            continue
        if clusters is None and cluster_id in cluster_metadata_by_id:
            raise ValueError(f"Duplicate cluster_id in cluster annotations: {cluster_id}")
        if clusters is not None and cluster_id not in cluster_metadata_by_id:
            counts["skipped_missing_cluster_annotation"] += 1
            continue
        annotation_meta = {
            "cluster_id": cluster_id,
            "task_description": str(annotation["task_description"]),
            "phrase": phrase,
            "phase": phase,
            "episode_coverage": float(annotation.get("episode_coverage", 0.0)),
            "model": str(annotation.get("model", "")),
            "prompt_version": str(annotation.get("prompt_version", "")),
            "annotation_valid": True,
            "representative_sample_ids": list(annotation.get("representative_sample_ids", [])),
            "representative_clip_paths": list(annotation.get("representative_clip_paths", [])),
            "representative_frame_paths": list(annotation.get("representative_frame_paths", [])),
            "representative_progress_percents": list(annotation.get("representative_progress_percents", [])),
        }
        if clusters is None:
            cluster_metadata_by_id[cluster_id] = annotation_meta
        else:
            base = cluster_metadata_by_id[cluster_id]
            if base["task_description"] != annotation_meta["task_description"]:
                raise ValueError(f"Task description mismatch for cluster_id={cluster_id}")
            base.update(annotation_meta)
    counts["clusters_without_valid_annotation"] = sum(
        not bool(meta.get("annotation_valid")) for meta in cluster_metadata_by_id.values()
    )
    counts["valid_clusters"] = len(cluster_metadata_by_id)

    joined_events: list[dict] = []
    for assignment in cluster_assignments:
        cluster_id = str(assignment["cluster_id"])
        cluster_meta = cluster_metadata_by_id.get(cluster_id)
        if cluster_meta is None:
            counts["skipped_missing_cluster_annotation"] += 1
            continue
        sample_id = str(assignment["sample_id"])
        event = event_by_sample_id.get(sample_id)
        if event is None:
            counts["skipped_missing_event_features"] += 1
            continue
        if str(event["task_description"]) != str(assignment["task_description"]):
            raise ValueError(f"Task description mismatch for sample_id={sample_id}")
        joined_events.append(
            {
                "sample_id": sample_id,
                "task_description": str(event["task_description"]),
                "task_id": int(event["task_id"]),
                "task_episode_idx": int(event["task_episode_idx"]),
                "episode_num": int(event["episode_num"]),
                "waypoint_rank": int(event["waypoint_rank"]),
                "waypoint_step": int(event["waypoint_step"]),
                "progress_percent": float(event["progress_percent"]),
                "num_steps": int(event["num_steps"]),
                "cluster_id": cluster_id,
                "phrase": cluster_meta["phrase"],
                "phase": cluster_meta["phase"],
            }
        )
    counts["joined_events"] = len(joined_events)

    member_counts: dict[str, int] = defaultdict(int)
    episode_sets: dict[str, set[int]] = defaultdict(set)
    for event in joined_events:
        member_counts[event["cluster_id"]] += 1
        episode_sets[event["cluster_id"]].add(int(event["episode_num"]))
    for cluster_id, meta in cluster_metadata_by_id.items():
        meta["num_members"] = int(member_counts.get(cluster_id, 0))
        meta["num_episodes"] = int(len(episode_sets.get(cluster_id, set())))

    return _JoinResult(joined_events, cluster_metadata_by_id, counts)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _load_manifest(topk_run_dir: Path) -> dict:
    """Find a `token_topk_sparse_v1` manifest under ``topk_run_dir`` or any
    immediate ``sae_activations/<submodule>/`` subdir (online OpenPI puts
    shards in the subdir; offline extract_topk writes them flat)."""
    manifest_path = topk_run_dir / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        if manifest.get("format") != "token_topk_sparse_v1":
            raise ValueError(f"Unsupported manifest format: {manifest.get('format')!r}")
        return manifest
    for sub in (topk_run_dir / "sae_activations").glob("*"):
        cand = sub / "manifest.json"
        if cand.is_file():
            with cand.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("format") == "token_topk_sparse_v1":
                return manifest
    raise FileNotFoundError(f"No token_topk_sparse_v1 manifest under {topk_run_dir}")


def _resolve_shard_path(topk_run_dir: Path, shard_relpath: str) -> Path:
    """Shard paths in manifest may be relative to either the run root or
    the legacy ``sae_activations/<submodule>`` subdir. Resolve to absolute."""
    direct = topk_run_dir / shard_relpath
    if direct.is_file():
        return direct
    for sub in (topk_run_dir / "sae_activations").glob("*"):
        cand = sub / shard_relpath
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"Shard {shard_relpath} not under {topk_run_dir}")


def _load_timestep_vectors(
    topk_run_dir: Path,
    *,
    step_mapping: str,
    episode_to_task_id: dict[int, int],
    task_id_set: set[int],
    dict_size: int,
    required_timestep_keys: set[tuple[int, int]] | None = None,
) -> tuple[
    dict[tuple[int, int], torch.Tensor],
    dict[int, torch.Tensor],
    dict[int, int],
    dict,
    dict[str, int],
]:
    """Walk topk shards and accumulate per-``(episode, env_step)`` dense
    vectors under ``step_mapping``. Mirrors openpi-mech's
    ``_load_timestep_vectors``:

    * Filter by ``task_id`` (read EVERY episode in any selected task,
      not just episodes that produced clustered events). This matters
      because ``matrix_task_mean`` is the per-task mean over **all**
      rollout timesteps the cache covers, not only event-window ones.
    * Compute per-timestep means first, then aggregate task means as
      the mean of per-timestep means weighted equally by timestep — not
      by row count. Under ``chunk_executed`` a single row contributes to
      multiple timesteps; row-count weighting overcounts wide chunks.
    """
    manifest = _load_manifest(topk_run_dir)
    counters = {
        "shards_loaded": 0,
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped_unknown_task": 0,
        "rows_skipped_nonexecuted": 0,
        "rows_skipped_no_effective_step": 0,
    }
    # OpenVLA rows are already ordered by (episode, environment step). Stream
    # one timestep at a time so task means do not require a dense 32,768-D
    # vector for every timestep in the 500-rollout dataset. Only event-window
    # timesteps are retained for later scoring.
    if step_mapping == "inference_step":
        return _load_openvla_timestep_vectors_streaming(
            topk_run_dir,
            manifest=manifest,
            episode_to_task_id=episode_to_task_id,
            task_id_set=task_id_set,
            dict_size=dict_size,
            required_timestep_keys=required_timestep_keys,
            counters=counters,
        )

    timestep_sums: dict[tuple[int, int], torch.Tensor] = {}
    timestep_counts: dict[tuple[int, int], int] = defaultdict(int)
    timestep_task_ids: dict[tuple[int, int], int] = {}

    shard_iter = manifest["shards"]
    if tqdm is not None:
        shard_iter = tqdm(shard_iter, desc="Loading topk shards", unit="shard")
    for shard_meta in shard_iter:
        shard_path = _resolve_shard_path(topk_run_dir, shard_meta["path"])
        payload = torch.load(shard_path, map_location="cpu")
        counters["shards_loaded"] += 1
        ep_arr = payload["episode_num"].to(dtype=torch.int64)
        step_arr = payload["step_in_episode"].to(dtype=torch.int64)
        tok_arr = payload["token_idx"].to(dtype=torch.int64)
        chunk_start_arr = payload.get("chunk_start_step")
        if chunk_start_arr is None:
            chunk_start_arr = torch.full_like(ep_arr, -1)
        else:
            chunk_start_arr = chunk_start_arr.to(dtype=torch.int64)
        exec_len_arr = payload.get("executed_chunk_len")
        if exec_len_arr is None:
            exec_len_arr = torch.full_like(ep_arr, -1)
        else:
            exec_len_arr = exec_len_arr.to(dtype=torch.int64)
        feat_ids = payload["top_feature_ids"].to(dtype=torch.int64)
        feat_vals = payload["top_feature_vals"].to(dtype=torch.float32)
        n_rows = int(ep_arr.shape[0])
        counters["rows_seen"] += n_rows

        for row_idx in range(n_rows):
            ep = int(ep_arr[row_idx])
            task_id = episode_to_task_id.get(ep)
            if task_id is None or task_id not in task_id_set:
                counters["rows_skipped_unknown_task"] += 1
                continue
            steps = _effective_steps_for_row(
                step_mapping=step_mapping,
                step_in_episode=int(step_arr[row_idx]),
                token_idx=int(tok_arr[row_idx]),
                chunk_start_step=int(chunk_start_arr[row_idx]),
                executed_chunk_len=int(exec_len_arr[row_idx]),
            )
            if not steps:
                if step_mapping == "action_executed":
                    counters["rows_skipped_nonexecuted"] += 1
                else:
                    counters["rows_skipped_no_effective_step"] += 1
                continue
            row_indices = feat_ids[row_idx]
            row_values = feat_vals[row_idx]
            for step in steps:
                if step < 0:
                    counters["rows_skipped_no_effective_step"] += 1
                    continue
                key = (ep, step)
                vec = timestep_sums.get(key)
                if vec is None:
                    vec = torch.zeros(dict_size, dtype=torch.float32)
                    timestep_sums[key] = vec
                    timestep_task_ids[key] = task_id
                vec.index_add_(0, row_indices, row_values)
                timestep_counts[key] += 1
                counters["rows_used"] += 1

    # Normalize in place. Keeping a second dense vector dictionary here can
    # double host RAM for a full Spatial run, while sums are no longer needed.
    timestep_vectors = timestep_sums
    for key, vec_sum in timestep_vectors.items():
        c = timestep_counts[key]
        if c > 0:
            vec_sum.div_(float(c))

    # Per-task mean of per-timestep vectors. Mirrors openpi-mech.
    task_sums: dict[int, torch.Tensor] = {}
    task_counts: dict[int, int] = defaultdict(int)
    for key, vec in timestep_vectors.items():
        tid = timestep_task_ids[key]
        if tid not in task_sums:
            task_sums[tid] = torch.zeros(dict_size, dtype=torch.float32)
        task_sums[tid] += vec
        task_counts[tid] += 1
    task_means: dict[int, torch.Tensor] = {}
    for tid, vec_sum in task_sums.items():
        c = task_counts[tid]
        if c > 0:
            task_means[tid] = vec_sum / float(c)

    return timestep_vectors, task_means, dict(task_counts), manifest, counters


def _load_openvla_timestep_vectors_streaming(
    topk_run_dir: Path,
    *,
    manifest: dict,
    episode_to_task_id: dict[int, int],
    task_id_set: set[int],
    dict_size: int,
    required_timestep_keys: set[tuple[int, int]] | None,
    counters: dict[str, int],
):
    """Vectorized OpenVLA aggregation with bounded host memory.

    Dense activation and offline Top-K shards preserve rollout order. We
    validate that invariant and finalize each timestep once, retaining only
    timesteps used by event windows while still computing task means over the
    full dataset.
    """
    timestep_vectors: dict[tuple[int, int], torch.Tensor] = {}
    task_sums: dict[int, torch.Tensor] = {}
    task_counts: dict[int, int] = defaultdict(int)
    pending_key: tuple[int, int] | None = None
    pending_sum: torch.Tensor | None = None
    pending_count = 0
    previous_key: tuple[int, int] | None = None
    max_episode = max(episode_to_task_id, default=0)
    episode_task_lookup = torch.full((max_episode + 1,), -1, dtype=torch.int64)
    for episode, task_id in episode_to_task_id.items():
        if 0 <= episode <= max_episode:
            episode_task_lookup[episode] = int(task_id)
    allowed_tasks = torch.zeros(
        max(max(task_id_set, default=0) + 1, 1), dtype=torch.bool
    )
    for task_id in task_id_set:
        if task_id >= 0:
            allowed_tasks[task_id] = True

    def finalize() -> None:
        nonlocal pending_key, pending_sum, pending_count
        if pending_key is None or pending_sum is None or pending_count <= 0:
            return
        mean = pending_sum / float(pending_count)
        task_id = episode_to_task_id[pending_key[0]]
        if task_id not in task_sums:
            task_sums[task_id] = torch.zeros(dict_size, dtype=torch.float32)
        task_sums[task_id].add_(mean)
        task_counts[task_id] += 1
        if required_timestep_keys is None or pending_key in required_timestep_keys:
            timestep_vectors[pending_key] = mean
        pending_key = None
        pending_sum = None
        pending_count = 0

    shard_iter = manifest["shards"]
    if tqdm is not None:
        shard_iter = tqdm(shard_iter, desc="Streaming OpenVLA topk shards", unit="shard")
    for shard_meta in shard_iter:
        payload = torch.load(
            _resolve_shard_path(topk_run_dir, shard_meta["path"]), map_location="cpu"
        )
        counters["shards_loaded"] += 1
        episodes = payload["episode_num"].to(dtype=torch.int64)
        steps = payload["step_in_episode"].to(dtype=torch.int64)
        feature_ids = payload["top_feature_ids"].to(dtype=torch.int64)
        feature_values = payload["top_feature_vals"].to(dtype=torch.float32)
        counters["rows_seen"] += int(episodes.shape[0])

        episode_in_range = (episodes >= 0) & (episodes <= max_episode)
        row_task_ids = torch.full_like(episodes, -1)
        row_task_ids[episode_in_range] = episode_task_lookup[episodes[episode_in_range]]
        task_in_range = (row_task_ids >= 0) & (row_task_ids < len(allowed_tasks))
        task_allowed = torch.zeros_like(task_in_range)
        task_allowed[task_in_range] = allowed_tasks[row_task_ids[task_in_range]]
        valid_task = episode_in_range & task_allowed
        valid = valid_task & (steps >= 0)
        counters["rows_skipped_unknown_task"] += int((~valid_task).sum().item())
        counters["rows_skipped_no_effective_step"] += int(
            (valid_task & (steps < 0)).sum().item()
        )
        if not bool(valid.any()):
            continue
        episodes = episodes[valid]
        steps = steps[valid]
        feature_ids = feature_ids[valid]
        feature_values = feature_values[valid]
        counters["rows_used"] += int(episodes.shape[0])

        pairs = torch.stack((episodes, steps), dim=1)
        unique_pairs, inverse, row_counts = torch.unique_consecutive(
            pairs, dim=0, return_inverse=True, return_counts=True
        )
        group_sums = torch.zeros(
            (int(unique_pairs.shape[0]), dict_size), dtype=torch.float32
        )
        group_indices = inverse[:, None].expand_as(feature_ids).reshape(-1)
        group_sums.index_put_(
            (group_indices, feature_ids.reshape(-1)),
            feature_values.reshape(-1),
            accumulate=True,
        )

        for group_idx, pair in enumerate(unique_pairs.tolist()):
            key = (int(pair[0]), int(pair[1]))
            if previous_key is not None and key < previous_key:
                raise ValueError(
                    "OpenVLA Top-K rows are not ordered by episode/step; "
                    "streaming aggregation would be unsafe."
                )
            previous_key = key
            group_sum = group_sums[group_idx]
            group_count = int(row_counts[group_idx])
            if key == pending_key:
                pending_sum.add_(group_sum)
                pending_count += group_count
            else:
                finalize()
                pending_key = key
                pending_sum = group_sum.clone()
                pending_count = group_count
    finalize()

    task_means = {
        task_id: value / float(task_counts[task_id])
        for task_id, value in task_sums.items()
        if task_counts[task_id] > 0
    }
    counters["timestep_vectors_retained"] = len(timestep_vectors)
    counters["timestep_vectors_total"] = sum(task_counts.values())
    return timestep_vectors, task_means, dict(task_counts), manifest, counters


def score_cluster_features(
    *,
    topk_run_dir: Path,
    event_features_path: Path,
    cluster_assignments_path: Path,
    cluster_annotations_path: Path | None,
    clusters_path: Path | None = None,
    output_path: Path,
    window_size: int = 5,
    top_n: int = 20,
    step_mapping: str = "auto",
    prompt_records_path: Path | None = None,
) -> dict:
    """Build the event-feature score matrices and save a single `.pt`
    payload. Mirrors openpi-mech ``build_openpi_feature_score_matrix.py``.

    Three matrices are produced (paper Table 4 / Fig 3):

      * ``matrix_raw``: max-over-templates projection (event_aligned)
      * ``matrix_window_mean``: mean activation over the event window
      * ``matrix_task_mean``: per-task mean activation across all
        timesteps the cache covers

    ``step_mapping`` defaults to ``"auto"``: pick per ``manifest.capture_target``
    (``action_executed`` for ``action_expert``, ``chunk_executed`` for
    ``paligemma``, otherwise ``inference_step``).
    """
    topk_run_dir = Path(topk_run_dir).resolve()
    event_features_path = Path(event_features_path).resolve()
    cluster_assignments_path = Path(cluster_assignments_path).resolve()
    cluster_annotations_path = (
        Path(cluster_annotations_path).resolve() if cluster_annotations_path is not None else None
    )
    clusters_path = Path(clusters_path).resolve() if clusters_path is not None else None
    if cluster_annotations_path is None and clusters_path is None:
        raise ValueError("Provide --clusters-path and/or --cluster-annotations-path.")
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Peek manifest for capture_target before joining, so step_mapping
    # auto-detect runs before any aggregation.
    manifest = _load_manifest(topk_run_dir)
    dict_size = int(manifest["dict_size"])
    capture_target = manifest.get("capture_target")
    if step_mapping == "auto":
        step_mapping = _default_step_mapping(capture_target)

    join = join_cluster_events(
        event_features=load_jsonl(event_features_path),
        cluster_assignments=load_jsonl(cluster_assignments_path),
        cluster_annotations=(
            load_jsonl(cluster_annotations_path) if cluster_annotations_path is not None else None
        ),
        clusters=load_jsonl(clusters_path) if clusters_path is not None else None,
    )
    if not join.selected_events:
        raise RuntimeError("No usable clustered events after the join.")

    w = window_size
    usable_events: list[dict] = []
    skipped_window = 0
    shifted_window_count = 0
    event_episode_to_task_id: dict[int, int] = {}
    for event in join.selected_events:
        episode = int(event["episode_num"])
        num_steps = int(event["num_steps"])
        window_steps, requested, shift = _fit_step_window(
            center_step=int(event["waypoint_step"]), window_size=w, num_steps=num_steps
        )
        if window_steps is None:
            skipped_window += 1
            continue
        event_idx_in_window = w - int(shift)
        if event_idx_in_window < 0 or event_idx_in_window > 2 * w:
            skipped_window += 1
            continue
        if shift != 0:
            shifted_window_count += 1
        event = dict(event)
        event.update(
            {
                "window_steps": window_steps,
                "requested_steps": requested,
                "window_shift": int(shift),
                "event_idx_in_window": int(event_idx_in_window),
            }
        )
        usable_events.append(event)
        event_episode_to_task_id[episode] = int(event["task_id"])
    if not usable_events:
        raise RuntimeError("No events remained after centered-window filtering.")

    # Episode → task_id for the WHOLE run (not only event-window episodes).
    # Paper's matrix_task_mean is the per-task mean over every cached
    # timestep, so we need a full episode mapping. Prefer prompt_records;
    # fall back to the event-only mapping if not provided (paper-style
    # task_mean will be approximated).
    episode_to_task_id: dict[int, int] = dict(event_episode_to_task_id)
    if prompt_records_path is not None:
        for record in load_jsonl(Path(prompt_records_path).resolve()):
            ep = int(record["episode_num"])
            tid = int(record["task_id"])
            episode_to_task_id[ep] = tid
    task_id_set: set[int] = set(event_episode_to_task_id.values())

    required_timestep_keys = {
        (int(event["episode_num"]), int(step))
        for event in usable_events
        for step in event["window_steps"]
    }
    timestep_vectors, task_means, task_counts, _manifest, load_counters = _load_timestep_vectors(
        topk_run_dir,
        step_mapping=step_mapping,
        episode_to_task_id=episode_to_task_id,
        task_id_set=task_id_set,
        dict_size=dict_size,
        required_timestep_keys=required_timestep_keys,
    )

    # ---- score per event ----
    episode_group_scores: dict[tuple[str, int], dict[str, list[torch.Tensor]]] = defaultdict(
        lambda: defaultdict(list)
    )
    episode_group_window_means: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    episode_group_event_counts: dict[tuple[str, int], int] = defaultdict(int)
    selected_event_payloads: list[dict] = []
    skipped_missing_window_vectors = 0

    event_iter = tqdm(usable_events, desc="Scoring events", unit="event") if tqdm is not None else usable_events
    for event in event_iter:
        episode = int(event["episode_num"])
        centered = torch.zeros((2 * w + 1, dict_size), dtype=torch.float32)
        ok = True
        for row_idx, step in enumerate(event["window_steps"]):
            vec = timestep_vectors.get((episode, int(step)))
            if vec is None:
                ok = False
                break
            centered[row_idx] = vec
        if not ok:
            skipped_missing_window_vectors += 1
            continue
        templates_for_event = build_templates_at_event_idx(w, int(event["event_idx_in_window"]))
        pattern_scores = _project_pattern_scores(centered, templates_for_event)
        group_key = (str(event["cluster_id"]), episode)
        for name, vec in pattern_scores.items():
            episode_group_scores[group_key][name].append(vec)
        episode_group_window_means[group_key].append(centered.mean(dim=0))
        episode_group_event_counts[group_key] += 1
        selected_event_payloads.append(
            {
                "sample_id": event["sample_id"],
                "task_description": event["task_description"],
                "task_id": event["task_id"],
                "task_episode_idx": event["task_episode_idx"],
                "episode_num": episode,
                "waypoint_rank": event["waypoint_rank"],
                "waypoint_step": event["waypoint_step"],
                "progress_percent": event["progress_percent"],
                "num_steps": event["num_steps"],
                "cluster_id": event["cluster_id"],
                "phrase": event["phrase"],
                "phase": event["phase"],
                "requested_steps": event["requested_steps"],
                "window_steps": event["window_steps"],
                "window_shift": event["window_shift"],
                "event_idx_in_window": event["event_idx_in_window"],
            }
        )
    if not selected_event_payloads:
        raise RuntimeError("No events remained after activation-window filtering.")

    # ---- aggregate per (cluster, episode) → per cluster ----
    row_scores: dict[str, list[torch.Tensor]] = defaultdict(list)
    row_window_means: dict[str, list[torch.Tensor]] = defaultdict(list)
    row_episode_counts: dict[str, int] = defaultdict(int)
    row_event_counts: dict[str, int] = defaultdict(int)
    for group_key, pattern_lists in episode_group_scores.items():
        cluster_id, _ep = group_key
        group_means = {
            name: torch.stack(score_list, dim=0).mean(dim=0)
            for name, score_list in pattern_lists.items()
        }
        combined = torch.maximum(
            group_means["pulse"], torch.maximum(group_means["step_up"], group_means["step_down"])
        )
        row_scores[cluster_id].append(combined)
        row_window_means[cluster_id].append(
            torch.stack(episode_group_window_means[group_key], dim=0).mean(dim=0)
        )
        row_episode_counts[cluster_id] += 1
        row_event_counts[cluster_id] += episode_group_event_counts[group_key]

    row_cluster_ids = sorted(
        row_scores,
        key=lambda cid: (
            str(join.cluster_metadata_by_id[cid]["task_description"]),
            str(cid),
        ),
    )
    num_rows = len(row_cluster_ids)
    matrix_raw = torch.zeros((num_rows, dict_size), dtype=torch.float32)
    matrix_window_mean = torch.zeros((num_rows, dict_size), dtype=torch.float32)
    matrix_task_mean = torch.zeros((num_rows, dict_size), dtype=torch.float32)
    cluster_to_task_id: dict[str, int] = {}
    for event in selected_event_payloads:
        cluster_to_task_id.setdefault(str(event["cluster_id"]), int(event["task_id"]))
    for row_idx, cluster_id in enumerate(row_cluster_ids):
        matrix_raw[row_idx] = torch.stack(row_scores[cluster_id], dim=0).mean(dim=0)
        matrix_window_mean[row_idx] = torch.stack(row_window_means[cluster_id], dim=0).mean(dim=0)
        task_id = cluster_to_task_id.get(cluster_id, -1)
        if task_id in task_means:
            matrix_task_mean[row_idx] = task_means[task_id]

    row_results = []
    for row_idx, cluster_id in enumerate(row_cluster_ids):
        meta = join.cluster_metadata_by_id[cluster_id]
        row_results.append(
            {
                "task_description": meta["task_description"],
                "cluster_id": cluster_id,
                "phrase": meta["phrase"],
                "phase": meta["phase"],
                "num_episode_groups": row_episode_counts[cluster_id],
                "num_events": row_event_counts[cluster_id],
                "episode_coverage": meta["episode_coverage"],
                "raw_top_features": _row_top_summary(matrix_raw[row_idx], top_n),
                "window_mean_top_features": _row_top_summary(matrix_window_mean[row_idx], top_n),
                "task_mean_top_features": _row_top_summary(matrix_task_mean[row_idx], top_n),
            }
        )

    payload = {
        "source": {
            "topk_run_dir": str(topk_run_dir),
            "event_features_path": str(event_features_path),
            "cluster_assignments_path": str(cluster_assignments_path),
            "cluster_annotations_path": (
                str(cluster_annotations_path) if cluster_annotations_path is not None else None
            ),
            "clusters_path": str(clusters_path) if clusters_path is not None else None,
            "dict_size": dict_size,
            "topk": int(manifest["topk"]),
            "layer": manifest.get("layer"),
            "sae_path": manifest.get("sae_path"),
            "capture_target": capture_target,
        },
        "window_size": window_size,
        "top_n": top_n,
        "step_mapping": step_mapping,
        "row_semantics": "(task_description, cluster_id, phrase, phase)",
        "score_definitions": {
            "pulse": "positive projection onto a symmetric local-peak template (event-centered) after time-centering",
            "step_up": "positive projection onto a low-to-high step template (event-centered) after time-centering",
            "step_down": "positive projection onto a high-to-low step template (event-centered) after time-centering",
            "combined_score": "max(pulse, step_up, step_down) per event, then averaged within (cluster, episode) and across episodes",
            "matrix_raw": "per-cluster mean of combined_score (== event_aligned ranking)",
            "matrix_window_mean": "per-cluster mean of window-mean activation",
            "matrix_task_mean": "per-cluster, broadcast the per-task mean activation over all cached timesteps",
        },
        "selection_counts": {
            **join.counts,
            "skipped_window": skipped_window,
            "shifted_window_count": shifted_window_count,
            "selected_events_before_activation_filter": len(usable_events),
            "selected_events_after_activation_filter": len(selected_event_payloads),
            "skipped_missing_window_vectors": skipped_missing_window_vectors,
            "task_timestep_counts": dict(task_counts),
            **load_counters,
        },
        "selected_events": selected_event_payloads,
        "row_keys": [
            {
                **join.cluster_metadata_by_id[cid],
                "num_episode_groups": row_episode_counts[cid],
                "num_events": row_event_counts[cid],
                "task_id": cluster_to_task_id.get(cid),
            }
            for cid in row_cluster_ids
        ],
        # Three paper-faithful matrices + a `matrix` alias for backward compat.
        "matrix_raw": matrix_raw,
        "matrix_window_mean": matrix_window_mean,
        "matrix_task_mean": matrix_task_mean,
        "matrix": matrix_raw,
        "row_results": row_results,
    }
    # A disconnected terminal or full ephemeral disk must not leave a partial
    # artifact that looks complete to the resumable orchestrator.
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(output_path)
    return {
        "output_path": str(output_path),
        "num_rows": num_rows,
        "dict_size": dict_size,
        "step_mapping": step_mapping,
        "selected_events": len(selected_event_payloads),
    }
