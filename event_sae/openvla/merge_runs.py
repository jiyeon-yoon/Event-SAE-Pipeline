"""Lossless merge of task-sharded OpenVLA collection runs.

Parallel collection runs restart ``episode_num`` and activation shard names.
Downstream Event-SAE code joins on episode numbers, so merely copying those
directories together corrupts the event/activation relationship.  This module
assigns one deterministic global episode id and rewrites every dependent
artifact consistently while leaving dense tensor contents unchanged.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MergeConfig:
    input_dirs: tuple[str, ...]
    output_dir: str
    trials_per_task: int = 50
    expected_tasks: int | None = 10
    expected_shards: int | None = None
    link_mode: str = "symlink"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


def _materialize(source: Path, target: Path, mode: str) -> None:
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        target.symlink_to(source)
    elif mode == "hardlink":
        os.link(source, target)
    elif mode == "copy":
        shutil.copy2(source, target)
    else:
        raise ValueError(f"Unsupported link_mode={mode!r}")


def _global_episode(task_id: int, task_episode_idx: int, trials_per_task: int) -> int:
    if task_id < 0 or not (0 <= task_episode_idx < trials_per_task):
        raise ValueError(
            f"Invalid task/trial pair ({task_id}, {task_episode_idx}) for "
            f"trials_per_task={trials_per_task}"
        )
    return task_id * trials_per_task + task_episode_idx + 1


def _source_record(record: dict, *, run_idx: int, local_episode: int, global_episode: int) -> dict:
    rewritten = dict(record)
    rewritten["episode_num"] = global_episode
    rewritten["source_run_idx"] = run_idx
    rewritten["source_episode_num"] = local_episode
    return rewritten


def _load_prompts(
    input_dirs: list[Path], trials_per_task: int
) -> tuple[list[dict], dict[tuple[int, int], int], dict[int, dict]]:
    prompts: list[dict] = []
    local_to_global: dict[tuple[int, int], int] = {}
    global_to_prompt: dict[int, dict] = {}
    seen_task_trials: set[tuple[int, int]] = set()
    for run_idx, run_dir in enumerate(input_dirs):
        for record in _load_jsonl(run_dir / "prompt_records.jsonl"):
            local_episode = int(record["episode_num"])
            task_id = int(record["task_id"])
            trial = int(record["task_episode_idx"])
            task_trial = (task_id, trial)
            if task_trial in seen_task_trials:
                raise ValueError(f"Duplicate task/trial across runs: {task_trial}")
            seen_task_trials.add(task_trial)
            global_episode = _global_episode(task_id, trial, trials_per_task)
            key = (run_idx, local_episode)
            if key in local_to_global:
                raise ValueError(f"Duplicate local episode in run {run_idx}: {local_episode}")
            rewritten = _source_record(
                record,
                run_idx=run_idx,
                local_episode=local_episode,
                global_episode=global_episode,
            )
            local_to_global[key] = global_episode
            global_to_prompt[global_episode] = rewritten
            prompts.append(rewritten)
    prompts.sort(key=lambda item: int(item["episode_num"]))
    return prompts, local_to_global, global_to_prompt


def _merge_trajectories(
    input_dirs: list[Path],
    local_to_global: dict[tuple[int, int], int],
    global_to_prompt: dict[int, dict],
) -> tuple[list[dict], dict[int, int]]:
    records: list[dict] = []
    by_episode: dict[int, list[dict]] = defaultdict(list)
    for run_idx, run_dir in enumerate(input_dirs):
        for record in _load_jsonl(run_dir / "trajectory_records.jsonl"):
            local_episode = int(record["episode_num"])
            key = (run_idx, local_episode)
            if key not in local_to_global:
                raise ValueError(f"Trajectory references unknown episode {key}")
            global_episode = local_to_global[key]
            prompt = global_to_prompt[global_episode]
            if int(record["task_id"]) != int(prompt["task_id"]) or int(
                record["task_episode_idx"]
            ) != int(prompt["task_episode_idx"]):
                raise ValueError(f"Trajectory/prompt mismatch for episode {global_episode}")
            rewritten = _source_record(
                record,
                run_idx=run_idx,
                local_episode=local_episode,
                global_episode=global_episode,
            )
            by_episode[global_episode].append(rewritten)
    step_counts: dict[int, int] = {}
    for global_episode in sorted(by_episode):
        episode_records = sorted(
            by_episode[global_episode], key=lambda item: int(item["step_in_episode"])
        )
        steps = [int(item["step_in_episode"]) for item in episode_records]
        if steps != list(range(len(steps))):
            raise ValueError(f"Non-contiguous trajectory steps for episode {global_episode}")
        step_counts[global_episode] = len(steps)
        records.extend(episode_records)
    if set(step_counts) != set(global_to_prompt):
        raise ValueError("Prompt and trajectory episode sets differ")
    return records, step_counts


def _merge_actions(
    input_dirs: list[Path], global_to_prompt: dict[int, dict], step_counts: dict[int, int]
) -> tuple[dict, dict[str, dict]]:
    merged: dict[str, dict[str, list]] = {}
    for run_dir in input_dirs:
        path = run_dir / "actions.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for task_description, trials in payload.items():
            target = merged.setdefault(str(task_description), {})
            for trial, actions in trials.items():
                if str(trial) in target:
                    raise ValueError(
                        f"Duplicate action trace for task={task_description!r}, trial={trial}"
                    )
                target[str(trial)] = actions

    by_episode: dict[str, dict] = {}
    for global_episode, prompt in global_to_prompt.items():
        description = str(prompt["task_description"])
        trial = str(int(prompt["task_episode_idx"]))
        try:
            actions = merged[description][trial]
        except KeyError as exc:
            raise ValueError(f"Missing action trace for episode {global_episode}") from exc
        if len(actions) != step_counts[global_episode]:
            raise ValueError(
                f"Action/trajectory length mismatch for episode {global_episode}: "
                f"actions={len(actions)}, steps={step_counts[global_episode]}"
            )
        by_episode[str(global_episode)] = {
            "episode_num": global_episode,
            "task_id": int(prompt["task_id"]),
            "task_episode_idx": int(prompt["task_episode_idx"]),
            "task_description": description,
            "actions": actions,
        }
    return merged, by_episode


def _merge_videos(
    input_dirs: list[Path],
    local_to_global: dict[tuple[int, int], int],
    output_dir: Path,
    link_mode: str,
) -> int:
    count = 0
    for (run_idx, local_episode), global_episode in sorted(
        local_to_global.items(), key=lambda item: item[1]
    ):
        matches = sorted(
            (input_dirs[run_idx] / "videos").glob(f"*episode={local_episode}--*.mp4")
        )
        if len(matches) != 1:
            raise ValueError(
                f"Expected one video for run={run_idx}, episode={local_episode}; got {len(matches)}"
            )
        name = re.sub(
            rf"episode={local_episode}--",
            f"episode={global_episode}--",
            matches[0].name,
            count=1,
        )
        _materialize(matches[0], output_dir / "videos" / name, link_mode)
        count += 1
    return count


def _merge_events(input_dirs: list[Path], output_path: Path) -> int:
    rows_by_task: dict[str, dict] = {}
    fieldnames: list[str] | None = None
    for run_dir in input_dirs:
        path = run_dir / "events.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"events.csv columns differ in {path}")
            for row in reader:
                task = str(row["Task Description"])
                if task in rows_by_task:
                    raise ValueError(f"Duplicate events.csv task: {task}")
                rows_by_task[task] = row
    if not fieldnames:
        raise ValueError("events.csv has no columns")
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_by_task[key] for key in sorted(rows_by_task))
    return len(rows_by_task)


def _merge_activation_index(
    input_dirs: list[Path],
    local_to_global: dict[tuple[int, int], int],
    global_to_prompt: dict[int, dict],
    step_counts: dict[int, int],
    output_dir: Path,
    link_mode: str,
) -> tuple[list[dict], int]:
    merged_records: list[dict] = []
    shard_count = 0
    global_forward = 0
    for run_idx, run_dir in enumerate(input_dirs):
        source_dense_dir = run_dir / "sae_activations" / "post_mlp_residual"
        index_path = source_dense_dir / "activation_index.jsonl"
        records = _load_jsonl(index_path)
        local_forward_map: dict[int, int] = {}
        by_shard: dict[str, list[dict]] = defaultdict(list)
        for record in records:
            by_shard[str(record["shard_path"])].append(record)

        for shard_relpath in sorted(by_shard):
            source_shard = (source_dense_dir / shard_relpath).resolve()
            target_relpath = Path("sources") / f"run_{run_idx:02d}" / source_shard.name
            target_shard = output_dir / "sae_activations" / "post_mlp_residual" / target_relpath
            _materialize(source_shard, target_shard, link_mode)
            shard_records = sorted(
                by_shard[shard_relpath], key=lambda item: int(item["row_start"])
            )
            expected_row = 0
            for record in shard_records:
                r0, r1 = int(record["row_start"]), int(record["row_end"])
                if r0 != expected_row or r1 <= r0:
                    raise ValueError(
                        f"Invalid activation ranges in run {run_idx}, shard {shard_relpath}"
                    )
                expected_row = r1
                local_episode = int(record["episode_num"])
                key = (run_idx, local_episode)
                if key not in local_to_global:
                    raise ValueError(f"Activation index references unknown episode {key}")
                global_episode = local_to_global[key]
                prompt = global_to_prompt[global_episode]
                if int(record["task_id"]) != int(prompt["task_id"]) or int(
                    record["task_episode_idx"]
                ) != int(prompt["task_episode_idx"]):
                    raise ValueError(
                        f"Activation/prompt task mismatch for episode {global_episode}"
                    )
                step = int(record["step_in_episode"])
                if not (0 <= step < step_counts[global_episode]):
                    raise ValueError(
                        f"Activation step {step} is outside trajectory for episode "
                        f"{global_episode} (steps={step_counts[global_episode]})"
                    )
                local_forward = int(record.get("global_forward_idx") or 0)
                if local_forward <= 0:
                    raise ValueError(
                        f"Invalid global_forward_idx in run {run_idx}: {local_forward}"
                    )
                if local_forward not in local_forward_map:
                    global_forward += 1
                    local_forward_map[local_forward] = global_forward
                rewritten = _source_record(
                    record,
                    run_idx=run_idx,
                    local_episode=local_episode,
                    global_episode=global_episode,
                )
                rewritten["global_forward_idx"] = local_forward_map[local_forward]
                rewritten["source_shard_path"] = str(record["shard_path"])
                rewritten["shard_path"] = target_relpath.as_posix()
                merged_records.append(rewritten)
            shard_count += 1
    merged_records.sort(
        key=lambda item: (
            int(item["global_forward_idx"]),
            str(item["shard_path"]),
            int(item["row_start"]),
        )
    )
    return merged_records, shard_count


def merge_openvla_runs(cfg: MergeConfig) -> dict:
    """Merge task-sharded runs into one validated downstream-ready run."""
    if cfg.trials_per_task <= 0:
        raise ValueError("trials_per_task must be positive")
    if cfg.link_mode not in {"symlink", "hardlink", "copy"}:
        raise ValueError("link_mode must be symlink, hardlink, or copy")
    input_dirs = [Path(value).expanduser().resolve() for value in cfg.input_dirs]
    if not input_dirs:
        raise ValueError("At least one input directory is required")
    if len(set(input_dirs)) != len(input_dirs):
        raise ValueError("Duplicate input directories")
    for path in input_dirs:
        if not path.is_dir():
            raise FileNotFoundError(path)
    # Stable source namespaces and global-forward ordering regardless of CLI
    # argument order. Episode IDs themselves are task/trial-derived.
    def source_sort_key(path: Path):
        task_ids = sorted(
            {int(row["task_id"]) for row in _load_jsonl(path / "prompt_records.jsonl")}
        )
        return (task_ids, str(path))

    input_dirs.sort(key=source_sort_key)

    output_dir = Path(cfg.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()
    try:
        prompts, local_to_global, global_to_prompt = _load_prompts(
            input_dirs, cfg.trials_per_task
        )
        if cfg.expected_tasks is not None:
            expected_episode_count = cfg.expected_tasks * cfg.trials_per_task
            expected_ids = set(range(1, expected_episode_count + 1))
            if set(global_to_prompt) != expected_ids:
                missing = sorted(expected_ids.difference(global_to_prompt))
                extra = sorted(set(global_to_prompt).difference(expected_ids))
                raise ValueError(
                    f"Expected {expected_episode_count} episodes; missing={missing[:10]}, "
                    f"extra={extra[:10]}"
                )

        trajectories, step_counts = _merge_trajectories(
            input_dirs, local_to_global, global_to_prompt
        )
        actions, actions_by_episode = _merge_actions(
            input_dirs, global_to_prompt, step_counts
        )
        _write_jsonl(temporary / "prompt_records.jsonl", prompts)
        _write_jsonl(temporary / "trajectory_records.jsonl", trajectories)
        (temporary / "actions.json").write_text(
            json.dumps(actions, indent=2) + "\n", encoding="utf-8"
        )
        (temporary / "actions_by_episode.json").write_text(
            json.dumps(actions_by_episode, indent=2) + "\n", encoding="utf-8"
        )
        video_count = _merge_videos(
            input_dirs, local_to_global, temporary, cfg.link_mode
        )
        event_task_count = _merge_events(input_dirs, temporary / "events.csv")
        activation_records, shard_count = _merge_activation_index(
            input_dirs,
            local_to_global,
            global_to_prompt,
            step_counts,
            temporary,
            cfg.link_mode,
        )
        dense_dir = temporary / "sae_activations" / "post_mlp_residual"
        _write_jsonl(dense_dir / "activation_index.jsonl", activation_records)

        if cfg.expected_shards is not None and shard_count != cfg.expected_shards:
            raise ValueError(
                f"Expected {cfg.expected_shards} shards, found {shard_count}"
            )
        if event_task_count != len({int(p["task_id"]) for p in prompts}):
            raise ValueError("events.csv task count does not match prompt tasks")
        if video_count != len(prompts):
            raise ValueError("Video count does not match prompt count")

        task_counts: dict[str, int] = defaultdict(int)
        for prompt in prompts:
            task_counts[str(int(prompt["task_id"]))] += 1
        manifest = {
            "schema_version": "event_sae_openvla_merged_run_v1",
            "input_dirs": [str(path) for path in input_dirs],
            "link_mode": cfg.link_mode,
            "trials_per_task": cfg.trials_per_task,
            "num_episodes": len(prompts),
            "task_episode_counts": dict(sorted(task_counts.items(), key=lambda item: int(item[0]))),
            "num_trajectory_rows": len(trajectories),
            "num_videos": video_count,
            "num_activation_shards": shard_count,
            "num_activation_index_records": len(activation_records),
            "global_episode_rule": "task_id * trials_per_task + task_episode_idx + 1",
        }
        (temporary / "merge_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(output_dir)
        return {**manifest, "output_dir": str(output_dir)}
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
