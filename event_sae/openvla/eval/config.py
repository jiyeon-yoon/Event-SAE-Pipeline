"""Configuration dataclasses and YAML loader for openVLA LIBERO eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    family: str = "openvla"
    checkpoint: str = "openvla/openvla-7b-finetuned-libero-10"
    revision: str = ""  # Empty follows Hub HEAD; set a commit hash to pin it.
    code_revision: str = ""  # Separate trust_remote_code commit (often the base OpenVLA repo).
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True


@dataclass
class EnvConfig:
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 7
    task_ids: str | list[int] = ""


@dataclass
class LoggingConfig:
    root_dir: str = "logs"
    save_video: bool = True
    save_actions: bool = True
    save_prompt_records: bool = False
    save_trajectory_records: bool = False


@dataclass
class SAECollectConfig:
    enabled: bool = True
    mode: str = "dense"        # "dense" (offline-friendly) or "topk" (online; requires sae_checkpoint).
    layer_idxs: str = "0"      # Comma-separated layer indices to hook; e.g., "31" or "0,16,24,31".
    flush_every: int = 50000   # Dense mode: buffered sample rows per layer before flushing a shard.
    sae_checkpoint: str = ""   # Topk mode: absolute path to a trained `ae.pt` (BatchTopKSAE).
    topk: int = 64             # Topk mode: number of top features kept per row.
    rows_per_shard: int = 20000  # Topk mode: rows per sparse shard.


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    sae_collect: SAECollectConfig = field(default_factory=SAECollectConfig)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Dict[str, Any] | None = None) -> RunConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if overrides:
        data = _deep_update(data, overrides)

    return RunConfig(
        model=ModelConfig(**data.get("model", {})),
        env=EnvConfig(**data.get("env", {})),
        logging=LoggingConfig(**data.get("logging", {})),
        sae_collect=SAECollectConfig(**data.get("sae_collect", {})),
    )


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    """Parse key=value overrides into nested dicts (dot notation)."""
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, raw = pair.split("=", 1)
        if raw.lower() in {"true", "false"}:
            value: Any = raw.lower() == "true"
        else:
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        target = overrides
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return overrides


def resolve_task_ids(task_ids: str | list[int] | tuple[int, ...] | int | None, num_tasks: int) -> list[int]:
    """Resolve an optional task subset spec into validated task ids."""
    if task_ids is None or task_ids == "" or task_ids == "all":
        return list(range(num_tasks))

    if isinstance(task_ids, int):
        selected = [task_ids]
    elif isinstance(task_ids, str):
        selected = [int(part.strip()) for part in task_ids.split(",") if part.strip()]
    else:
        selected = [int(task_id) for task_id in task_ids]

    if not selected:
        raise ValueError("env.task_ids resolved to an empty task list.")
    if len(set(selected)) != len(selected):
        raise ValueError(f"env.task_ids contains duplicates: {selected}")
    invalid = [task_id for task_id in selected if task_id < 0 or task_id >= num_tasks]
    if invalid:
        raise ValueError(f"env.task_ids contains ids outside [0, {num_tasks}): {invalid}")
    return selected
