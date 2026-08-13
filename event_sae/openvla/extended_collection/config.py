"""Configuration for the standalone extended LIBERO collector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class ModelConfig:
    family: str = "openvla"
    checkpoint: str = "openvla/openvla-7b-finetuned-libero-spatial"
    revision: str = ""
    code_revision: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True


@dataclass
class EnvConfig:
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    seed: int = 0
    task_ids: str | list[int] = ""
    resolution: int = 256


@dataclass
class OutputConfig:
    root_dir: str = "/workspace/results/extended-libero"
    save_video: bool = True
    video_fps: int = 30
    save_model_input_rgb: bool = True
    compress_npz: bool = True
    flush_jsonl_every_step: bool = True


@dataclass
class CollectionConfig:
    # Deliberately one layer: the selected research stream, not a layer sweep.
    layer_idx: int = 31
    activation_flush_every: int = 50000
    save_observation_state: bool = True
    save_simulator_state: bool = True
    save_object_and_fixture_state: bool = True
    save_contacts_and_grasp: bool = True
    save_goal_predicates: bool = True
    save_policy_uncertainty: bool = True
    strict_preflight: bool = True
    fail_fast: bool = True


@dataclass
class ExtendedRunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_overrides(pairs: list[str]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def load_extended_config(
    path: str | Path,
    overrides: Dict[str, Any] | None = None,
) -> ExtendedRunConfig:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if overrides:
        data = _deep_update(data, overrides)
    cfg = ExtendedRunConfig(
        model=ModelConfig(**data.get("model", {})),
        env=EnvConfig(**data.get("env", {})),
        output=OutputConfig(**data.get("output", {})),
        collection=CollectionConfig(**data.get("collection", {})),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ExtendedRunConfig) -> None:
    if cfg.model.family != "openvla":
        raise ValueError(
            "The extended collector currently supports model.family='openvla' only"
        )
    if cfg.collection.layer_idx != 31:
        raise ValueError(
            "This collector is intentionally fixed to layer 31. Use the original "
            "layer-sweep collector for other layers."
        )
    if cfg.env.num_trials_per_task <= 0:
        raise ValueError("env.num_trials_per_task must be positive")
    if cfg.env.num_steps_wait < 0:
        raise ValueError("env.num_steps_wait cannot be negative")
    if cfg.env.resolution <= 0:
        raise ValueError("env.resolution must be positive")
    if cfg.output.video_fps <= 0:
        raise ValueError("output.video_fps must be positive")
    if cfg.collection.activation_flush_every <= 0:
        raise ValueError("collection.activation_flush_every must be positive")
    required_streams = {
        "save_observation_state": cfg.collection.save_observation_state,
        "save_simulator_state": cfg.collection.save_simulator_state,
        "save_object_and_fixture_state": cfg.collection.save_object_and_fixture_state,
        "save_contacts_and_grasp": cfg.collection.save_contacts_and_grasp,
        "save_goal_predicates": cfg.collection.save_goal_predicates,
        "save_policy_uncertainty": cfg.collection.save_policy_uncertainty,
    }
    disabled = [name for name, enabled in required_streams.items() if not enabled]
    if disabled:
        raise ValueError(
            "The extended-v1 schema requires every low-volume telemetry stream; "
            "disabled fields: " + ", ".join(disabled)
        )
