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


@dataclass
class PairedReleaseConfig:
    """Objective trigger and validation rules for controlled release pairs."""

    stable_grasp_steps: int = 3
    min_lift_delta_m: float = 0.02
    trigger_delay_steps: int = 0
    max_force_open_steps: int = 20
    stable_detach_steps: int = 2
    # LIBERO's goal predicate can become true while the robot still holds the
    # object. Keep the normal policy running briefly so its natural release is
    # observed instead of ending on the first successful placement frame.
    normal_post_success_steps: int = 20
    post_detach_goal_stable_steps: int = 2
    forced_gripper_value: float = -1.0
    action_atol: float = 1e-6
    state_atol: float = 1e-6
    fail_on_invalid_pair: bool = False
    # env.num_trials_per_task is the maximum number of distinct initial-state
    # attempts. A task stops early only after both quotas are reached.
    target_valid_pairs_per_task: int = 20
    target_primary_pairs_per_task: int = 20
    min_free_disk_gb_at_start: float = 80.0
    abort_below_free_disk_gb: float = 20.0
    disk_check_every_steps: int = 25
    target_object_by_task: Dict[str, str] = field(default_factory=dict)
    destination_by_task: Dict[str, str] = field(default_factory=dict)


@dataclass
class PairedReleaseRunConfig:
    """Standalone config for normal/forced-release paired collection."""

    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    paired_release: PairedReleaseConfig = field(default_factory=PairedReleaseConfig)

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


def load_paired_release_config(
    path: str | Path,
    overrides: Dict[str, Any] | None = None,
) -> PairedReleaseRunConfig:
    """Load the paired collector without changing the normal-only config path."""

    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if overrides:
        data = _deep_update(data, overrides)
    paired_data = dict(data.get("paired_release", {}))
    for mapping_name in ("target_object_by_task", "destination_by_task"):
        mapping = paired_data.get(mapping_name, {}) or {}
        paired_data[mapping_name] = {
            str(key): str(value) for key, value in mapping.items()
        }
    cfg = PairedReleaseRunConfig(
        model=ModelConfig(**data.get("model", {})),
        env=EnvConfig(**data.get("env", {})),
        output=OutputConfig(**data.get("output", {})),
        collection=CollectionConfig(**data.get("collection", {})),
        paired_release=PairedReleaseConfig(**paired_data),
    )
    validate_config(cfg)
    validate_paired_release_config(cfg)
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


def validate_paired_release_config(cfg: PairedReleaseRunConfig) -> None:
    paired = cfg.paired_release
    if not cfg.model.revision or not cfg.model.code_revision:
        raise ValueError(
            "Paired collection requires pinned model.revision and "
            "model.code_revision"
        )
    if paired.stable_grasp_steps <= 0:
        raise ValueError("paired_release.stable_grasp_steps must be positive")
    if paired.min_lift_delta_m <= 0:
        raise ValueError("paired_release.min_lift_delta_m must be positive")
    if paired.trigger_delay_steps < 0:
        raise ValueError("paired_release.trigger_delay_steps cannot be negative")
    if paired.max_force_open_steps <= 0:
        raise ValueError("paired_release.max_force_open_steps must be positive")
    if paired.stable_detach_steps <= 0:
        raise ValueError("paired_release.stable_detach_steps must be positive")
    if paired.normal_post_success_steps <= 0:
        raise ValueError("paired_release.normal_post_success_steps must be positive")
    if paired.post_detach_goal_stable_steps <= 0:
        raise ValueError(
            "paired_release.post_detach_goal_stable_steps must be positive"
        )
    if paired.forced_gripper_value != -1.0:
        raise ValueError(
            "LIBERO paired release is fixed to forced_gripper_value=-1.0 (open)"
        )
    if paired.action_atol < 0 or paired.state_atol < 0:
        raise ValueError("paired release tolerances cannot be negative")
    if paired.target_valid_pairs_per_task <= 0:
        raise ValueError("paired_release.target_valid_pairs_per_task must be positive")
    if paired.target_primary_pairs_per_task < 0:
        raise ValueError(
            "paired_release.target_primary_pairs_per_task cannot be negative"
        )
    if paired.target_valid_pairs_per_task > cfg.env.num_trials_per_task:
        raise ValueError(
            "paired_release.target_valid_pairs_per_task cannot exceed "
            "env.num_trials_per_task (the maximum attempts per task)"
        )
    if paired.target_primary_pairs_per_task > cfg.env.num_trials_per_task:
        raise ValueError(
            "paired_release.target_primary_pairs_per_task cannot exceed "
            "env.num_trials_per_task (the maximum attempts per task)"
        )
    if paired.target_primary_pairs_per_task > paired.target_valid_pairs_per_task:
        raise ValueError(
            "paired_release.target_primary_pairs_per_task cannot exceed "
            "target_valid_pairs_per_task because every primary pair is valid"
        )
    if paired.min_free_disk_gb_at_start <= 0:
        raise ValueError("paired_release.min_free_disk_gb_at_start must be positive")
    if paired.abort_below_free_disk_gb <= 0:
        raise ValueError("paired_release.abort_below_free_disk_gb must be positive")
    if paired.min_free_disk_gb_at_start <= paired.abort_below_free_disk_gb:
        raise ValueError("initial free-disk guard must exceed the runtime guard")
    if paired.disk_check_every_steps <= 0:
        raise ValueError("paired_release.disk_check_every_steps must be positive")
    if any(not value for value in paired.target_object_by_task.values()):
        raise ValueError("paired_release.target_object_by_task values cannot be empty")
    if any(not value for value in paired.destination_by_task.values()):
        raise ValueError("paired_release.destination_by_task values cannot be empty")
    if not cfg.output.save_model_input_rgb:
        raise ValueError("Paired validation requires output.save_model_input_rgb=true")
    if not cfg.collection.fail_fast:
        raise ValueError(
            "Paired collection requires collection.fail_fast=true so a failed "
            "policy forward cannot leave misaligned activation rows"
        )
