"""Paired normal/forced-release OpenVLA + LIBERO data collector.

This module is independent of Event-SAE's original collector and of the
normal-only extended collector.  For each candidate it runs the normal policy,
finds an objective stable-grasp-plus-lift trigger, then replays the exact same
initial state and seed while forcing only the gripper command open.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import time
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from event_sae.openvla.extended_collection.activation import (
    Layer31ActivationCollector,
)
from event_sae.openvla.extended_collection.config import PairedReleaseRunConfig
from event_sae.openvla.extended_collection.controlled_release import (
    FORCED_RELEASE_CONDITION,
    NORMAL_CONDITION,
    ReleaseTriggerDetector,
    force_gripper_open,
    resolve_goal_destination,
    resolve_target_object,
    target_destination_contact,
    target_grasped,
    target_z,
)
from event_sae.openvla.extended_collection.policy import (
    _openvla_action_vocab_size,
    infer_action_with_uncertainty,
)
from event_sae.openvla.extended_collection.runner import (
    _MAX_STEPS_PER_SUITE,
    _git_state,
    _make_env,
    _new_run_dir,
    _resolve_task_ids,
    _resolve_unnorm_key,
    _safe_info,
    _save_video,
)
from event_sae.openvla.extended_collection.runtime import (
    dummy_action,
    get_source_rgb,
    load_openvla,
    load_processor,
    set_seed,
    to_executed_libero_action,
)
from event_sae.openvla.extended_collection.telemetry import (
    build_simulator_schema,
    capture_snapshot,
    validate_preflight,
)
from event_sae.openvla.extended_collection.writer import (
    ExtendedRunWriter,
    array_sha256,
    jsonable,
)


STATEFUL_SIM_FIELDS = (
    "time",
    "qpos",
    "qvel",
    "act",
    "history",
    "qacc_warmstart",
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "eq_active",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "plugin_state",
)

MUJOCO_MODEL_DIMENSIONS = (
    "nq",
    "nv",
    "na",
    "nu",
    "nmocap",
    "nuserdata",
    "neq",
    "npluginstate",
)

CONTROLLER_REPLAY_FIELDS = (
    "initial_joint",
    "initial_ee_pos",
    "initial_ee_ori_mat",
    "goal_pos",
    "goal_ori",
    "relative_ori",
    "ori_ref",
    "kp",
    "kd",
    "torques",
    "action_scale",
    "action_input_transform",
    "action_output_transform",
    "new_update",
    "interpolator_pos",
    "interpolator_ori",
)

ROBOT_REPLAY_FIELDS = (
    "torques",
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_vel_buffer",
    "recent_ee_acc",
)


@dataclass(frozen=True)
class ReplayObjectCheckpoint:
    """Serializable mutable fields of a nested robosuite helper object."""

    object_type: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class RobotControlCheckpoint:
    """Python-side robosuite state that is not stored in ``MjData``."""

    robot_type: str
    controller_type: str
    controller_fields: dict[str, Any]
    robot_fields: dict[str, Any]
    gripper_type: str | None
    gripper_fields: dict[str, Any]


@dataclass(frozen=True)
class ObservableCheckpoint:
    """Mutable state of one robosuite ``Observable``."""

    observable_type: str
    fields: dict[str, Any]


@dataclass(frozen=True)
class RobosuitePythonCheckpoint:
    """Python-only control and observation state at the paired branch point."""

    robots: tuple[RobotControlCheckpoint, ...]
    observables: dict[str, ObservableCheckpoint]
    obs_cache: dict[str, Any]
    rng_state: dict[str, Any] | None
    python_random_state: tuple[Any, ...]
    numpy_random_state: tuple[Any, ...]
    torch_cpu_rng_state: np.ndarray
    torch_cuda_rng_states: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class MujocoIntegrationCheckpoint:
    """Exact MuJoCo forward-dynamics input at the paired branch point."""

    state: np.ndarray
    state_sha256: str
    state_size: int
    model_dimensions: tuple[int, ...]
    model_xml_sha256: str | None
    timestep: int | None
    cur_time: float | None
    done: bool | None
    robosuite_python_state: RobosuitePythonCheckpoint | None = None


@dataclass
class PrefixTrace:
    model_input_rgb: np.ndarray
    model_input_rgb_sha256: str
    observed_source_rgb: np.ndarray
    observed_source_rgb_sha256: str
    stateful_sim_sha256: str
    stateful_sim: dict[str, np.ndarray]
    raw_action: np.ndarray
    policy_action: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    integration_state: np.ndarray
    integration_state_sha256: str


@dataclass
class PrefixComparison:
    steps_compared: int = 0
    rgb_exact: bool = True
    observed_rgb_exact: bool = True
    stateful_sim_exact: bool = True
    integration_state_exact: bool = True
    max_observed_rgb_abs_delta: float = 0.0
    max_raw_action_abs_delta: float = 0.0
    max_policy_action_abs_delta: float = 0.0
    max_qpos_abs_delta: float = 0.0
    max_qvel_abs_delta: float = 0.0
    max_stateful_sim_abs_delta: float = 0.0
    max_integration_state_abs_delta: float = 0.0
    first_mismatch_step: int | None = None
    mismatch_field: str | None = None

    def compare(
        self,
        step: int,
        normal: PrefixTrace,
        forced: PrefixTrace,
        *,
        action_atol: float,
        state_atol: float,
    ) -> bool:
        self.steps_compared += 1
        raw_delta = _max_abs_delta(normal.raw_action, forced.raw_action)
        policy_delta = _max_abs_delta(normal.policy_action, forced.policy_action)
        qpos_delta = _max_abs_delta(normal.qpos, forced.qpos)
        qvel_delta = _max_abs_delta(normal.qvel, forced.qvel)
        observed_rgb_delta = _max_abs_delta(
            normal.observed_source_rgb, forced.observed_source_rgb
        )
        integration_state_delta = _max_abs_delta(
            normal.integration_state, forced.integration_state
        )
        if set(normal.stateful_sim) != set(forced.stateful_sim):
            state_field, state_delta = "field_set", float("inf")
        else:
            state_deltas = {
                name: _max_abs_delta(
                    normal.stateful_sim[name], forced.stateful_sim[name]
                )
                for name in normal.stateful_sim
            }
            state_field, state_delta = max(
                state_deltas.items(), key=lambda item: item[1]
            )
        self.max_raw_action_abs_delta = max(self.max_raw_action_abs_delta, raw_delta)
        self.max_policy_action_abs_delta = max(
            self.max_policy_action_abs_delta, policy_delta
        )
        self.max_qpos_abs_delta = max(self.max_qpos_abs_delta, qpos_delta)
        self.max_qvel_abs_delta = max(self.max_qvel_abs_delta, qvel_delta)
        self.max_observed_rgb_abs_delta = max(
            self.max_observed_rgb_abs_delta, observed_rgb_delta
        )
        self.max_stateful_sim_abs_delta = max(
            self.max_stateful_sim_abs_delta, state_delta
        )
        self.max_integration_state_abs_delta = max(
            self.max_integration_state_abs_delta, integration_state_delta
        )
        rgb_equal = normal.model_input_rgb_sha256 == forced.model_input_rgb_sha256
        observed_rgb_equal = (
            normal.observed_source_rgb_sha256 == forced.observed_source_rgb_sha256
        )
        sim_equal = normal.stateful_sim_sha256 == forced.stateful_sim_sha256
        integration_state_equal = (
            normal.integration_state_sha256 == forced.integration_state_sha256
        )
        self.rgb_exact = self.rgb_exact and rgb_equal
        self.observed_rgb_exact = self.observed_rgb_exact and observed_rgb_equal
        self.stateful_sim_exact = self.stateful_sim_exact and sim_equal
        self.integration_state_exact = (
            self.integration_state_exact and integration_state_equal
        )

        mismatch = None
        if not rgb_equal:
            mismatch = "model_input_rgb"
        elif raw_delta > action_atol:
            mismatch = "raw_openvla_action"
        elif policy_delta > action_atol:
            mismatch = "policy_libero_action"
        elif qpos_delta > state_atol:
            mismatch = "pre_qpos"
        elif qvel_delta > state_atol:
            mismatch = "pre_qvel"
        elif state_delta > state_atol:
            mismatch = f"pre_{state_field}"
        elif not integration_state_equal:
            mismatch = "pre_mujoco_integration_state_not_exact"
        if mismatch is not None and self.first_mismatch_step is None:
            self.first_mismatch_step = int(step)
            self.mismatch_field = mismatch
        return mismatch is None

    def as_dict(self) -> dict[str, Any]:
        return jsonable(self.__dict__)


@dataclass
class TaskPairQuota:
    """Track one task's adaptive pair-collection stopping criteria."""

    task_id: int
    max_attempts: int
    target_valid_pairs: int
    target_primary_pairs: int
    attempts: int = 0
    eligible_pairs: int = 0
    valid_pairs: int = 0
    invalid_pairs: int = 0
    ineligible_pairs: int = 0
    primary_analysis_pairs: int = 0

    @property
    def target_reached(self) -> bool:
        return (
            self.valid_pairs >= self.target_valid_pairs
            and self.primary_analysis_pairs >= self.target_primary_pairs
        )

    @property
    def can_attempt(self) -> bool:
        return not self.target_reached and self.attempts < self.max_attempts

    def begin_attempt(self, trial_idx: int) -> None:
        if not self.can_attempt:
            raise RuntimeError(f"Task {self.task_id} cannot start another attempt")
        if int(trial_idx) != self.attempts:
            raise ValueError(
                f"Task {self.task_id} expected trial {self.attempts}, "
                f"got {trial_idx}"
            )
        self.attempts += 1

    def record_ineligible(self) -> None:
        self.ineligible_pairs += 1

    def record_eligible(self, *, valid: bool, primary: bool) -> None:
        if primary and not valid:
            raise ValueError("A primary-analysis pair must also be valid")
        self.eligible_pairs += 1
        if valid:
            self.valid_pairs += 1
        else:
            self.invalid_pairs += 1
        self.primary_analysis_pairs += int(primary)

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "max_attempts": self.max_attempts,
            "target_valid_pairs": self.target_valid_pairs,
            "target_primary_pairs": self.target_primary_pairs,
            "attempts": self.attempts,
            "attempts_remaining": self.max_attempts - self.attempts,
            "eligible_pairs": self.eligible_pairs,
            "valid_pairs": self.valid_pairs,
            "invalid_pairs": self.invalid_pairs,
            "ineligible_pairs": self.ineligible_pairs,
            "primary_analysis_pairs": self.primary_analysis_pairs,
            "target_reached": self.target_reached,
        }


def _require_single_task_run(task_ids: list[int]) -> int:
    """Keep each expensive collection independently recoverable."""

    if len(task_ids) != 1:
        raise ValueError(
            "Paired collection requires exactly one task per run. Run tasks "
            "sequentially into separate output directories so a later task "
            "shortfall cannot invalidate data already collected for another task."
        )
    return int(task_ids[0])


def _max_abs_delta(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        return float("inf")
    if left_array.size == 0:
        return 0.0
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        return float("inf")
    return float(np.max(np.abs(left_array - right_array)))


def _sim_state_sha256(vectors: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    included = 0
    for name in STATEFUL_SIM_FIELDS:
        if name not in vectors:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(array_sha256(vectors[name]).encode("ascii"))
        included += 1
    if not included:
        raise ValueError("No stateful MuJoCo vectors were captured")
    return digest.hexdigest()


def _require_free_disk(path: Path, minimum_gb: float, *, phase: str) -> float:
    """Fail before quota exhaustion while leaving the current run incomplete."""

    free_gb = shutil.disk_usage(path).free / (1024**3)
    if free_gb < float(minimum_gb):
        raise RuntimeError(
            f"Insufficient free disk during {phase}: {free_gb:.1f} GiB remain; "
            f"required at least {float(minimum_gb):.1f} GiB"
        )
    return free_gb


def _runtime_provenance() -> dict[str, Any]:
    import torch

    packages = {}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "libero",
        "robosuite",
        "mujoco",
        "numpy",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "packages": packages,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _pair_seed(base_seed: int, task_id: int, trial_idx: int) -> int:
    return int(base_seed) + int(task_id) * 10_000 + int(trial_idx)


def _pair_id(suite: str, task_id: int, trial_idx: int, pair_seed: int) -> str:
    return f"{suite}-task{task_id:02d}-trial{trial_idx:03d}-seed{pair_seed}"


def _pair_invalid_reasons(
    normal: dict[str, Any],
    forced: dict[str, Any],
    forced_fatal: Exception | None,
) -> tuple[list[str], dict[str, bool]]:
    """Keep the primary rollout failure while recording exact-hash audit evidence."""

    audit = {
        "initial_state_exact": (
            normal["initial_state_sha256"] == forced["initial_state_sha256"]
        ),
        "warm_start_sim_state_exact": (
            normal["warm_start_sim_state_sha256"]
            == forced["warm_start_sim_state_sha256"]
        ),
        "warm_start_source_rgb_exact": (
            normal["warm_start_source_rgb_sha256"]
            == forced["warm_start_source_rgb_sha256"]
        ),
        "warm_start_integration_state_exact": (
            normal["warm_start_integration_state_sha256"]
            == forced["warm_start_integration_state_sha256"]
        ),
        "warm_start_model_xml_exact": (
            normal["warm_start_model_xml_sha256"]
            == forced["warm_start_model_xml_sha256"]
        ),
        "warm_start_robosuite_python_state_exact": (
            normal["warm_start_robosuite_python_state_sha256"]
            == forced["warm_start_robosuite_python_state_sha256"]
        ),
    }
    reasons = []
    if forced["invalid_reason"] is not None:
        reasons.append(str(forced["invalid_reason"]))
    if not audit["initial_state_exact"]:
        reasons.append("initial_state_hash_mismatch")
    if not audit["warm_start_integration_state_exact"]:
        reasons.append("warm_start_integration_state_hash_mismatch")
    if not audit["warm_start_model_xml_exact"]:
        reasons.append("warm_start_model_xml_hash_mismatch")
    if not audit["warm_start_robosuite_python_state_exact"]:
        reasons.append("warm_start_python_state_hash_mismatch")
    if forced_fatal is not None:
        reasons.append("forced_exception")
    return reasons, audit


def _raw_env(env):
    return getattr(env, "env", env)


def _qualified_type(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _snapshot_replay_value(value: Any) -> Any:
    """Copy state without copying controllers, robots, sensors, or simulator refs."""

    if value is None or isinstance(value, (str, bytes, bool, int, float)):
        return copy.deepcopy(value)
    if isinstance(value, np.generic):
        return value.copy()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, tuple):
        return tuple(_snapshot_replay_value(item) for item in value)
    if isinstance(value, list):
        return [_snapshot_replay_value(item) for item in value]
    if isinstance(value, dict):
        return {
            copy.deepcopy(key): _snapshot_replay_value(item)
            for key, item in value.items()
        }
    fields = getattr(value, "__dict__", None)
    if fields is None:
        raise TypeError(f"Unsupported replay-state value: {_qualified_type(value)}")
    return ReplayObjectCheckpoint(
        object_type=_qualified_type(value),
        fields={
            name: _snapshot_replay_value(item)
            for name, item in fields.items()
            if not callable(item)
        },
    )


def _restore_replay_value(current: Any, checkpoint: Any) -> Any:
    if isinstance(checkpoint, ReplayObjectCheckpoint):
        if current is None or _qualified_type(current) != checkpoint.object_type:
            current_type = None if current is None else _qualified_type(current)
            raise RuntimeError(
                "Replay helper type changed: "
                f"expected {checkpoint.object_type}, got {current_type}"
            )
        for name, value in checkpoint.fields.items():
            if not hasattr(current, name):
                raise RuntimeError(
                    f"Replay helper {_qualified_type(current)} lost field {name!r}"
                )
            restored = _restore_replay_value(getattr(current, name), value)
            setattr(current, name, restored)
        return current
    if isinstance(checkpoint, np.ndarray):
        return checkpoint.copy()
    if isinstance(checkpoint, np.generic):
        return checkpoint.copy()
    if isinstance(checkpoint, tuple):
        current_items = (
            current if isinstance(current, tuple) else (None,) * len(checkpoint)
        )
        if len(current_items) != len(checkpoint):
            raise RuntimeError("Replay tuple length changed")
        return tuple(
            _restore_replay_value(item, saved)
            for item, saved in zip(current_items, checkpoint)
        )
    if isinstance(checkpoint, list):
        current_items = (
            current if isinstance(current, list) else [None] * len(checkpoint)
        )
        if len(current_items) != len(checkpoint):
            raise RuntimeError("Replay list length changed")
        return [
            _restore_replay_value(item, saved)
            for item, saved in zip(current_items, checkpoint)
        ]
    if isinstance(checkpoint, dict):
        current_mapping = current if isinstance(current, dict) else {}
        return {
            copy.deepcopy(key): _restore_replay_value(current_mapping.get(key), value)
            for key, value in checkpoint.items()
        }
    return copy.deepcopy(checkpoint)


def _replay_values_equal(left: Any, right: Any) -> bool:
    if is_dataclass(left) or is_dataclass(right):
        return (
            type(left) is type(right)
            and is_dataclass(left)
            and is_dataclass(right)
            and all(
                _replay_values_equal(
                    getattr(left, field.name), getattr(right, field.name)
                )
                for field in dataclass_fields(left)
            )
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(
                np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True)
            )
        except TypeError:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if isinstance(left, np.generic) or isinstance(right, np.generic):
        return _replay_values_equal(np.asarray(left), np.asarray(right))
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_replay_values_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        return (
            isinstance(left, type(right))
            and len(left) == len(right)
            and all(_replay_values_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return True
    return bool(left == right)


def _update_replay_digest(digest, value: Any) -> None:
    if is_dataclass(value):
        digest.update(b"dataclass:")
        digest.update(_qualified_type(value).encode("utf-8"))
        for field in dataclass_fields(value):
            digest.update(field.name.encode("utf-8"))
            _update_replay_digest(digest, getattr(value, field.name))
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"array:")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(array.shape).encode("ascii"))
        digest.update(array.tobytes())
        return
    if isinstance(value, np.generic):
        _update_replay_digest(digest, np.asarray(value))
        return
    if isinstance(value, dict):
        digest.update(b"dict:")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_replay_digest(digest, key)
            _update_replay_digest(digest, value[key])
        return
    if isinstance(value, (tuple, list)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _update_replay_digest(digest, item)
        return
    digest.update(type(value).__name__.encode("ascii"))
    digest.update(repr(value).encode("utf-8"))


def _robosuite_python_state_sha256(
    checkpoint: RobosuitePythonCheckpoint,
) -> str:
    digest = hashlib.sha256()
    _update_replay_digest(digest, checkpoint)
    return digest.hexdigest()


def _capture_named_fields(value: Any, names: tuple[str, ...]) -> dict[str, Any]:
    return {
        name: _snapshot_replay_value(getattr(value, name))
        for name in names
        if hasattr(value, name)
    }


def _restore_named_fields(value: Any, fields: dict[str, Any]) -> None:
    for name, checkpoint in fields.items():
        if not hasattr(value, name):
            raise RuntimeError(
                f"Replay target {_qualified_type(value)} lost field {name!r}"
            )
        restored = _restore_replay_value(getattr(value, name), checkpoint)
        setattr(value, name, restored)


def _observable_fields(observable: Any) -> dict[str, Any]:
    excluded = {"_sensor", "_corrupter", "_filter", "_delayer"}
    return {
        name: _snapshot_replay_value(value)
        for name, value in vars(observable).items()
        if name not in excluded and not callable(value)
    }


def _capture_robosuite_python_checkpoint(env) -> RobosuitePythonCheckpoint:
    import torch

    raw = _raw_env(env)
    robots = []
    for robot in getattr(raw, "robots", ()):
        controller = getattr(robot, "controller", None)
        if controller is None:
            raise RuntimeError("Robosuite robot has no controller")
        gripper = getattr(robot, "gripper", None)
        gripper_fields = (
            _capture_named_fields(gripper, ("current_action",))
            if gripper is not None
            else {}
        )
        robots.append(
            RobotControlCheckpoint(
                robot_type=_qualified_type(robot),
                controller_type=_qualified_type(controller),
                controller_fields=_capture_named_fields(
                    controller, CONTROLLER_REPLAY_FIELDS
                ),
                robot_fields=_capture_named_fields(robot, ROBOT_REPLAY_FIELDS),
                gripper_type=(None if gripper is None else _qualified_type(gripper)),
                gripper_fields=gripper_fields,
            )
        )

    observables = getattr(raw, "_observables", None)
    obs_cache = getattr(raw, "_obs_cache", None)
    if not isinstance(observables, dict) or not isinstance(obs_cache, dict):
        raise RuntimeError(
            "Exact paired replay requires robosuite observables and _obs_cache"
        )
    observable_checkpoints = {
        name: ObservableCheckpoint(
            observable_type=_qualified_type(observable),
            fields=_observable_fields(observable),
        )
        for name, observable in observables.items()
    }
    rng = getattr(raw, "rng", None)
    rng_state = None
    if rng is not None and hasattr(rng, "bit_generator"):
        rng_state = copy.deepcopy(rng.bit_generator.state)
    return RobosuitePythonCheckpoint(
        robots=tuple(robots),
        observables=observable_checkpoints,
        obs_cache=_snapshot_replay_value(obs_cache),
        rng_state=rng_state,
        python_random_state=copy.deepcopy(random.getstate()),
        numpy_random_state=_snapshot_replay_value(np.random.get_state()),
        torch_cpu_rng_state=torch.get_rng_state().cpu().numpy().copy(),
        torch_cuda_rng_states=tuple(
            state.cpu().numpy().copy()
            for state in (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else ()
            )
        ),
    )


def _restore_robosuite_python_checkpoint(env, checkpoint: RobosuitePythonCheckpoint):
    import torch

    raw = _raw_env(env)
    robots = tuple(getattr(raw, "robots", ()))
    if len(robots) != len(checkpoint.robots):
        raise RuntimeError(
            "Robosuite robot count changed between paired branches: "
            f"expected {len(checkpoint.robots)}, got {len(robots)}"
        )
    for index, (robot, saved) in enumerate(zip(robots, checkpoint.robots)):
        if _qualified_type(robot) != saved.robot_type:
            raise RuntimeError(f"Robosuite robot {index} type changed")
        controller = getattr(robot, "controller", None)
        if controller is None or _qualified_type(controller) != saved.controller_type:
            raise RuntimeError(f"Robosuite robot {index} controller type changed")
        update = getattr(controller, "update", None)
        if callable(update):
            update(force=True)
        _restore_named_fields(controller, saved.controller_fields)
        _restore_named_fields(robot, saved.robot_fields)
        gripper = getattr(robot, "gripper", None)
        current_gripper_type = None if gripper is None else _qualified_type(gripper)
        if current_gripper_type != saved.gripper_type:
            raise RuntimeError(f"Robosuite robot {index} gripper type changed")
        if gripper is not None:
            _restore_named_fields(gripper, saved.gripper_fields)

    observables = getattr(raw, "_observables", None)
    if not isinstance(observables, dict) or set(observables) != set(
        checkpoint.observables
    ):
        raise RuntimeError("Robosuite observable topology changed between branches")
    for name, saved in checkpoint.observables.items():
        observable = observables[name]
        if _qualified_type(observable) != saved.observable_type:
            raise RuntimeError(f"Robosuite observable {name!r} type changed")
        _restore_named_fields(observable, saved.fields)
    raw._obs_cache = _restore_replay_value(
        getattr(raw, "_obs_cache", {}), checkpoint.obs_cache
    )
    if checkpoint.rng_state is not None:
        rng = getattr(raw, "rng", None)
        if rng is None or not hasattr(rng, "bit_generator"):
            raise RuntimeError("Robosuite RNG topology changed between branches")
        rng.bit_generator.state = copy.deepcopy(checkpoint.rng_state)

    random.setstate(copy.deepcopy(checkpoint.python_random_state))
    np.random.set_state(
        _restore_replay_value(np.random.get_state(), checkpoint.numpy_random_state)
    )
    torch.set_rng_state(
        torch.from_numpy(checkpoint.torch_cpu_rng_state.copy()).to(dtype=torch.uint8)
    )
    if checkpoint.torch_cuda_rng_states:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state exists but CUDA is unavailable")
        if len(checkpoint.torch_cuda_rng_states) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count changed between paired branches")
        torch.cuda.set_rng_state_all(
            [
                torch.from_numpy(state.copy()).to(dtype=torch.uint8)
                for state in checkpoint.torch_cuda_rng_states
            ]
        )

    restored = _capture_robosuite_python_checkpoint(env)
    if not _replay_values_equal(checkpoint, restored):
        raise RuntimeError("Robosuite Python state did not restore exactly")
    get_observations = getattr(raw, "_get_observations", None)
    if not callable(get_observations):
        raise RuntimeError("LIBERO environment lacks _get_observations")
    return get_observations()


def _native_mujoco_handles(env):
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - runtime image owns MuJoCo
        raise RuntimeError(
            "Exact paired replay requires the native mujoco Python package"
        ) from exc
    raw = _raw_env(env)
    native_model = getattr(raw.sim.model, "_model", raw.sim.model)
    native_data = getattr(raw.sim.data, "_data", raw.sim.data)
    return mujoco, native_model, native_data


def _mujoco_model_xml_sha256(env) -> str:
    """Fingerprint the compiled model without relying on Python object identity."""

    model = _raw_env(env).sim.model
    get_xml = getattr(model, "get_xml", None)
    if not callable(get_xml):
        raise RuntimeError("MuJoCo model lacks get_xml(); cannot verify replay layout")
    xml = get_xml()
    if not isinstance(xml, str) or not xml:
        raise RuntimeError("MuJoCo model returned an empty XML fingerprint source")
    return hashlib.sha256(xml.encode("utf-8")).hexdigest()


def _capture_mujoco_integration_checkpoint(
    env,
    *,
    include_model_fingerprint: bool = False,
    include_robosuite_python_state: bool = False,
) -> MujocoIntegrationCheckpoint:
    mujoco, native_model, native_data = _native_mujoco_handles(env)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = int(mujoco.mj_stateSize(native_model, state_spec))
    if state_size <= 0:
        raise RuntimeError("MuJoCo returned an empty mjSTATE_INTEGRATION")
    state = np.empty(state_size, dtype=np.float64)
    mujoco.mj_getState(native_model, native_data, state, state_spec)
    raw = _raw_env(env)
    timestep = getattr(raw, "timestep", None)
    cur_time = getattr(raw, "cur_time", None)
    done = getattr(raw, "done", None)
    return MujocoIntegrationCheckpoint(
        state=state.copy(),
        state_sha256=array_sha256(state),
        state_size=state_size,
        model_dimensions=tuple(
            int(getattr(native_model, name, -1)) for name in MUJOCO_MODEL_DIMENSIONS
        ),
        model_xml_sha256=(
            _mujoco_model_xml_sha256(env) if include_model_fingerprint else None
        ),
        timestep=None if timestep is None else int(timestep),
        cur_time=None if cur_time is None else float(cur_time),
        done=None if done is None else bool(done),
        robosuite_python_state=(
            _capture_robosuite_python_checkpoint(env)
            if include_robosuite_python_state
            else None
        ),
    )


def _restore_mujoco_integration_checkpoint(
    env,
    checkpoint: MujocoIntegrationCheckpoint,
    *,
    state_atol: float,
):
    mujoco, native_model, native_data = _native_mujoco_handles(env)
    if checkpoint.model_xml_sha256 is None:
        raise RuntimeError("Warm-start checkpoint lacks a MuJoCo model fingerprint")
    current_model_xml_sha256 = _mujoco_model_xml_sha256(env)
    if current_model_xml_sha256 != checkpoint.model_xml_sha256:
        raise RuntimeError(
            "Forced replay rebuilt a different MuJoCo model XML; exact pairing is "
            "impossible"
        )
    dimensions = tuple(
        int(getattr(native_model, name, -1)) for name in MUJOCO_MODEL_DIMENSIONS
    )
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state_size = int(mujoco.mj_stateSize(native_model, state_spec))
    if dimensions != checkpoint.model_dimensions or state_size != checkpoint.state_size:
        raise RuntimeError(
            "MuJoCo model/state dimensions changed between paired branches"
        )
    mujoco.mj_setState(native_model, native_data, checkpoint.state, state_spec)
    pre_forward = _capture_mujoco_integration_checkpoint(env)
    pre_forward_delta = _max_abs_delta(checkpoint.state, pre_forward.state)
    if pre_forward.state_sha256 != checkpoint.state_sha256:
        raise RuntimeError(
            "MuJoCo mj_setState did not exactly restore mjSTATE_INTEGRATION: "
            f"max_abs_delta={pre_forward_delta}"
        )
    mujoco.mj_forward(native_model, native_data)
    post_forward = _capture_mujoco_integration_checkpoint(env)
    post_forward_delta = _max_abs_delta(checkpoint.state, post_forward.state)
    if post_forward_delta > state_atol:
        raise RuntimeError(
            "MuJoCo integration state changed too much during mj_forward: "
            f"max_abs_delta={post_forward_delta} > state_atol={state_atol}"
        )

    raw = _raw_env(env)
    for name, value in (
        ("timestep", checkpoint.timestep),
        ("cur_time", checkpoint.cur_time),
        ("done", checkpoint.done),
    ):
        if value is not None and hasattr(raw, name):
            setattr(raw, name, value)
    if checkpoint.robosuite_python_state is None:
        raise RuntimeError("Warm-start checkpoint lacks robosuite Python control state")
    # mjSTATE_INTEGRATION does not include OSC goals, Panda gripper's
    # accumulated current_action, robot recent buffers, or Observable clocks.
    # Restoring these is required before the first paired policy step.
    obs = _restore_robosuite_python_checkpoint(env, checkpoint.robosuite_python_state)
    post_python = _capture_mujoco_integration_checkpoint(env)
    post_python_state_delta = _max_abs_delta(checkpoint.state, post_python.state)
    if post_python_state_delta > state_atol:
        raise RuntimeError(
            "MuJoCo integration state changed too much during Python-state restore: "
            f"max_abs_delta={post_python_state_delta} > state_atol={state_atol}"
        )

    # Controller.update(force=True) needs a forward pass to rebuild Jacobians and
    # mass matrices, but that pass may perturb qacc_warmstart. Reapply the exact
    # integration checkpoint without another forward so the next env.step starts
    # from precisely the same forward-dynamics input as the normal branch.
    mujoco.mj_setState(native_model, native_data, checkpoint.state, state_spec)
    restored = _capture_mujoco_integration_checkpoint(
        env,
        include_model_fingerprint=True,
        include_robosuite_python_state=True,
    )
    if restored.state_sha256 != checkpoint.state_sha256:
        raise RuntimeError(
            "Final MuJoCo integration-state restore was not exact: "
            f"max_abs_delta={_max_abs_delta(checkpoint.state, restored.state)}"
        )
    if not _replay_values_equal(
        checkpoint.robosuite_python_state, restored.robosuite_python_state
    ):
        raise RuntimeError("Final robosuite Python state restore was not exact")
    return (
        obs,
        restored,
        pre_forward_delta,
        post_forward_delta,
        post_python_state_delta,
    )


def _reset_to_initial_state(
    env,
    initial_state: np.ndarray,
    pair_seed: int,
    wait: int,
):
    raw = _raw_env(env)
    if getattr(raw, "hard_reset", None) is not True:
        raise RuntimeError(
            "Exact paired replay requires LIBERO / robosuite hard_reset=True"
        )
    set_seed(pair_seed)
    env.seed(pair_seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(wait):
        obs, _, _, _ = env.step(dummy_action())
    return obs


def _trace(
    model_input_rgb: np.ndarray,
    observed_source_rgb: np.ndarray,
    raw_action: np.ndarray,
    policy_action: np.ndarray,
    pre_vectors: dict[str, np.ndarray],
    integration_checkpoint: MujocoIntegrationCheckpoint,
) -> PrefixTrace:
    return PrefixTrace(
        model_input_rgb=np.asarray(model_input_rgb, dtype=np.uint8).copy(),
        model_input_rgb_sha256=array_sha256(model_input_rgb),
        observed_source_rgb=np.asarray(observed_source_rgb, dtype=np.uint8).copy(),
        observed_source_rgb_sha256=array_sha256(observed_source_rgb),
        stateful_sim_sha256=_sim_state_sha256(pre_vectors),
        stateful_sim={
            name: np.asarray(pre_vectors[name]).copy()
            for name in STATEFUL_SIM_FIELDS
            if name in pre_vectors
        },
        raw_action=np.asarray(raw_action).copy(),
        policy_action=np.asarray(policy_action).copy(),
        qpos=np.asarray(pre_vectors["qpos"]).copy(),
        qvel=np.asarray(pre_vectors["qvel"]).copy(),
        integration_state=integration_checkpoint.state.copy(),
        integration_state_sha256=integration_checkpoint.state_sha256,
    )


def _update_detach_state(
    *,
    step: int,
    pre_state: dict[str, Any],
    post_state: dict[str, Any],
    target_object: str,
    candidate: int | None,
    consecutive_ungrasped: int,
    required_steps: int,
) -> tuple[int | None, int, int | None, int | None]:
    pre_grasped = target_grasped(pre_state, target_object)
    post_grasped = target_grasped(post_state, target_object)
    transition = None
    if not post_grasped:
        if candidate is None:
            if not pre_grasped:
                return None, 0, None, None
            candidate = int(step)
            transition = int(step)
        consecutive_ungrasped += 1
    else:
        candidate = None
        consecutive_ungrasped = 0
    confirmed = (
        int(step)
        if candidate is not None and consecutive_ungrasped >= required_steps
        else None
    )
    return candidate, consecutive_ungrasped, transition, confirmed


def _run_episode(
    *,
    cfg: PairedReleaseRunConfig,
    model,
    processor,
    env,
    writer: ExtendedRunWriter,
    activation_collector: Layer31ActivationCollector,
    run_dir: Path,
    description: str,
    bddl_path: str,
    task_id: int,
    trial_idx: int,
    initial_state: np.ndarray,
    pair_id: str,
    pair_seed: int,
    episode_num: int,
    condition: str,
    target_object: str,
    destination: str | None,
    unnorm_key: str,
    max_steps: int,
    scheduled_trigger_step: int | None = None,
    normal_trace: dict[int, PrefixTrace] | None = None,
    normal_warm_start_checkpoint: MujocoIntegrationCheckpoint | None = None,
    reference_initial_z: float | None = None,
) -> tuple[
    dict[str, Any],
    dict[int, PrefixTrace],
    PrefixComparison,
    Exception | None,
    MujocoIntegrationCheckpoint,
]:
    if condition not in (NORMAL_CONDITION, FORCED_RELEASE_CONDITION):
        raise ValueError(f"Unsupported condition: {condition}")
    if condition == FORCED_RELEASE_CONDITION and (
        scheduled_trigger_step is None
        or normal_trace is None
        or normal_warm_start_checkpoint is None
        or reference_initial_z is None
    ):
        raise ValueError(
            "Forced release requires the normal trigger, trace, and warm-start state"
        )

    # Normal and forced branches use the same seeded hard-reset/controller
    # lifecycle and identical dummy warmup. Forced replay then overlays the
    # normal branch's complete MuJoCo forward-dynamics input before policy step 0.
    obs = _reset_to_initial_state(
        env,
        initial_state,
        pair_seed,
        cfg.env.num_steps_wait,
    )
    warm_start_restore_pre_forward_max_abs_delta = 0.0
    warm_start_restore_post_forward_max_abs_delta = 0.0
    warm_start_restore_post_python_state_max_abs_delta = 0.0
    if condition == FORCED_RELEASE_CONDITION:
        (
            obs,
            warm_start_checkpoint,
            warm_start_restore_pre_forward_max_abs_delta,
            warm_start_restore_post_forward_max_abs_delta,
            warm_start_restore_post_python_state_max_abs_delta,
        ) = _restore_mujoco_integration_checkpoint(
            env,
            normal_warm_start_checkpoint,
            state_atol=cfg.paired_release.state_atol,
        )
        warm_start_source = "seeded_hard_reset_then_normal_mjstate_and_python_replay"
    else:
        # Capture the normal branch exactly as returned by the official dummy
        # warmup. The forced branch restores these Observable clocks/caches
        # directly, rather than running an asymmetric extra sensor update.
        warm_start_checkpoint = _capture_mujoco_integration_checkpoint(
            env,
            include_model_fingerprint=True,
            include_robosuite_python_state=True,
        )
        warm_start_source = "seeded_hard_reset_warmup_checkpoint"
    prompt = writer.begin_episode(
        prompt_record={
            "episode_num": episode_num,
            "pair_id": pair_id,
            "condition": condition,
            "pair_seed": pair_seed,
            "task_id": task_id,
            "task_episode_idx": trial_idx,
            "task_description": description,
            "suite_seed": cfg.env.seed,
            "task_initial_state_index": trial_idx,
            "bddl_file": bddl_path,
            "target_object": target_object,
            "goal_destination": destination,
        },
        initial_state=initial_state,
    )

    frames: list[np.ndarray] = []
    trace: dict[int, PrefixTrace] = {}
    comparison = PrefixComparison()
    success = False
    caught_exception: str | None = None
    fatal: Exception | None = None
    invalid_reason: str | None = None
    video_error: str | None = None
    step_count = 0
    previous_pre = None
    control_dt = 1.0 / float(getattr(env.env, "control_freq", 20))

    detector: ReleaseTriggerDetector | None = None
    initial_z = reference_initial_z
    warm_start_sim_state_sha256: str | None = None
    warm_start_source_rgb_sha256: str | None = None
    t_cmd: int | None = None
    t_detach: int | None = None
    t_detach_confirmed: int | None = None
    t_obs: int | None = None
    t_post_detach_destination_contact: int | None = None
    t_goal_satisfied: int | None = None
    detach_candidate: int | None = None
    destination_contact_candidate: int | None = None
    consecutive_ungrasped = 0
    forced_open_steps = 0
    success_step: int | None = None

    for step in range(max_steps):
        break_after_step = False
        transaction_started = False
        try:
            if step % cfg.paired_release.disk_check_every_steps == 0:
                _require_free_disk(
                    run_dir,
                    cfg.paired_release.abort_below_free_disk_gb,
                    phase=f"{condition} pair={pair_id} step={step}",
                )
            source_rgb = get_source_rgb(obs, 224)
            pre = capture_snapshot(env, obs, previous=previous_pre, dt=control_dt)
            integration_checkpoint = _capture_mujoco_integration_checkpoint(env)
            previous_pre = pre
            if step == 0:
                warm_start_sim_state_sha256 = _sim_state_sha256(pre.sim_vectors)
                warm_start_source_rgb_sha256 = array_sha256(source_rgb)
            if initial_z is None:
                initial_z = target_z(pre.json_state, target_object)
            if detector is None:
                detector = ReleaseTriggerDetector(
                    target_object=target_object,
                    initial_z=float(initial_z),
                    stable_grasp_steps=cfg.paired_release.stable_grasp_steps,
                    min_lift_delta_m=cfg.paired_release.min_lift_delta_m,
                    trigger_delay_steps=cfg.paired_release.trigger_delay_steps,
                )
            detector.observe(step, pre.json_state)
            if (
                t_detach is not None
                and t_obs is None
                and step > t_detach
                and not target_grasped(pre.json_state, target_object)
            ):
                t_obs = int(step)

            reference = None
            model_input_rgb_override = None
            policy_input_source = "observed"
            if condition == FORCED_RELEASE_CONDITION and step <= int(
                scheduled_trigger_step
            ):
                reference = normal_trace.get(step)
                if reference is None:
                    invalid_reason = "normal_prefix_step_missing"
                else:
                    model_input_rgb_override = reference.model_input_rgb
                    policy_input_source = "normal_prefix_replay"

            common = {
                "episode_num": episode_num,
                "pair_id": pair_id,
                "condition": condition,
                "pair_seed": pair_seed,
                "task_id": task_id,
                "task_episode_idx": trial_idx,
                "task_description": description,
                "step_in_episode": step,
                "policy_input_source": policy_input_source,
                "sim_state_integration_sha256": (integration_checkpoint.state_sha256),
            }
            activation_collector.begin_step(common)
            transaction_started = True
            policy_out = infer_action_with_uncertainty(
                model,
                processor,
                cfg,
                source_rgb,
                description,
                unnorm_key,
                model_input_rgb_override=model_input_rgb_override,
            )
            raw_action = np.asarray(policy_out.raw_action).copy()
            policy_action = to_executed_libero_action(raw_action)
            current_trace = _trace(
                policy_out.model_input_rgb,
                source_rgb,
                raw_action,
                policy_action,
                pre.sim_vectors,
                integration_checkpoint,
            )
            trace[step] = current_trace

            forced_open_applied = False
            executed_action = policy_action.copy()
            if condition == NORMAL_CONDITION:
                if (
                    detector.trigger_step is not None
                    and t_cmd is None
                    and abs(
                        float(policy_action[-1])
                        - cfg.paired_release.forced_gripper_value
                    )
                    <= cfg.paired_release.action_atol
                ):
                    t_cmd = int(step)
            else:
                if step <= int(scheduled_trigger_step):
                    if reference is not None and not comparison.compare(
                        step,
                        reference,
                        current_trace,
                        action_atol=cfg.paired_release.action_atol,
                        state_atol=cfg.paired_release.state_atol,
                    ):
                        invalid_reason = "pretrigger_divergence"
                if invalid_reason is not None:
                    break_after_step = True
                elif step == scheduled_trigger_step:
                    lifted = (
                        target_z(pre.json_state, target_object)
                        - float(reference_initial_z)
                        >= cfg.paired_release.min_lift_delta_m
                    )
                    if not target_grasped(pre.json_state, target_object) or not lifted:
                        invalid_reason = "trigger_state_mismatch"
                        break_after_step = True
                    elif (
                        abs(
                            float(policy_action[-1])
                            - cfg.paired_release.forced_gripper_value
                        )
                        <= cfg.paired_release.action_atol
                    ):
                        invalid_reason = "policy_already_open_at_trigger"
                        break_after_step = True
                    else:
                        t_cmd = int(step)

                force_active = (
                    invalid_reason is None
                    and t_cmd is not None
                    and t_detach_confirmed is None
                    and forced_open_steps < cfg.paired_release.max_force_open_steps
                )
                if force_active:
                    executed_action = force_gripper_open(
                        policy_action,
                        forced_value=cfg.paired_release.forced_gripper_value,
                    )
                    forced_open_applied = True
                    forced_open_steps += 1

            next_obs, reward, done, info = env.step(executed_action.tolist())
            post_integration_checkpoint = _capture_mujoco_integration_checkpoint(env)
            post = capture_snapshot(env, next_obs, previous=pre, dt=control_dt)
            confirmed = None
            if t_cmd is not None and t_detach_confirmed is None:
                (
                    detach_candidate,
                    consecutive_ungrasped,
                    transition,
                    confirmed,
                ) = _update_detach_state(
                    step=step,
                    pre_state=pre.json_state,
                    post_state=post.json_state,
                    target_object=target_object,
                    candidate=detach_candidate,
                    consecutive_ungrasped=consecutive_ungrasped,
                    required_steps=cfg.paired_release.stable_detach_steps,
                )
                contact_now = target_destination_contact(
                    post.json_state, target_object, destination
                )
                if detach_candidate is None:
                    destination_contact_candidate = None
                elif contact_now and destination_contact_candidate is None:
                    destination_contact_candidate = int(step)
                if confirmed is not None:
                    # Commit the detach timestamp only after the object has
                    # remained ungrasped for the confirmation window. A
                    # one-frame contact jitter must not become t_detach.
                    t_detach = int(detach_candidate)
                    t_detach_confirmed = int(confirmed)
                    if destination_contact_candidate is not None:
                        t_post_detach_destination_contact = int(
                            destination_contact_candidate
                        )
                    if step > t_detach and t_obs is None:
                        t_obs = t_detach + 1
            if (
                t_detach is not None
                and t_post_detach_destination_contact is None
                and target_destination_contact(
                    post.json_state, target_object, destination
                )
            ):
                t_post_detach_destination_contact = int(step)
            goal_satisfied_post = (
                post.json_state["goals"].get("all_goal_predicates_satisfied") is True
            )
            if t_goal_satisfied is None and goal_satisfied_post:
                t_goal_satisfied = int(step)
            if (
                condition == FORCED_RELEASE_CONDITION
                and t_cmd is not None
                and t_detach_confirmed is None
                and forced_open_steps >= cfg.paired_release.max_force_open_steps
            ):
                invalid_reason = "no_confirmed_detach_within_max_force_open_steps"
                break_after_step = True

            writer.write_step(
                common=common,
                pre_json=pre.json_state,
                post_json=post.json_state,
                pre_vectors={
                    **pre.sim_vectors,
                    "mujoco_integration_state": integration_checkpoint.state,
                },
                post_vectors={
                    **post.sim_vectors,
                    "mujoco_integration_state": post_integration_checkpoint.state,
                },
                raw_action=raw_action,
                policy_action=policy_action,
                executed_action=executed_action,
                intervention={
                    "type": (
                        "none"
                        if condition == NORMAL_CONDITION
                        else "forced_gripper_open"
                    ),
                    "forced_open_applied": forced_open_applied,
                    "target_object": target_object,
                    "goal_destination": destination,
                    "scheduled_trigger_step": scheduled_trigger_step,
                    "forced_gripper_value": cfg.paired_release.forced_gripper_value,
                },
                policy={
                    "action_token_ids": policy_out.action_token_ids,
                    "uncertainty": policy_out.uncertainty,
                    "preprocessing": policy_out.preprocessing,
                },
                model_input_rgb=policy_out.model_input_rgb,
                observed_source_rgb=source_rgb,
                reward=float(reward),
                done=bool(done),
                info=_safe_info(info),
            )
            activation_collector.commit_step()
            transaction_started = False
            if cfg.output.save_video:
                frames.append(source_rgb)
            obs = next_obs
            step_count += 1
            if done:
                success = True
                if success_step is None:
                    success_step = int(step)
            if success:
                if condition == NORMAL_CONDITION:
                    # Match the official LIBERO/OpenVLA evaluation protocol:
                    # the unmodified control rollout ends at benchmark success.
                    break
                else:
                    # Forced release only needs a stable detach after success;
                    # its goal is expected to fail in the intervention branch.
                    confirmation_complete = t_detach_confirmed is not None
                    confirmation_timeout = (
                        step - int(success_step)
                        >= cfg.paired_release.stable_detach_steps
                    )
                    if confirmation_complete or confirmation_timeout:
                        break
            if break_after_step:
                break
        except Exception as exc:
            if transaction_started:
                activation_collector.abort_step()
            caught_exception = repr(exc)
            fatal = exc
            break

    if condition == FORCED_RELEASE_CONDITION and invalid_reason is None:
        if t_cmd is None:
            invalid_reason = "forced_ended_before_trigger"
        elif t_detach_confirmed is None:
            invalid_reason = "forced_rollout_ended_without_confirmed_detach"

    video_rel: str | None = None
    if cfg.output.save_video and frames:
        video_path = (
            run_dir
            / "videos"
            / f"{pair_id}--condition={condition}--success={success}.mp4"
        )
        try:
            _save_video(frames, video_path, cfg.output.video_fps)
            video_rel = str(video_path.relative_to(run_dir))
        except Exception as exc:
            video_error = repr(exc)
            if fatal is None:
                fatal = RuntimeError(f"Video encoding failed: {exc}")

    trigger = (
        detector.as_dict()
        if detector is not None
        else {
            "target_object": target_object,
            "initial_object_z": initial_z,
            "t_grasp": None,
            "t_stable_grasp": None,
            "t_lift": None,
            "trigger_step": None,
        }
    )
    if condition == FORCED_RELEASE_CONDITION:
        trigger["normal_scheduled_trigger_step"] = scheduled_trigger_step
    result = {
        "episode_num": episode_num,
        "pair_id": pair_id,
        "condition": condition,
        "pair_seed": pair_seed,
        "task_id": task_id,
        "task_episode_idx": trial_idx,
        "task_description": description,
        "target_object": target_object,
        "goal_destination": destination,
        "initial_state_sha256": prompt["initial_state_sha256"],
        "warm_start_sim_state_sha256": warm_start_sim_state_sha256,
        "warm_start_source_rgb_sha256": warm_start_source_rgb_sha256,
        "warm_start_integration_state_sha256": (warm_start_checkpoint.state_sha256),
        "warm_start_model_xml_sha256": warm_start_checkpoint.model_xml_sha256,
        "warm_start_robosuite_python_state_sha256": (
            _robosuite_python_state_sha256(warm_start_checkpoint.robosuite_python_state)
            if warm_start_checkpoint.robosuite_python_state is not None
            else None
        ),
        "warm_start_restore_pre_forward_max_abs_delta": (
            warm_start_restore_pre_forward_max_abs_delta
        ),
        "warm_start_restore_post_forward_max_abs_delta": (
            warm_start_restore_post_forward_max_abs_delta
        ),
        "warm_start_restore_post_python_state_max_abs_delta": (
            warm_start_restore_post_python_state_max_abs_delta
        ),
        "warm_start_source": warm_start_source,
        "success": success,
        "success_step": success_step,
        "num_actions": step_count,
        "caught_exception": caught_exception,
        "video_error": video_error,
        "video_path": video_rel,
        "trigger": trigger,
        "t_cmd": t_cmd,
        "t_detach": t_detach,
        "t_detach_confirmed": t_detach_confirmed,
        "t_obs": t_obs,
        "t_post_detach_destination_contact": (t_post_detach_destination_contact),
        "t_goal_satisfied": t_goal_satisfied,
        "detach_confirmation_steps": cfg.paired_release.stable_detach_steps,
        "forced_open_steps": forced_open_steps,
        "invalid_reason": invalid_reason,
    }
    final = writer.finish_episode(
        result=result,
        compress_npz=cfg.output.compress_npz,
    )
    activation_collector.flush_episode()
    return final, trace, comparison, fatal, warm_start_checkpoint


def _empty_condition_summary() -> dict[str, int | float]:
    return {"episodes": 0, "successes": 0, "failures": 0, "success_rate": 0.0}


def _update_condition_summary(
    aggregate: dict[str, Any],
    condition: str,
    success: bool,
) -> None:
    summary = aggregate["conditions"][condition]
    summary["episodes"] += 1
    summary["successes" if success else "failures"] += 1


def _finish_condition_rates(aggregate: dict[str, Any]) -> None:
    for summary in aggregate["conditions"].values():
        summary["success_rate"] = (
            summary["successes"] / summary["episodes"] if summary["episodes"] else 0.0
        )


def collect_paired_release_libero(
    cfg: PairedReleaseRunConfig,
) -> dict[str, Any]:
    """Collect matched normal/forced-release candidates and valid pairs."""

    from libero.libero import benchmark

    output_root = Path(cfg.output.root_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    free_disk_gb_at_start = _require_free_disk(
        output_root,
        cfg.paired_release.min_free_disk_gb_at_start,
        phase="collection preflight",
    )
    set_seed(cfg.env.seed)
    benchmark_cls = benchmark.get_benchmark_dict()[cfg.env.task_suite_name]
    suite = benchmark_cls()
    task_ids = _resolve_task_ids(cfg.env.task_ids, suite.n_tasks)
    _require_single_task_run(task_ids)
    model = load_openvla(cfg)
    processor = load_processor(cfg)
    unnorm_key = _resolve_unnorm_key(model, cfg.env.task_suite_name)
    action_vocab_size = _openvla_action_vocab_size(model)
    if cfg.env.task_suite_name not in _MAX_STEPS_PER_SUITE:
        raise ValueError(f"Unsupported task suite: {cfg.env.task_suite_name}")
    max_steps = _MAX_STEPS_PER_SUITE[cfg.env.task_suite_name]

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = _new_run_dir(
        cfg.output.root_dir,
        f"{cfg.env.task_suite_name}-paired-release",
    )
    manifest = {
        "schema_version": "extended_openvla_libero_paired_release_v7",
        "collection_status": "in_progress",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code": _git_state(repo_root),
        "runtime": _runtime_provenance(),
        "free_disk_gb_at_start": free_disk_gb_at_start,
        "config": cfg.to_dict(),
        "resolved_task_ids": task_ids,
        "conditions": [NORMAL_CONDITION, FORCED_RELEASE_CONDITION],
        "pairing": {
            "unit": "task_id + task_initial_state_index + pair_seed",
            "normal_first": True,
            "forced_skipped_when_normal_ineligible": True,
            "adaptive_stop_per_task": True,
            "max_attempts_per_task": cfg.env.num_trials_per_task,
            "target_valid_pairs_per_task": (
                cfg.paired_release.target_valid_pairs_per_task
            ),
            "target_primary_pairs_per_task": (
                cfg.paired_release.target_primary_pairs_per_task
            ),
            "prefix_comparison_includes_trigger_step": True,
            "prefix_policy_input": (
                "replay exact normal model_input_rgb through trigger; retain "
                "forced observed_source_rgb separately for audit"
            ),
            "forced_warm_start": (
                "repeat the same seeded hard reset and controller warmup, verify "
                "the compiled model XML fingerprint, then restore normal "
                "mjSTATE_INTEGRATION plus OSC, gripper, robot-buffer, and "
                "Observable state before policy step 0"
            ),
            "prefix_sim_state_audit": (
                "persist and compare complete mjSTATE_INTEGRATION through trigger"
            ),
            "warm_start_hashes": (
                "MuJoCo and robosuite Python checkpoints are exact restore gates; "
                "rendered-source hashes remain audit-only"
            ),
            "detach_confirmation_steps": cfg.paired_release.stable_detach_steps,
            "normal_termination": "official_libero_success_or_horizon",
            "primary_analysis_filter": "valid_pair and normal_success",
        },
        "alignment": "pre_state + raw/policy/executed action -> post_state",
        "activation_stream": {
            "layer": 31,
            "submodule": "post_mlp_residual",
            "format": "dense torch float32 shards + activation_index.jsonl",
            "forwards_per_policy_step": 7,
            "step_transaction": True,
            "episode_boundary_flush": True,
        },
        "policy_uncertainty": {
            "action_vocab_size": action_vocab_size,
            "action_vocab_definition": "config.n_action_bins (token ids)",
            "stored": [
                "full_next_token_entropy/top1_probability/top1_top2_margin",
                "action_token_probability_mass",
                "conditional_action_token_entropy/top1_probability/top1_top2_margin",
                "selected_token_probability",
                "selected_token_conditional_probability/rank",
            ],
            "full_logits_stored": False,
        },
        "vision": {
            "model_input_rgb": "lossless uint8 arrays in per-episode NPZ",
            "rollout_video": bool(cfg.output.save_video),
            "depth": False,
            "segmentation": False,
        },
    }
    aggregate: dict[str, Any] = {
        "run_dir": str(run_dir),
        "max_attempts_per_task": cfg.env.num_trials_per_task,
        "target_valid_pairs_per_task": (cfg.paired_release.target_valid_pairs_per_task),
        "target_primary_pairs_per_task": (
            cfg.paired_release.target_primary_pairs_per_task
        ),
        "pair_candidates": 0,
        "eligible_pairs": 0,
        "valid_pairs": 0,
        "invalid_pairs": 0,
        "ineligible_pairs": 0,
        "primary_analysis_pairs": 0,
        "per_task": {},
        "episodes": 0,
        "conditions": {
            NORMAL_CONDITION: _empty_condition_summary(),
            FORCED_RELEASE_CONDITION: _empty_condition_summary(),
        },
    }

    activation_collector = None
    with ExtendedRunWriter(
        run_dir,
        flush_every_step=cfg.output.flush_jsonl_every_step,
        enable_pair_results=True,
    ) as writer:
        writer.write_manifest(manifest)
        activation_collector = Layer31ActivationCollector(
            model,
            run_dir / "sae_activations" / "post_mlp_residual",
            cfg.collection.activation_flush_every,
        )
        try:
            for task_id in task_ids:
                task = suite.get_task(task_id)
                initial_states = suite.get_task_init_states(task_id)
                if len(initial_states) < cfg.env.num_trials_per_task:
                    raise ValueError(
                        f"Task {task_id} has {len(initial_states)} initial states, "
                        f"but {cfg.env.num_trials_per_task} were requested"
                    )
                quota = TaskPairQuota(
                    task_id=task_id,
                    max_attempts=cfg.env.num_trials_per_task,
                    target_valid_pairs=(cfg.paired_release.target_valid_pairs_per_task),
                    target_primary_pairs=(
                        cfg.paired_release.target_primary_pairs_per_task
                    ),
                )
                aggregate["per_task"][str(task_id)] = quota.as_dict()
                env, description, bddl_path = _make_env(
                    task, cfg.env.resolution, cfg.env.seed
                )
                try:
                    env.reset()
                    preflight_obs = env.set_init_state(initial_states[0])
                    preflight_obs, _, _, _ = env.step(dummy_action())
                    capabilities = validate_preflight(
                        env,
                        preflight_obs,
                        strict=cfg.collection.strict_preflight,
                    )
                    preflight = capture_snapshot(
                        env, preflight_obs, previous=None, dt=None
                    )
                    task_key = str(task_id)
                    explicit_target = cfg.paired_release.target_object_by_task.get(
                        task_key
                    )
                    target_object = resolve_target_object(
                        preflight.json_state,
                        explicit=explicit_target,
                    )
                    explicit_destination = cfg.paired_release.destination_by_task.get(
                        task_key
                    )
                    destination = resolve_goal_destination(
                        preflight.json_state,
                        target_object,
                        explicit=explicit_destination,
                    )
                    schema = build_simulator_schema(env, preflight_obs)
                    schema.update(
                        {
                            "task_id": task_id,
                            "task_description": description,
                            "bddl_file": bddl_path,
                            "capabilities": capabilities,
                            "paired_release_target_object": target_object,
                            "paired_release_goal_destination": destination,
                            "paired_release_target_source": (
                                "config"
                                if explicit_target is not None
                                else "first BDDL goal object argument"
                            ),
                            "paired_release_destination_source": (
                                "config"
                                if explicit_destination is not None
                                else "BDDL goal argument after target"
                            ),
                        }
                    )
                    writer.write_schema(task_id, schema)

                    for trial_idx in range(cfg.env.num_trials_per_task):
                        if quota.target_reached:
                            break
                        quota.begin_attempt(trial_idx)
                        aggregate["pair_candidates"] += 1
                        initial_state = np.asarray(initial_states[trial_idx]).copy()
                        pair_seed = _pair_seed(cfg.env.seed, task_id, trial_idx)
                        pair_id = _pair_id(
                            cfg.env.task_suite_name,
                            task_id,
                            trial_idx,
                            pair_seed,
                        )
                        normal_episode_num = aggregate["episodes"] + 1
                        (
                            normal,
                            normal_trace,
                            _,
                            normal_fatal,
                            normal_warm_start_checkpoint,
                        ) = _run_episode(
                            cfg=cfg,
                            model=model,
                            processor=processor,
                            env=env,
                            writer=writer,
                            activation_collector=activation_collector,
                            run_dir=run_dir,
                            description=description,
                            bddl_path=bddl_path,
                            task_id=task_id,
                            trial_idx=trial_idx,
                            initial_state=initial_state,
                            pair_id=pair_id,
                            pair_seed=pair_seed,
                            episode_num=normal_episode_num,
                            condition=NORMAL_CONDITION,
                            target_object=target_object,
                            destination=destination,
                            unnorm_key=unnorm_key,
                            max_steps=max_steps,
                        )
                        aggregate["episodes"] += 1
                        _update_condition_summary(
                            aggregate, NORMAL_CONDITION, normal["success"]
                        )
                        trigger_step = normal["trigger"]["trigger_step"]
                        policy_open_at_trigger = (
                            trigger_step is not None
                            and int(trigger_step) in normal_trace
                            and abs(
                                float(normal_trace[int(trigger_step)].policy_action[-1])
                                - cfg.paired_release.forced_gripper_value
                            )
                            <= cfg.paired_release.action_atol
                        )

                        skip_reason = None
                        if normal_fatal is not None:
                            skip_reason = "normal_exception"
                        elif trigger_step is None:
                            skip_reason = "normal_no_stable_grasp_and_lift"
                        elif policy_open_at_trigger:
                            skip_reason = "normal_policy_already_open_at_trigger"

                        if skip_reason is not None:
                            aggregate["ineligible_pairs"] += 1
                            quota.record_ineligible()
                            aggregate["per_task"][str(task_id)] = quota.as_dict()
                            pair_result = {
                                "pair_id": pair_id,
                                "status": "ineligible",
                                "eligible": False,
                                "valid": False,
                                "skip_reason": skip_reason,
                                "invalid_reason": None,
                                "task_id": task_id,
                                "task_episode_idx": trial_idx,
                                "pair_seed": pair_seed,
                                "target_object": target_object,
                                "goal_destination": destination,
                                "initial_state_sha256": normal["initial_state_sha256"],
                                "normal_episode_num": normal_episode_num,
                                "forced_episode_num": None,
                                "normal_success": bool(normal["success"]),
                                "normal_natural_release_observed": bool(
                                    normal.get("t_cmd") is not None
                                    and normal.get("t_detach_confirmed") is not None
                                ),
                                "primary_analysis_eligible": False,
                                "normal": normal,
                                "forced_release": None,
                                "prefix_comparison": None,
                            }
                            writer.write_pair_result(pair_result)
                            print(
                                f"pair={pair_id} status=ineligible "
                                f"reason={skip_reason}",
                                flush=True,
                            )
                            if normal_fatal is not None:
                                raise RuntimeError(
                                    "Normal paired collection failed: "
                                    f"{normal_fatal!r}"
                                ) from normal_fatal
                            continue

                        aggregate["eligible_pairs"] += 1
                        forced_episode_num = aggregate["episodes"] + 1
                        (
                            forced,
                            _,
                            comparison,
                            forced_fatal,
                            _,
                        ) = _run_episode(
                            cfg=cfg,
                            model=model,
                            processor=processor,
                            env=env,
                            writer=writer,
                            activation_collector=activation_collector,
                            run_dir=run_dir,
                            description=description,
                            bddl_path=bddl_path,
                            task_id=task_id,
                            trial_idx=trial_idx,
                            initial_state=initial_state,
                            pair_id=pair_id,
                            pair_seed=pair_seed,
                            episode_num=forced_episode_num,
                            condition=FORCED_RELEASE_CONDITION,
                            target_object=target_object,
                            destination=destination,
                            unnorm_key=unnorm_key,
                            max_steps=max_steps,
                            scheduled_trigger_step=int(trigger_step),
                            normal_trace=normal_trace,
                            normal_warm_start_checkpoint=(normal_warm_start_checkpoint),
                            reference_initial_z=float(
                                normal["trigger"]["initial_object_z"]
                            ),
                        )
                        aggregate["episodes"] += 1
                        _update_condition_summary(
                            aggregate,
                            FORCED_RELEASE_CONDITION,
                            forced["success"],
                        )

                        invalid_reasons, pairing_audit = _pair_invalid_reasons(
                            normal, forced, forced_fatal
                        )
                        invalid_reason = invalid_reasons[0] if invalid_reasons else None
                        valid = invalid_reason is None
                        status = "valid" if valid else "invalid"
                        aggregate["valid_pairs" if valid else "invalid_pairs"] += 1
                        normal_natural_release = bool(
                            normal.get("t_cmd") is not None
                            and normal.get("t_detach_confirmed") is not None
                        )
                        primary = bool(valid and normal["success"])
                        aggregate["primary_analysis_pairs"] += int(primary)
                        quota.record_eligible(valid=valid, primary=primary)
                        aggregate["per_task"][str(task_id)] = quota.as_dict()
                        pair_result = {
                            "pair_id": pair_id,
                            "status": status,
                            "eligible": True,
                            "valid": valid,
                            "skip_reason": None,
                            "invalid_reason": invalid_reason,
                            "invalid_reasons": invalid_reasons,
                            "pairing_audit": pairing_audit,
                            "task_id": task_id,
                            "task_episode_idx": trial_idx,
                            "pair_seed": pair_seed,
                            "target_object": target_object,
                            "goal_destination": destination,
                            "initial_state_sha256": normal["initial_state_sha256"],
                            "normal_episode_num": normal_episode_num,
                            "forced_episode_num": forced_episode_num,
                            "normal_success": bool(normal["success"]),
                            "normal_natural_release_observed": (normal_natural_release),
                            "primary_analysis_eligible": primary,
                            "normal": normal,
                            "forced_release": forced,
                            "prefix_comparison": comparison.as_dict(),
                        }
                        writer.write_pair_result(pair_result)
                        print(
                            f"pair={pair_id} status={status} "
                            f"trigger={trigger_step} "
                            f"detach={forced['t_detach']} "
                            f"confirmed={forced['t_detach_confirmed']} "
                            f"task_valid={quota.valid_pairs}/"
                            f"{quota.target_valid_pairs} "
                            f"task_primary={quota.primary_analysis_pairs}/"
                            f"{quota.target_primary_pairs} "
                            f"attempt={quota.attempts}/{quota.max_attempts}",
                            flush=True,
                        )
                        if forced_fatal is not None:
                            raise RuntimeError(
                                "Forced paired collection failed: " f"{forced_fatal!r}"
                            ) from forced_fatal
                        if not valid and cfg.paired_release.fail_on_invalid_pair:
                            raise RuntimeError(
                                f"Invalid paired release {pair_id}: "
                                f"{invalid_reason}"
                            )
                    if not quota.target_reached:
                        failure = {
                            "type": "task_pair_target_shortfall",
                            "task_id": task_id,
                            "task_progress": quota.as_dict(),
                        }
                        aggregate["failure"] = failure
                        _finish_condition_rates(aggregate)
                        (run_dir / "summary.json").write_text(
                            json.dumps(jsonable(aggregate), indent=2) + "\n",
                            encoding="utf-8",
                        )
                        failed_manifest = {
                            **manifest,
                            "collection_status": "failed_target_shortfall",
                            "failed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "failure": failure,
                            "summary": aggregate,
                        }
                        writer.write_manifest(failed_manifest)
                        raise RuntimeError(
                            f"Task {task_id} exhausted {quota.max_attempts} "
                            "attempts before reaching its pair targets: "
                            f"valid={quota.valid_pairs}/"
                            f"{quota.target_valid_pairs}, "
                            f"primary={quota.primary_analysis_pairs}/"
                            f"{quota.target_primary_pairs}. "
                            "COLLECTION_COMPLETE was not created."
                        )
                    print(
                        f"task={task_id} pair_targets_reached "
                        f"valid={quota.valid_pairs} "
                        f"primary={quota.primary_analysis_pairs} "
                        f"attempts={quota.attempts}",
                        flush=True,
                    )
                finally:
                    env.close()
        finally:
            if activation_collector is not None:
                activation_collector.close()

        _finish_condition_rates(aggregate)
        aggregate["valid_rate_among_eligible"] = (
            aggregate["valid_pairs"] / aggregate["eligible_pairs"]
            if aggregate["eligible_pairs"]
            else 0.0
        )
        (run_dir / "summary.json").write_text(
            json.dumps(jsonable(aggregate), indent=2) + "\n",
            encoding="utf-8",
        )
    from event_sae.openvla.extended_collection.paired_validate import (
        validate_paired_release_run,
    )

    validate_paired_release_run(
        run_dir,
        min_valid_pairs_per_task=(cfg.paired_release.target_valid_pairs_per_task),
        min_primary_pairs_per_task=(cfg.paired_release.target_primary_pairs_per_task),
        require_complete=False,
    )
    manifest["collection_status"] = "complete"
    manifest["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["summary"] = aggregate
    manifest_path = run_dir / "manifest.json"
    temporary_manifest = run_dir / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(jsonable(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    (run_dir / "COLLECTION_COMPLETE").write_text(
        "paired release collection complete\n",
        encoding="utf-8",
    )
    print(f"PAIRED_RELEASE_COLLECTION_OK: {run_dir}")
    return aggregate
