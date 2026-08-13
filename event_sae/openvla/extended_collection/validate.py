"""Structural validation for an extended collection before upload or deletion."""

from __future__ import annotations

import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.lib import format as npformat

from event_sae.openvla.extended_collection.writer import array_sha256


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _iter_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc


def _npz_shape(path: Path, key: str) -> tuple[int, ...]:
    """Read an NPY member header without decompressing the full array."""

    with zipfile.ZipFile(path) as archive:
        name = f"{key}.npy"
        if name not in archive.namelist():
            raise KeyError(f"{path}: missing {key}")
        with archive.open(name) as stream:
            version = npformat.read_magic(stream)
            if version == (1, 0):
                shape, _, _ = npformat.read_array_header_1_0(stream)
            elif version == (2, 0):
                shape, _, _ = npformat.read_array_header_2_0(stream)
            else:
                shape, _, _ = npformat._read_array_header(stream, version)
    return tuple(int(value) for value in shape)


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_keys(item)


def _record_key(row: dict[str, Any]) -> tuple[int, int]:
    return int(row["episode_num"]), int(row["step_in_episode"])


def _finite_vector(value: Any, *, size: int | None = None) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.dtype.kind not in "biufc" or not np.all(np.isfinite(array)):
        return False
    return size is None or int(array.size) == size


def _validate_rich_state(
    state: dict[str, Any], key: tuple[int, int], phase: str
) -> None:
    robot = state["robot"]
    required_robot = {
        "joint_position": None,
        "joint_velocity": None,
        "joint_torque_command": None,
        "eef_position": 3,
        "eef_quaternion_xyzw": 4,
        "eef_linear_velocity": 3,
        "eef_angular_velocity": 3,
        "gripper_qpos": None,
        "gripper_qvel": None,
    }
    for name, size in required_robot.items():
        if not _finite_vector(robot.get(name), size=size):
            raise ValueError(f"Invalid {phase}.robot.{name} at policy step {key}")

    named_bodies = list(state["objects"].values()) + list(state["fixtures"].values())
    if not named_bodies:
        raise ValueError(f"No object/fixture state at policy step {key}")
    for record in named_bodies:
        for name, size in (
            ("position_world", 3),
            ("quaternion_wxyz_world", 4),
            ("linear_velocity_world", 3),
            ("angular_velocity_world", 3),
        ):
            if not _finite_vector(record.get(name), size=size):
                raise ValueError(f"Invalid {phase} body {name} at policy step {key}")

    contact_state = state["contact_and_grasp"]
    for contact in contact_state["contacts"]:
        if not _finite_vector(contact.get("force_torque_contact_frame"), size=6):
            raise ValueError(f"Invalid {phase} contact force at policy step {key}")
        if not isinstance(contact.get("potentially_unwanted_collision"), bool):
            raise ValueError(
                f"Missing {phase} collision candidate at policy step {key}"
            )
    if not contact_state["grasped_objects"] or not all(
        isinstance(value, bool) for value in contact_state["grasped_objects"].values()
    ):
        raise ValueError(f"Invalid {phase} grasp flags at policy step {key}")

    goals = state["goals"]
    if not goals["goal_predicates"] or not all(
        isinstance(row.get("satisfied"), bool) and row.get("error") is None
        for row in goals["goal_predicates"]
    ):
        raise ValueError(f"Invalid {phase} goal predicates at policy step {key}")


def _validate_uncertainty(row: dict[str, Any], key: tuple[int, int]) -> None:
    if any("logit" in name.lower() for name in _walk_keys(row)):
        raise ValueError(
            "Full/raw logits-like field found in policy uncertainty output"
        )
    if len(row.get("action_token_ids", [])) != 7:
        raise ValueError(f"Expected seven action token ids at policy step {key}")
    dimensions = row.get("uncertainty", {}).get("per_action_dimension", [])
    if len(dimensions) != 7 or [x.get("action_dimension") for x in dimensions] != list(
        range(7)
    ):
        raise ValueError(f"Expected seven ordered uncertainty dimensions at step {key}")
    for dimension in dimensions:
        if not dimension.get("selected_token_is_action_token"):
            raise ValueError(f"Generated a non-action token at policy step {key}")
        for name in (
            "selected_token_probability",
            "selected_token_conditional_probability",
        ):
            value = dimension.get(name)
            if not _finite_vector(value, size=1) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"Invalid {name} at policy step {key}")
        for group in ("full_next_token", "conditional_action_token"):
            metrics = dimension.get(group, {})
            for name in ("entropy_nats", "normalized_entropy"):
                if not _finite_vector(metrics.get(name), size=1):
                    raise ValueError(f"Invalid {group}.{name} at policy step {key}")
                if float(metrics[name]) < 0.0:
                    raise ValueError(f"Negative {group}.{name} at policy step {key}")
            for name in ("top1_probability", "top1_top2_margin"):
                if (
                    not _finite_vector(metrics.get(name), size=1)
                    or not 0.0 <= float(metrics[name]) <= 1.0
                ):
                    raise ValueError(f"Invalid {group}.{name} at policy step {key}")
        mass = dimension["conditional_action_token"].get(
            "probability_mass_in_full_vocabulary"
        )
        if not _finite_vector(mass, size=1) or not 0.0 <= float(mass) <= 1.0:
            raise ValueError(f"Invalid action-token probability mass at step {key}")


def validate_extended_run(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "extended_openvla_libero_v1":
        raise ValueError("Unexpected extended dataset schema")
    if manifest["activation_stream"]["layer"] != 31:
        raise ValueError("Only layer-31 activations are valid for this dataset")
    if manifest["policy_uncertainty"].get("full_logits_stored") is not False:
        raise ValueError(
            "Manifest must explicitly state that full logits are not stored"
        )

    prompts = _jsonl(run_dir / "prompt_records.jsonl")
    episodes = _jsonl(run_dir / "episode_results.jsonl")
    expected = len(manifest["resolved_task_ids"]) * int(
        manifest["config"]["env"]["num_trials_per_task"]
    )
    if not (len(prompts) == len(episodes) == expected):
        raise ValueError(
            f"Episode count mismatch: prompts={len(prompts)} results={len(episodes)} expected={expected}"
        )
    prompt_by_episode = {int(row["episode_num"]): row for row in prompts}
    result_by_episode = {int(row["episode_num"]): row for row in episodes}
    if set(prompt_by_episode) != set(result_by_episode):
        raise ValueError("Prompt/result episode ids differ")
    trajectory_order: list[tuple[int, int]] = []
    trajectory_keys: set[tuple[int, int]] = set()
    steps_by_episode: dict[int, list[int]] = defaultdict(list)
    for row in _iter_jsonl(run_dir / "trajectory_records.jsonl"):
        key = _record_key(row)
        if key in trajectory_keys:
            raise ValueError(f"Duplicate trajectory policy-step key: {key}")
        trajectory_keys.add(key)
        trajectory_order.append(key)
        steps_by_episode[key[0]].append(key[1])
        if not _finite_vector(row.get("raw_openvla_action"), size=7):
            raise ValueError(f"Invalid raw OpenVLA action at policy step {key}")
        if not _finite_vector(row.get("executed_libero_action"), size=7):
            raise ValueError(f"Invalid executed LIBERO action at policy step {key}")
        if not _finite_vector(row.get("reward"), size=1):
            raise ValueError(f"Invalid reward at policy step {key}")
        _validate_rich_state(row["pre"], key, "pre")
        _validate_rich_state(row["post"], key, "post")

    action_count = 0
    for action_count, row in enumerate(
        _iter_jsonl(run_dir / "action_records.jsonl"), start=1
    ):
        index = action_count - 1
        if (
            index >= len(trajectory_order)
            or _record_key(row) != trajectory_order[index]
        ):
            raise ValueError(
                f"Action/trajectory policy-step order mismatch at row {action_count}"
            )
        if not _finite_vector(
            row.get("raw_openvla_action"), size=7
        ) or not _finite_vector(row.get("executed_libero_action"), size=7):
            raise ValueError(f"Invalid action record at row {action_count}")

    uncertainty_count = 0
    for uncertainty_count, row in enumerate(
        _iter_jsonl(run_dir / "policy_uncertainty.jsonl"), start=1
    ):
        index = uncertainty_count - 1
        if (
            index >= len(trajectory_order)
            or _record_key(row) != trajectory_order[index]
        ):
            raise ValueError(
                f"Uncertainty/trajectory policy-step order mismatch at row {uncertainty_count}"
            )
        _validate_uncertainty(row, trajectory_order[index])
    if not (len(trajectory_order) == action_count == uncertainty_count):
        raise ValueError(
            "Step count mismatch: "
            f"trajectory={len(trajectory_order)} actions={action_count} "
            f"uncertainty={uncertainty_count}"
        )

    for episode_num, result in result_by_episode.items():
        prompt = prompt_by_episode[episode_num]
        initial_path = run_dir / prompt["initial_state_path"]
        with np.load(initial_path) as initial:
            digest = array_sha256(initial["initial_state"])
        if digest != prompt["initial_state_sha256"]:
            raise ValueError(f"Initial-state checksum mismatch: episode {episode_num}")
        steps = int(result["recorded_steps"])
        expected_steps = list(range(steps))
        actual_steps = sorted(steps_by_episode[episode_num])
        if actual_steps != expected_steps:
            raise ValueError(f"Non-contiguous policy steps: episode {episode_num}")

        sim_path = run_dir / result["sim_state_path"]
        for key in (
            "step_in_episode",
            "pre_qpos",
            "post_qpos",
            "pre_qvel",
            "post_qvel",
            "pre_qacc",
            "post_qacc",
            "pre_ctrl",
            "post_ctrl",
        ):
            shape = _npz_shape(sim_path, key)
            if not shape or shape[0] != steps:
                raise ValueError(
                    f"{sim_path}:{key} has shape {shape}, expected first dim {steps}"
                )
        with np.load(sim_path) as simulator:
            for name in simulator.files:
                array = simulator[name]
                if array.dtype.kind in "biufc" and not np.all(np.isfinite(array)):
                    raise ValueError(f"Non-finite simulator values: {sim_path}:{name}")

        if manifest["config"]["output"]["save_model_input_rgb"]:
            if not result.get("vision_path"):
                raise ValueError(
                    f"Missing exact model-input RGB: episode {episode_num}"
                )
            vision_shape = _npz_shape(
                run_dir / result["vision_path"], "model_input_rgb"
            )
            if vision_shape != (steps, 224, 224, 3):
                raise ValueError(
                    f"Unexpected model-input RGB shape: episode {episode_num}: {vision_shape}"
                )
        if manifest["config"]["output"]["save_video"]:
            video = result.get("video_path")
            if not video or not (run_dir / video).is_file():
                raise ValueError(f"Missing rollout video: episode {episode_num}")

    activation_dir = run_dir / "sae_activations" / "post_mlp_residual"
    activation_keys: set[tuple[int, int]] = set()
    activation_counts: Counter[tuple[int, int]] = Counter()
    shard_rows: dict[str, int] = {}
    shard_names: set[str] = set()
    activation_index_count = 0
    for activation_index_count, row in enumerate(
        _iter_jsonl(activation_dir / "activation_index.jsonl"), start=1
    ):
        if int(row["layer_idx"]) != 31:
            raise ValueError(f"Non-layer31 activation index row: {row}")
        key = (int(row["episode_num"]), int(row["step_in_episode"]))
        if key not in trajectory_keys:
            raise ValueError(f"Activation index points outside policy steps: {key}")
        activation_keys.add(key)
        activation_counts[key] += 1
        shard_names.add(row["shard_path"])
        row_start, row_end = int(row["row_start"]), int(row["row_end"])
        if row_end <= row_start:
            raise ValueError(f"Empty activation range: {row}")
        expected_start = shard_rows.get(row["shard_path"], 0)
        if row_start != expected_start:
            raise ValueError(
                f"Non-contiguous activation rows in {row['shard_path']}: "
                f"expected start {expected_start}, got {row_start}"
            )
        shard_rows[row["shard_path"]] = row_end
        if int(row["global_forward_idx"]) != activation_index_count:
            raise ValueError("Activation global_forward_idx is not contiguous from 1")
    missing_activation = trajectory_keys - activation_keys
    if missing_activation:
        raise ValueError(
            f"Policy steps without activation records: {sorted(missing_activation)[:5]}"
        )
    expected_forwards = int(
        manifest["activation_stream"].get("forwards_per_policy_step", 7)
    )
    wrong_counts = {
        key: activation_counts[key]
        for key in trajectory_keys
        if activation_counts[key] != expected_forwards
    }
    if wrong_counts:
        raise ValueError(
            f"Expected {expected_forwards} layer-31 forwards per policy step; "
            f"mismatches={list(wrong_counts.items())[:5]}"
        )
    for shard_name in shard_names:
        shard_path = activation_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(shard_path)
        import torch

        try:
            tensor = torch.load(
                str(shard_path), map_location="cpu", mmap=True, weights_only=True
            )
        except TypeError:  # compatibility with an older torch runtime
            tensor = torch.load(str(shard_path), map_location="cpu", mmap=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"Unexpected activation shard payload: {shard_path}")
        if tuple(tensor.shape) != (shard_rows[shard_name], 4096):
            raise ValueError(
                f"Activation shard shape mismatch: {shard_name}: "
                f"tensor={tuple(tensor.shape)} index={(shard_rows[shard_name], 4096)}"
            )
        del tensor

    summary = {
        "schema_version": manifest["schema_version"],
        "episodes": len(episodes),
        "policy_steps": len(trajectory_order),
        "activation_index_records": activation_index_count,
        "activation_shards": len(shard_names),
        "successes": sum(bool(row["success"]) for row in episodes),
    }
    return summary
