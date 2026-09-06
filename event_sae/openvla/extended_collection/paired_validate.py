"""Semantic validation for paired normal/forced-release LIBERO collections."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from event_sae.openvla.extended_collection.controlled_release import (
    FORCED_RELEASE_CONDITION,
    NORMAL_CONDITION,
    only_gripper_was_overridden,
    target_destination_contact,
    target_grasped,
)
from event_sae.openvla.extended_collection.validate import validate_extended_run


PAIRED_SCHEMA_VERSION = "extended_openvla_libero_paired_release_v2"


def _json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc


def _episode_map(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        episode_num = int(row["episode_num"])
        if episode_num in result:
            raise ValueError(f"Duplicate {label} episode: {episode_num}")
        result[episode_num] = row
    return result


def _assert_episode_identity(
    row: dict[str, Any],
    expected: dict[str, Any],
    *,
    source: str,
) -> None:
    episode_num = int(row["episode_num"])
    checks = {
        "pair_id": str,
        "condition": str,
        "pair_seed": int,
        "task_id": int,
        "task_episode_idx": int,
    }
    for field, cast in checks.items():
        if field not in row or cast(row[field]) != cast(expected[field]):
            raise ValueError(
                f"{source} identity mismatch for episode {episode_num}: {field}"
            )


def _max_delta(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def _load_npz(path: Path, relative: str) -> dict[str, np.ndarray]:
    target = path / relative
    if not target.is_file():
        raise FileNotFoundError(target)
    with np.load(target) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _validate_prefix(
    root: Path,
    normal_result: dict[str, Any],
    forced_result: dict[str, Any],
    *,
    through_step: int,
    action_rows: dict[int, list[dict[str, Any]]],
    action_atol: float,
    state_atol: float,
) -> dict[str, float | bool]:
    count = through_step + 1
    normal_actions = action_rows[int(normal_result["episode_num"])]
    forced_actions = action_rows[int(forced_result["episode_num"])]
    if len(normal_actions) < count or len(forced_actions) < count:
        raise ValueError("A valid pair ended before its intervention step")

    raw_normal = np.asarray(
        [row["raw_openvla_action"] for row in normal_actions[:count]]
    )
    raw_forced = np.asarray(
        [row["raw_openvla_action"] for row in forced_actions[:count]]
    )
    policy_normal = np.asarray(
        [row["policy_libero_action"] for row in normal_actions[:count]]
    )
    policy_forced = np.asarray(
        [row["policy_libero_action"] for row in forced_actions[:count]]
    )
    raw_delta = _max_delta(raw_normal, raw_forced)
    policy_delta = _max_delta(policy_normal, policy_forced)
    if raw_delta > action_atol or policy_delta > action_atol:
        raise ValueError(
            "Normal/forced policy actions differ before intervention: "
            f"raw={raw_delta} policy={policy_delta}"
        )

    normal_sim = _load_npz(root, normal_result["sim_state_path"])
    forced_sim = _load_npz(root, forced_result["sim_state_path"])
    stateful_fields = (
        "time",
        "qpos",
        "qvel",
        "act",
        "ctrl",
        "mocap_pos",
        "mocap_quat",
        "userdata",
    )
    state_deltas: dict[str, float] = {}
    for field in stateful_fields:
        key = f"pre_{field}"
        if (key in normal_sim) != (key in forced_sim):
            raise ValueError(f"Paired simulator field presence differs: {key}")
        if key not in normal_sim:
            continue
        delta = _max_delta(
            normal_sim[key][:count], forced_sim[key][:count]
        )
        state_deltas[field] = delta
        if delta > state_atol:
            raise ValueError(
                "Normal/forced simulator state differs before intervention: "
                f"{field}={delta}"
            )
    qpos_delta = state_deltas.get("qpos", 0.0)
    qvel_delta = state_deltas.get("qvel", 0.0)

    normal_vision = _load_npz(root, normal_result["vision_path"])
    forced_vision = _load_npz(root, forced_result["vision_path"])
    rgb_equal = np.array_equal(
        normal_vision["model_input_rgb"][:count],
        forced_vision["model_input_rgb"][:count],
    )
    if not rgb_equal:
        raise ValueError("Normal/forced model-input RGB differs before intervention")
    return {
        "rgb_exact": rgb_equal,
        "max_raw_action_abs_delta": raw_delta,
        "max_policy_action_abs_delta": policy_delta,
        "max_qpos_abs_delta": qpos_delta,
        "max_qvel_abs_delta": qvel_delta,
        **{
            f"max_pre_{field}_abs_delta": delta
            for field, delta in state_deltas.items()
        },
    }


def _validate_actions(
    pair: dict[str, Any],
    normal_actions: list[dict[str, Any]],
    forced_actions: list[dict[str, Any]],
    *,
    action_atol: float,
) -> None:
    for row in normal_actions:
        policy = np.asarray(row["policy_libero_action"], dtype=np.float64)
        executed = np.asarray(row["executed_libero_action"], dtype=np.float64)
        if not np.allclose(policy, executed, rtol=0.0, atol=action_atol):
            raise ValueError(f"Normal action was modified: {pair['pair_id']}")
        intervention = row.get("intervention", {})
        if bool(intervention.get("forced_open_applied")):
            raise ValueError(f"Normal row marked as intervention: {pair['pair_id']}")
        if intervention.get("type", "none") != "none":
            raise ValueError(
                f"Normal row has a non-none intervention type: {pair['pair_id']}"
            )

    t_cmd = int(pair["forced_release"]["t_cmd"])
    t_detach = int(pair["forced_release"]["t_detach"])
    t_detach_confirmed = int(
        pair["forced_release"]["t_detach_confirmed"]
    )
    applied_steps: list[int] = []
    for row in forced_actions:
        step = int(row["step_in_episode"])
        policy = np.asarray(row["policy_libero_action"], dtype=np.float64)
        executed = np.asarray(row["executed_libero_action"], dtype=np.float64)
        intervention = row.get("intervention", {})
        applied = bool(intervention.get("forced_open_applied"))
        if intervention.get("type") != "forced_gripper_open":
            raise ValueError(
                f"Forced row lacks intervention identity: {pair['pair_id']} "
                f"step={step}"
            )
        should_apply = t_cmd <= step <= t_detach_confirmed
        if applied != should_apply:
            raise ValueError(
                f"Forced-open flag interval mismatch: {pair['pair_id']} step={step}"
            )
        if applied:
            applied_steps.append(step)
            if not only_gripper_was_overridden(
                policy, executed, atol=action_atol
            ):
                raise ValueError(
                    f"Forced release changed a non-gripper dimension: "
                    f"{pair['pair_id']} step={step}"
                )
        elif not np.allclose(policy, executed, rtol=0.0, atol=action_atol):
            raise ValueError(
                f"Forced action changed outside intervention: "
                f"{pair['pair_id']} step={step}"
            )
    if (
        not applied_steps
        or applied_steps[0] != t_cmd
        or applied_steps[-1] != t_detach_confirmed
    ):
        raise ValueError(f"Forced-open interval is incomplete: {pair['pair_id']}")


def validate_paired_release_run(
    run_dir: str | Path,
    *,
    min_valid_pairs_per_task: int | None = None,
    min_primary_pairs_per_task: int | None = None,
    require_complete: bool = True,
) -> dict[str, Any]:
    """Validate structure, pairing, determinism, intervention, and detach events."""

    root = Path(run_dir).expanduser().resolve()
    manifest = _json(root / "manifest.json")
    if manifest.get("schema_version") != PAIRED_SCHEMA_VERSION:
        raise ValueError("Unexpected paired-release dataset schema")
    collection_status = manifest.get("collection_status")
    if require_complete:
        if collection_status != "complete":
            raise ValueError("Paired collection is not marked complete")
        if not (root / "COLLECTION_COMPLETE").is_file():
            raise ValueError("COLLECTION_COMPLETE marker is missing")
    completed_summary: dict[str, Any] | None = None
    if collection_status == "complete":
        completed_summary = _json(root / "summary.json")
        if manifest.get("summary") != completed_summary:
            raise ValueError("manifest.summary and summary.json differ")
    pair_rows = _jsonl(root / "pair_results.jsonl")
    selected_task_ids = [int(value) for value in manifest["resolved_task_ids"]]
    if len(selected_task_ids) != len(set(selected_task_ids)):
        raise ValueError("resolved_task_ids contains duplicates")
    paired_cfg = manifest["config"]["paired_release"]
    configured_valid_target = int(
        paired_cfg["target_valid_pairs_per_task"]
    )
    configured_primary_target = int(
        paired_cfg.get("target_primary_pairs_per_task", 0)
    )
    required_valid = max(
        configured_valid_target,
        int(min_valid_pairs_per_task or 0),
    )
    required_primary = max(
        configured_primary_target,
        int(min_primary_pairs_per_task or 0),
    )
    max_attempts = int(manifest["config"]["env"]["num_trials_per_task"])

    rows_by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        task_id = int(row["task_id"])
        if task_id not in selected_task_ids:
            raise ValueError(f"Pair row has an unselected task_id: {task_id}")
        rows_by_task[task_id].append(row)

    per_task: dict[str, dict[str, Any]] = {}
    quota_summary_by_task: dict[str, dict[str, Any]] = {}
    for task_id in selected_task_ids:
        task_rows = sorted(
            rows_by_task.get(task_id, []),
            key=lambda row: int(row["task_episode_idx"]),
        )
        if not task_rows:
            raise ValueError(f"Task {task_id} has no pair candidates")
        indices = [int(row["task_episode_idx"]) for row in task_rows]
        if indices != list(range(len(task_rows))):
            raise ValueError(
                f"Task {task_id} candidate indices are not contiguous from zero"
            )
        if len(task_rows) > max_attempts:
            raise ValueError(
                f"Task {task_id} has {len(task_rows)} attempts; "
                f"maximum is {max_attempts}"
            )

        running_valid = 0
        running_primary = 0
        first_target_row: int | None = None
        for row_index, row in enumerate(task_rows):
            running_valid += int(bool(row.get("valid")))
            running_primary += int(bool(row.get("primary_analysis_eligible")))
            if (
                first_target_row is None
                and running_valid >= configured_valid_target
                and running_primary >= configured_primary_target
            ):
                first_target_row = row_index
        if first_target_row is not None and first_target_row != len(task_rows) - 1:
            raise ValueError(
                f"Task {task_id} continued after reaching its configured targets"
            )

        eligible_count = sum(bool(row.get("eligible")) for row in task_rows)
        valid_count = sum(bool(row.get("valid")) for row in task_rows)
        primary_count = sum(
            bool(row.get("primary_analysis_eligible")) for row in task_rows
        )
        if valid_count < required_valid:
            raise ValueError(
                f"Task {task_id} has {valid_count} valid pairs; "
                f"required at least {required_valid}"
            )
        if primary_count < required_primary:
            raise ValueError(
                f"Task {task_id} has {primary_count} primary-analysis pairs; "
                f"required at least {required_primary}"
            )
        quota_summary = {
            "task_id": task_id,
            "max_attempts": max_attempts,
            "target_valid_pairs": configured_valid_target,
            "target_primary_pairs": configured_primary_target,
            "attempts": len(task_rows),
            "attempts_remaining": max_attempts - len(task_rows),
            "eligible_pairs": eligible_count,
            "valid_pairs": valid_count,
            "invalid_pairs": eligible_count - valid_count,
            "ineligible_pairs": len(task_rows) - eligible_count,
            "primary_analysis_pairs": primary_count,
            "target_reached": (
                valid_count >= configured_valid_target
                and primary_count >= configured_primary_target
            ),
        }
        quota_summary_by_task[str(task_id)] = quota_summary
        per_task[str(task_id)] = {
            **quota_summary,
            "required_valid_pairs": required_valid,
            "required_primary_pairs": required_primary,
        }

    expected_candidates = len(pair_rows)
    pair_ids = [str(row["pair_id"]) for row in pair_rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate pair_id in pair_results.jsonl")

    eligible = [row for row in pair_rows if bool(row.get("eligible"))]
    ineligible = [row for row in pair_rows if not bool(row.get("eligible"))]
    expected_episodes = expected_candidates + len(eligible)
    structural = validate_extended_run(
        root,
        expected_schema_version=PAIRED_SCHEMA_VERSION,
        expected_episode_count=expected_episodes,
    )

    prompts = _episode_map(_jsonl(root / "prompt_records.jsonl"), "prompt")
    episodes = _episode_map(_jsonl(root / "episode_results.jsonl"), "result")
    actions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    expected_by_episode: dict[int, dict[str, Any]] = {}

    for pair in pair_rows:
        normal_num = int(pair["normal_episode_num"])
        if normal_num in expected_by_episode:
            raise ValueError(f"Episode reused by multiple pairs: {normal_num}")
        expected_by_episode[normal_num] = {
            "pair_id": pair["pair_id"],
            "condition": NORMAL_CONDITION,
            "pair_seed": pair["pair_seed"],
            "task_id": pair["task_id"],
            "task_episode_idx": pair["task_episode_idx"],
        }
        if pair.get("normal") != episodes.get(normal_num):
            raise ValueError(
                f"Pair normal result differs from episode result: {pair['pair_id']}"
            )
        normal_result = episodes[normal_num]
        expected_natural_release = bool(
            normal_result.get("t_cmd") is not None
            and normal_result.get("t_detach_confirmed") is not None
        )
        if bool(pair.get("normal_success")) != bool(normal_result.get("success")):
            raise ValueError(
                f"Pair normal_success differs from episode result: {pair['pair_id']}"
            )
        if (
            bool(pair.get("normal_natural_release_observed"))
            != expected_natural_release
        ):
            raise ValueError(
                "Pair normal release status differs from episode result: "
                f"{pair['pair_id']}"
            )
        if not pair.get("eligible"):
            if (
                pair.get("status") != "ineligible"
                or pair.get("valid") is not False
                or pair.get("forced_episode_num") is not None
                or not pair.get("skip_reason")
                or bool(pair.get("primary_analysis_eligible"))
            ):
                raise ValueError(f"Malformed ineligible pair: {pair['pair_id']}")
            continue
        forced_num = int(pair["forced_episode_num"])
        if forced_num in expected_by_episode:
            raise ValueError(f"Episode reused by multiple pairs: {forced_num}")
        expected_by_episode[forced_num] = {
            "pair_id": pair["pair_id"],
            "condition": FORCED_RELEASE_CONDITION,
            "pair_seed": pair["pair_seed"],
            "task_id": pair["task_id"],
            "task_episode_idx": pair["task_episode_idx"],
        }
        if pair.get("forced_release") != episodes.get(forced_num):
            raise ValueError(
                f"Pair forced result differs from episode result: {pair['pair_id']}"
            )
        valid = bool(pair.get("valid"))
        if valid != (pair.get("status") == "valid"):
            raise ValueError(f"Pair status/valid mismatch: {pair['pair_id']}")
        if valid and pair.get("invalid_reason") is not None:
            raise ValueError(f"Valid pair has invalid_reason: {pair['pair_id']}")
        if not valid and not pair.get("invalid_reason"):
            raise ValueError(f"Invalid pair lacks invalid_reason: {pair['pair_id']}")
        expected_primary = bool(
            valid
            and normal_result.get("success")
            and expected_natural_release
        )
        if bool(pair.get("primary_analysis_eligible")) != expected_primary:
            raise ValueError(
                f"Primary-analysis status mismatch: {pair['pair_id']}"
            )

    if set(expected_by_episode) != set(prompts) or set(prompts) != set(episodes):
        raise ValueError("Pair results do not cover every recorded episode exactly once")

    if completed_summary is not None:
        eligible_count = len(eligible)
        valid_count = sum(bool(row.get("valid")) for row in eligible)
        primary_count = sum(
            bool(row.get("primary_analysis_eligible")) for row in pair_rows
        )
        expected_global = {
            "max_attempts_per_task": max_attempts,
            "target_valid_pairs_per_task": configured_valid_target,
            "target_primary_pairs_per_task": configured_primary_target,
            "pair_candidates": len(pair_rows),
            "eligible_pairs": eligible_count,
            "valid_pairs": valid_count,
            "invalid_pairs": eligible_count - valid_count,
            "ineligible_pairs": len(ineligible),
            "primary_analysis_pairs": primary_count,
            "episodes": len(episodes),
            "valid_rate_among_eligible": (
                valid_count / eligible_count if eligible_count else 0.0
            ),
        }
        for field, expected_value in expected_global.items():
            if completed_summary.get(field) != expected_value:
                raise ValueError(
                    f"summary.{field} does not match the raw records"
                )
        if completed_summary.get("per_task") != quota_summary_by_task:
            raise ValueError("summary.per_task does not match pair_results.jsonl")

        expected_conditions: dict[str, dict[str, int | float]] = {}
        for condition in (NORMAL_CONDITION, FORCED_RELEASE_CONDITION):
            condition_rows = [
                row for row in episodes.values() if row["condition"] == condition
            ]
            successes = sum(bool(row.get("success")) for row in condition_rows)
            count = len(condition_rows)
            expected_conditions[condition] = {
                "episodes": count,
                "successes": successes,
                "failures": count - successes,
                "success_rate": successes / count if count else 0.0,
            }
        if completed_summary.get("conditions") != expected_conditions:
            raise ValueError(
                "summary.conditions does not match episode_results.jsonl"
            )
    for episode_num, expected in expected_by_episode.items():
        _assert_episode_identity(prompts[episode_num], expected, source="prompt")
        _assert_episode_identity(episodes[episode_num], expected, source="result")

    for row in _iter_jsonl(root / "action_records.jsonl"):
        episode_num = int(row["episode_num"])
        _assert_episode_identity(row, expected_by_episode[episode_num], source="action")
        actions[episode_num].append(row)
    for rows in actions.values():
        steps = [int(row["step_in_episode"]) for row in rows]
        if steps != list(range(len(rows))):
            raise ValueError("Non-contiguous action steps in paired run")

    for filename in ("trajectory_records.jsonl", "policy_uncertainty.jsonl"):
        for row in _iter_jsonl(root / filename):
            episode_num = int(row["episode_num"])
            _assert_episode_identity(
                row, expected_by_episode[episode_num], source=filename
            )
    activation_index = (
        root
        / "sae_activations"
        / "post_mlp_residual"
        / "activation_index.jsonl"
    )
    for row in _iter_jsonl(activation_index):
        episode_num = int(row["episode_num"])
        _assert_episode_identity(
            row, expected_by_episode[episode_num], source="activation_index"
        )

    valid_pairs = [row for row in eligible if bool(row.get("valid"))]
    detach_checks: dict[tuple[int, int], dict[str, Any]] = {}
    confirmation_checks: dict[tuple[int, int], dict[str, Any]] = {}
    obs_checks: dict[tuple[int, int], dict[str, Any]] = {}
    contact_checks: dict[tuple[int, int], dict[str, Any]] = {}
    prefix_metrics: dict[str, dict[str, float | bool]] = {}
    action_atol = float(manifest["config"]["paired_release"]["action_atol"])
    state_atol = float(manifest["config"]["paired_release"]["state_atol"])
    max_force_steps = int(
        manifest["config"]["paired_release"]["max_force_open_steps"]
    )
    stable_detach_steps = int(
        manifest["config"]["paired_release"]["stable_detach_steps"]
    )

    for pair in valid_pairs:
        normal_num = int(pair["normal_episode_num"])
        forced_num = int(pair["forced_episode_num"])
        normal_prompt, forced_prompt = prompts[normal_num], prompts[forced_num]
        if (
            normal_prompt["initial_state_sha256"]
            != forced_prompt["initial_state_sha256"]
            or normal_prompt["initial_state_sha256"]
            != pair["initial_state_sha256"]
        ):
            raise ValueError(f"Paired initial states differ: {pair['pair_id']}")
        normal_result, forced_result = episodes[normal_num], episodes[forced_num]
        for field in (
            "warm_start_sim_state_sha256",
            "warm_start_source_rgb_sha256",
        ):
            if (
                not normal_result.get(field)
                or normal_result.get(field) != forced_result.get(field)
            ):
                raise ValueError(
                    f"Paired warm-start evidence differs: "
                    f"{pair['pair_id']} {field}"
                )
        t_cmd = pair["forced_release"].get("t_cmd")
        t_detach = pair["forced_release"].get("t_detach")
        t_detach_confirmed = pair["forced_release"].get(
            "t_detach_confirmed"
        )
        trigger_step = pair["normal"]["trigger"].get("trigger_step")
        if (
            t_cmd is None
            or t_detach is None
            or t_detach_confirmed is None
            or trigger_step is None
        ):
            raise ValueError(f"Valid pair lacks release timestamps: {pair['pair_id']}")
        t_cmd = int(t_cmd)
        t_detach = int(t_detach)
        t_detach_confirmed = int(t_detach_confirmed)
        trigger_step = int(trigger_step)
        if (
            t_cmd != trigger_step
            or t_detach < t_cmd
            or t_detach_confirmed < t_detach
        ):
            raise ValueError(f"Invalid release timestamp order: {pair['pair_id']}")
        if t_detach_confirmed - t_cmd + 1 > max_force_steps:
            raise ValueError(
                f"Forced-open interval exceeds configured limit: {pair['pair_id']}"
            )
        expected_confirmation = t_detach + stable_detach_steps - 1
        if t_detach_confirmed != expected_confirmation:
            raise ValueError(
                f"Detach confirmation window mismatch: {pair['pair_id']}"
            )
        t_obs = pair["forced_release"].get("t_obs")
        if t_obs is not None and int(t_obs) != t_detach + 1:
            raise ValueError(
                f"t_obs is not the first post-detach observation: {pair['pair_id']}"
            )

        _validate_actions(
            pair,
            actions[normal_num],
            actions[forced_num],
            action_atol=action_atol,
        )
        prefix_metrics[pair["pair_id"]] = _validate_prefix(
            root,
            normal_result,
            forced_result,
            through_step=t_cmd,
            action_rows=actions,
            action_atol=action_atol,
            state_atol=state_atol,
        )
        detach_checks[(forced_num, t_detach)] = pair
        for confirmation_step in range(t_detach, t_detach_confirmed + 1):
            confirmation_checks[(forced_num, confirmation_step)] = pair
        if t_obs is not None:
            obs_checks[(forced_num, int(t_obs))] = pair
        contact_step = pair["forced_release"].get(
            "t_post_detach_destination_contact"
        )
        if contact_step is not None:
            contact_checks[(forced_num, int(contact_step))] = pair

    seen_detach: set[str] = set()
    seen_confirmation: set[tuple[int, int]] = set()
    seen_obs: set[str] = set()
    seen_contact: set[str] = set()
    for row in _iter_jsonl(root / "trajectory_records.jsonl"):
        key = (int(row["episode_num"]), int(row["step_in_episode"]))
        if key in detach_checks:
            pair = detach_checks[key]
            target = pair["target_object"]
            if not target_grasped(row["pre"], target) or target_grasped(
                row["post"], target
            ):
                raise ValueError(f"t_detach is not grasp True->False: {pair['pair_id']}")
            seen_detach.add(pair["pair_id"])
        if key in confirmation_checks:
            pair = confirmation_checks[key]
            if target_grasped(row["post"], pair["target_object"]):
                raise ValueError(
                    f"Object was regrasped inside detach confirmation window: "
                    f"{pair['pair_id']}"
                )
            seen_confirmation.add(key)
        if key in obs_checks:
            pair = obs_checks[key]
            if target_grasped(row["pre"], pair["target_object"]):
                raise ValueError(f"Object is still grasped at t_obs: {pair['pair_id']}")
            seen_obs.add(pair["pair_id"])
        if key in contact_checks:
            pair = contact_checks[key]
            if not target_destination_contact(
                row["post"],
                pair["target_object"],
                pair.get("goal_destination"),
            ):
                raise ValueError(
                    f"No target-destination contact at recorded time: "
                    f"{pair['pair_id']}"
                )
            seen_contact.add(pair["pair_id"])

    if seen_detach != {row["pair_id"] for row in valid_pairs}:
        raise ValueError("Not every valid pair has a verified detach transition")
    if seen_confirmation != set(confirmation_checks):
        raise ValueError(
            "Not every valid pair has a stable verified detach window"
        )
    if seen_obs != {row["pair_id"] for row in valid_pairs if row["forced_release"].get("t_obs") is not None}:
        raise ValueError("Not every recorded t_obs was verified")
    if seen_contact != {
        row["pair_id"]
        for row in valid_pairs
        if row["forced_release"].get("t_post_detach_destination_contact") is not None
    }:
        raise ValueError("Not every recorded fixture-contact time was verified")

    return {
        **structural,
        "pair_candidates": len(pair_rows),
        "eligible_pairs": len(eligible),
        "valid_pairs": len(valid_pairs),
        "invalid_pairs": len(eligible) - len(valid_pairs),
        "ineligible_pairs": len(ineligible),
        "per_task": per_task,
        "prefix_metrics": prefix_metrics,
    }
