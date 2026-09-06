import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.paired_validate import (
    validate_paired_release_run,
)
from event_sae.openvla.extended_collection.writer import ExtendedRunWriter


PAIR_ID = "libero_spatial-task00-trial000-seed0"


def _state(grasped: bool, *, goal_satisfied: bool = False):
    return {
        "robot": {
            "joint_position": [0.0],
            "joint_velocity": [0.0],
            "joint_torque_command": [0.0],
            "eef_position": [0.0, 0.0, 0.1],
            "eef_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "eef_linear_velocity": [0.0, 0.0, 0.0],
            "eef_angular_velocity": [0.0, 0.0, 0.0],
            "gripper_qpos": [0.0, 0.0],
            "gripper_qvel": [0.0, 0.0],
        },
        "objects": {
            "bowl": {
                "position_world": [0.0, 0.0, 0.1],
                "quaternion_wxyz_world": [1.0, 0.0, 0.0, 0.0],
                "linear_velocity_world": [0.0, 0.0, 0.0],
                "angular_velocity_world": [0.0, 0.0, 0.0],
            }
        },
        "fixtures": {},
        "contact_and_grasp": {
            "contacts": [],
            "grasped_objects": {"bowl": grasped},
        },
        "goals": {
            "goal_predicates": [{"satisfied": goal_satisfied, "error": None}],
            "all_goal_predicates_satisfied": goal_satisfied,
        },
    }


def _uncertainty():
    dimensions = []
    for index in range(7):
        dimensions.append(
            {
                "action_dimension": index,
                "selected_token_is_action_token": True,
                "selected_token_probability": 0.8,
                "selected_token_conditional_probability": 0.9,
                "full_next_token": {
                    "entropy_nats": 1.0,
                    "normalized_entropy": 0.1,
                    "top1_probability": 0.8,
                    "top1_top2_margin": 0.7,
                },
                "conditional_action_token": {
                    "probability_mass_in_full_vocabulary": 0.99,
                    "entropy_nats": 0.5,
                    "normalized_entropy": 0.1,
                    "top1_probability": 0.9,
                    "top1_top2_margin": 0.8,
                },
            }
        )
    return {
        "action_token_ids": [32000] * 7,
        "uncertainty": {"per_action_dimension": dimensions},
    }


def _build_run(tmp_path: Path, *, primary: bool = True) -> Path:
    run_dir = tmp_path / "paired"
    initial = np.asarray([1.0, 2.0], dtype=np.float32)
    vectors = {
        name: np.asarray([0.0], dtype=np.float32)
        for name in ("qpos", "qvel", "qacc", "ctrl")
    }
    policy_action = np.asarray([0.1, 0, 0, 0, 0, 0, 1.0])
    manifest = {
        "schema_version": "extended_openvla_libero_paired_release_v4",
        "collection_status": "complete",
        "activation_stream": {"layer": 31, "forwards_per_policy_step": 7},
        "policy_uncertainty": {"full_logits_stored": False},
        "resolved_task_ids": [0],
        "config": {
            "env": {"num_trials_per_task": 1},
            "output": {
                "save_model_input_rgb": True,
                "save_video": False,
            },
            "paired_release": {
                "action_atol": 1e-6,
                "state_atol": 1e-6,
                "max_force_open_steps": 20,
                "stable_detach_steps": 1,
                "forced_gripper_value": -1.0,
                "target_valid_pairs_per_task": 1,
                "target_primary_pairs_per_task": int(primary),
            },
        },
    }
    with ExtendedRunWriter(run_dir, enable_pair_results=True) as writer:
        writer.write_manifest(manifest)
        episode_results = {}
        for episode_num, condition in ((1, "normal"), (2, "forced_release")):
            common_identity = {
                "episode_num": episode_num,
                "pair_id": PAIR_ID,
                "condition": condition,
                "pair_seed": 0,
                "task_id": 0,
                "task_episode_idx": 0,
            }
            prompt = writer.begin_episode(
                prompt_record={**common_identity, "task_description": "move bowl"},
                initial_state=initial,
            )
            for step in range(4):
                applied = condition == "forced_release" and step == 1
                step_policy_action = policy_action.copy()
                executed = step_policy_action.copy()
                if applied:
                    executed[-1] = -1.0
                if condition == "forced_release":
                    pre_grasped = step == 1
                    post_grasped = False
                else:
                    pre_grasped = step >= 1
                    post_grasped = step >= 1
                normal_goal = condition == "normal" and primary and step >= 3
                pre = _state(
                    grasped=pre_grasped,
                    goal_satisfied=normal_goal and step >= 3,
                )
                post = _state(
                    grasped=post_grasped,
                    goal_satisfied=normal_goal,
                )
                writer.write_step(
                    common={**common_identity, "step_in_episode": step},
                    pre_json=pre,
                    post_json=post,
                    pre_vectors=vectors,
                    post_vectors=vectors,
                    raw_action=np.zeros(7),
                    policy_action=step_policy_action,
                    executed_action=executed,
                    intervention={
                        "type": (
                            "forced_gripper_open"
                            if condition == "forced_release"
                            else "none"
                        ),
                        "forced_open_applied": applied,
                    },
                    policy=_uncertainty(),
                    model_input_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
                    reward=0.0,
                    done=(condition == "normal" and primary and step == 3),
                    info={},
                )
            result = {
                **common_identity,
                "success": condition == "normal" and primary,
                "success_step": (
                    3 if condition == "normal" and primary else None
                ),
                "initial_state_sha256": prompt["initial_state_sha256"],
                "trigger": {"trigger_step": 1, "initial_object_z": 0.0},
                "t_cmd": 1 if condition == "forced_release" else None,
                "t_detach": 1 if condition == "forced_release" else None,
                "t_detach_confirmed": (1 if condition == "forced_release" else None),
                "t_obs": None,
                "t_post_detach_destination_contact": None,
                "invalid_reason": None,
                "warm_start_sim_state_sha256": "same-sim",
                "warm_start_source_rgb_sha256": "same-rgb",
            }
            episode_results[condition] = writer.finish_episode(
                result=result,
                compress_npz=True,
            )

        writer.write_pair_result(
            {
                "pair_id": PAIR_ID,
                "status": "valid",
                "eligible": True,
                "valid": True,
                "skip_reason": None,
                "invalid_reason": None,
                "task_id": 0,
                "task_episode_idx": 0,
                "pair_seed": 0,
                "target_object": "bowl",
                "initial_state_sha256": next(
                    json.loads(line)["initial_state_sha256"]
                    for line in (run_dir / "prompt_records.jsonl")
                    .read_text()
                    .splitlines()
                ),
                "normal_episode_num": 1,
                "forced_episode_num": 2,
                "normal_success": primary,
                "normal_natural_release_observed": False,
                "primary_analysis_eligible": primary,
                "normal": episode_results["normal"],
                "forced_release": episode_results["forced_release"],
                "prefix_comparison": {"rgb_exact": True},
            }
        )

    activation_dir = run_dir / "sae_activations" / "post_mlp_residual"
    activation_dir.mkdir(parents=True)
    shard = "layer_31_shard_000000.pt"
    torch.save(torch.zeros((56, 4096)), activation_dir / shard)
    records = []
    offset = 0
    for episode_num, condition in ((1, "normal"), (2, "forced_release")):
        for step in range(4):
            for _ in range(7):
                records.append(
                    {
                        "layer_idx": 31,
                        "episode_num": episode_num,
                        "pair_id": PAIR_ID,
                        "condition": condition,
                        "pair_seed": 0,
                        "task_id": 0,
                        "task_episode_idx": 0,
                        "step_in_episode": step,
                        "shard_path": shard,
                        "row_start": offset,
                        "row_end": offset + 1,
                        "global_forward_idx": offset + 1,
                    }
                )
                offset += 1
    (activation_dir / "activation_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records),
        encoding="utf-8",
    )
    summary = {
        "run_dir": str(run_dir),
        "max_attempts_per_task": 1,
        "target_valid_pairs_per_task": 1,
        "target_primary_pairs_per_task": int(primary),
        "pair_candidates": 1,
        "eligible_pairs": 1,
        "valid_pairs": 1,
        "invalid_pairs": 0,
        "ineligible_pairs": 0,
        "primary_analysis_pairs": int(primary),
        "per_task": {
            "0": {
                "task_id": 0,
                "max_attempts": 1,
                "target_valid_pairs": 1,
                "target_primary_pairs": int(primary),
                "attempts": 1,
                "attempts_remaining": 0,
                "eligible_pairs": 1,
                "valid_pairs": 1,
                "invalid_pairs": 0,
                "ineligible_pairs": 0,
                "primary_analysis_pairs": int(primary),
                "target_reached": True,
            }
        },
        "episodes": 2,
        "conditions": {
            "normal": {
                "episodes": 1,
                "successes": int(primary),
                "failures": int(not primary),
                "success_rate": float(primary),
            },
            "forced_release": {
                "episodes": 1,
                "successes": 0,
                "failures": 1,
                "success_rate": 0.0,
            },
        },
        "valid_rate_among_eligible": 1.0,
    }
    manifest["summary"] = summary
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    (run_dir / "COLLECTION_COMPLETE").write_text("done\n", encoding="utf-8")
    return run_dir


def test_paired_validator_checks_matched_prefix_and_gripper_only(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    summary = validate_paired_release_run(run_dir)
    assert summary["pair_candidates"] == 1
    assert summary["valid_pairs"] == 1
    assert summary["ineligible_pairs"] == 0
    assert summary["prefix_metrics"][PAIR_ID]["rgb_exact"] is True


def test_paired_validator_accepts_successful_control_as_primary(tmp_path: Path):
    run_dir = _build_run(tmp_path, primary=True)
    summary = validate_paired_release_run(run_dir)
    assert summary["per_task"]["0"]["primary_analysis_pairs"] == 1


def test_paired_validator_rejects_success_without_trajectory_done(tmp_path: Path):
    run_dir = _build_run(tmp_path, primary=True)
    path = run_dir / "trajectory_records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["condition"] == "normal":
            row["done"] = False
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="success differs from trajectory done"):
        validate_paired_release_run(run_dir)


def test_paired_validator_rejects_wrong_success_step(tmp_path: Path):
    run_dir = _build_run(tmp_path, primary=True)
    result_path = run_dir / "episode_results.jsonl"
    results = [json.loads(line) for line in result_path.read_text().splitlines()]
    normal = next(row for row in results if row["condition"] == "normal")
    normal["success_step"] = 2
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
    )

    pair_path = run_dir / "pair_results.jsonl"
    pair = json.loads(pair_path.read_text())
    pair["normal"]["success_step"] = 2
    pair_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="success_step differs from first"):
        validate_paired_release_run(run_dir)


def test_paired_validator_rejects_normal_steps_after_success(tmp_path: Path):
    run_dir = _build_run(tmp_path, primary=True)
    trajectory_path = run_dir / "trajectory_records.jsonl"
    rows = [json.loads(line) for line in trajectory_path.read_text().splitlines()]
    normal_step_two = next(
        row
        for row in rows
        if row["condition"] == "normal" and row["step_in_episode"] == 2
    )
    normal_step_two["done"] = True
    trajectory_path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    result_path = run_dir / "episode_results.jsonl"
    results = [json.loads(line) for line in result_path.read_text().splitlines()]
    normal = next(row for row in results if row["condition"] == "normal")
    normal["success_step"] = 2
    result_path.write_text(
        "".join(json.dumps(row) + "\n" for row in results), encoding="utf-8"
    )

    pair_path = run_dir / "pair_results.jsonl"
    pair = json.loads(pair_path.read_text())
    pair["normal"]["success_step"] = 2
    pair_path.write_text(json.dumps(pair) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="continued after LIBERO done"):
        validate_paired_release_run(run_dir)


def test_paired_validator_can_run_before_completion_marker(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    (run_dir / "COLLECTION_COMPLETE").unlink()
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["collection_status"] = "in_progress"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not marked complete"):
        validate_paired_release_run(run_dir)
    summary = validate_paired_release_run(run_dir, require_complete=False)
    assert summary["valid_pairs"] == 1


def test_paired_validator_rejects_non_gripper_action_change(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    path = run_dir / "action_records.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    forced = next(
        row
        for row in rows
        if row["condition"] == "forced_release" and row["step_in_episode"] == 1
    )
    forced["executed_libero_action"][0] += 0.1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="non-gripper"):
        validate_paired_release_run(run_dir)


def test_paired_validator_accepts_adaptive_early_stop(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["env"]["num_trials_per_task"] = 3
    manifest["summary"]["max_attempts_per_task"] = 3
    manifest["summary"]["per_task"]["0"]["max_attempts"] = 3
    manifest["summary"]["per_task"]["0"]["attempts_remaining"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "summary.json").write_text(
        json.dumps(manifest["summary"]), encoding="utf-8"
    )

    summary = validate_paired_release_run(run_dir)
    assert summary["pair_candidates"] == 1
    assert summary["per_task"]["0"]["attempts"] == 1
    assert summary["per_task"]["0"]["valid_pairs"] == 1


def test_paired_validator_rejects_missing_selected_task(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resolved_task_ids"] = [0, 1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Task 1 has no pair candidates"):
        validate_paired_release_run(run_dir)


def test_paired_validator_override_cannot_weaken_manifest_target(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["env"]["num_trials_per_task"] = 3
    manifest["config"]["paired_release"]["target_valid_pairs_per_task"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="required at least 2"):
        validate_paired_release_run(
            run_dir,
            min_valid_pairs_per_task=1,
        )


def test_paired_validator_rejects_noncontiguous_task_attempts(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    pair_path = run_dir / "pair_results.jsonl"
    row = json.loads(pair_path.read_text())
    row["task_episode_idx"] = 1
    pair_path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not contiguous"):
        validate_paired_release_run(run_dir)


def test_paired_validator_rejects_primary_target_shortfall(tmp_path: Path):
    run_dir = _build_run(tmp_path, primary=False)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["paired_release"]["target_primary_pairs_per_task"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="primary-analysis pairs"):
        validate_paired_release_run(run_dir)


def test_paired_validator_rejects_rows_after_target(tmp_path: Path):
    run_dir = _build_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["env"]["num_trials_per_task"] = 3
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pair_path = run_dir / "pair_results.jsonl"
    rows = [json.loads(line) for line in pair_path.read_text().splitlines()]
    extra = dict(rows[0])
    extra["pair_id"] = PAIR_ID + "-extra"
    extra["task_episode_idx"] = 1
    pair_path.write_text(
        "".join(json.dumps(row) + "\n" for row in [*rows, extra]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continued after reaching"):
        validate_paired_release_run(run_dir)


def test_paired_validator_rejects_one_task_shortfall_despite_global_total(
    tmp_path: Path,
):
    run_dir = _build_run(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["resolved_task_ids"] = [0, 1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    pair_path = run_dir / "pair_results.jsonl"
    task_zero = json.loads(pair_path.read_text())
    task_one = dict(task_zero)
    task_one.update(
        {
            "pair_id": "libero_spatial-task01-trial000-seed10000",
            "task_id": 1,
            "status": "ineligible",
            "eligible": False,
            "valid": False,
            "skip_reason": "normal_no_stable_grasp_and_lift",
            "normal_episode_num": 3,
            "forced_episode_num": None,
            "primary_analysis_eligible": False,
        }
    )
    pair_path.write_text(
        json.dumps(task_zero) + "\n" + json.dumps(task_one) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Task 1 has 0 valid pairs"):
        validate_paired_release_run(run_dir)


def test_paired_validator_rejects_summary_not_backed_by_raw_records(
    tmp_path: Path,
):
    run_dir = _build_run(tmp_path)
    summary_path = run_dir / "summary.json"
    manifest_path = run_dir / "manifest.json"
    summary = json.loads(summary_path.read_text())
    summary["valid_pairs"] = 2
    manifest = json.loads(manifest_path.read_text())
    manifest["summary"] = summary
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="summary.valid_pairs"):
        validate_paired_release_run(run_dir)
