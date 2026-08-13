import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.validate import validate_extended_run
from event_sae.openvla.extended_collection.writer import ExtendedRunWriter


def rich_state():
    return {
        "robot": {
            "joint_position": [0.0],
            "joint_velocity": [0.0],
            "joint_torque_command": [0.0],
            "eef_position": [0.0, 0.0, 0.0],
            "eef_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            "eef_linear_velocity": [0.0, 0.0, 0.0],
            "eef_angular_velocity": [0.0, 0.0, 0.0],
            "gripper_qpos": [0.0, 0.0],
            "gripper_qvel": [0.0, 0.0],
        },
        "objects": {
            "bowl": {
                "position_world": [0.0, 0.0, 0.0],
                "quaternion_wxyz_world": [1.0, 0.0, 0.0, 0.0],
                "linear_velocity_world": [0.0, 0.0, 0.0],
                "angular_velocity_world": [0.0, 0.0, 0.0],
            }
        },
        "fixtures": {},
        "contact_and_grasp": {
            "contacts": [],
            "grasped_objects": {"bowl": False},
        },
        "goals": {
            "goal_predicates": [{"satisfied": False, "error": None}],
        },
    }


def uncertainty():
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


def test_validator_connects_episode_step_rgb_sim_and_activation(tmp_path: Path):
    run_dir = tmp_path / "run"
    with ExtendedRunWriter(run_dir) as writer:
        writer.write_manifest(
            {
                "schema_version": "extended_openvla_libero_v1",
                "activation_stream": {"layer": 31, "forwards_per_policy_step": 7},
                "policy_uncertainty": {"full_logits_stored": False},
                "resolved_task_ids": [0],
                "config": {
                    "env": {"num_trials_per_task": 1},
                    "output": {"save_model_input_rgb": True, "save_video": False},
                },
            }
        )
        prompt = writer.begin_episode(
            prompt_record={"episode_num": 1, "task_id": 0, "task_episode_idx": 0},
            initial_state=np.asarray([1.0, 2.0], dtype=np.float32),
        )
        vectors = {
            name: np.asarray([0.0], dtype=np.float32)
            for name in ("qpos", "qvel", "qacc", "ctrl")
        }
        writer.write_step(
            common={
                "episode_num": 1,
                "task_id": 0,
                "task_episode_idx": 0,
                "step_in_episode": 0,
            },
            pre_json=rich_state(),
            post_json=rich_state(),
            pre_vectors=vectors,
            post_vectors=vectors,
            raw_action=np.zeros(7),
            executed_action=np.zeros(7),
            policy=uncertainty(),
            model_input_rgb=np.zeros((224, 224, 3), dtype=np.uint8),
            reward=0.0,
            done=True,
            info={},
        )
        writer.finish_episode(
            result={
                "episode_num": 1,
                "task_id": 0,
                "success": True,
                "initial_state_sha256": prompt["initial_state_sha256"],
            },
            compress_npz=True,
        )

    activation_dir = run_dir / "sae_activations" / "post_mlp_residual"
    activation_dir.mkdir(parents=True)
    shard_name = "layer_31_shard_000000.pt"
    torch.save(torch.zeros((7, 4096), dtype=torch.float32), activation_dir / shard_name)
    (activation_dir / "activation_index.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "layer_idx": 31,
                    "episode_num": 1,
                    "step_in_episode": 0,
                    "shard_path": shard_name,
                    "row_start": index,
                    "row_end": index + 1,
                    "global_forward_idx": index + 1,
                }
            )
            + "\n"
            for index in range(7)
        ),
        encoding="utf-8",
    )

    summary = validate_extended_run(run_dir)
    assert summary == {
        "schema_version": "extended_openvla_libero_v1",
        "episodes": 1,
        "policy_steps": 1,
        "activation_index_records": 7,
        "activation_shards": 1,
        "successes": 1,
    }
