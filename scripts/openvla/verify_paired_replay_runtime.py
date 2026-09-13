#!/usr/bin/env python3
"""Verify paired LIBERO state replay without loading OpenVLA or collecting data."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.config import (  # noqa: E402
    load_paired_release_config,
)
from event_sae.openvla.extended_collection.paired_runner import (  # noqa: E402
    RobosuitePythonCheckpoint,
    _capture_mujoco_integration_checkpoint,
    _max_abs_delta,
    _pair_seed,
    _raw_env,
    _reset_to_initial_state,
    _restore_mujoco_integration_checkpoint,
    _robosuite_python_state_sha256,
)
from event_sae.openvla.extended_collection.runner import _make_env  # noqa: E402


@dataclass(frozen=True)
class ReplaySample:
    integration_state: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    control_state_sha256: str
    observation: dict[str, np.ndarray]
    reward: float | None
    done: bool | None


def _actions(count: int) -> list[np.ndarray]:
    """Small deterministic commands that exercise OSC and both gripper signs."""

    pattern = (
        (0.05, 0.00, 0.00, 0.00, 0.00, 0.00, -1.0),
        (0.00, 0.05, 0.00, 0.00, 0.00, 0.00, -1.0),
        (0.00, 0.00, 0.05, 0.02, 0.00, 0.00, -1.0),
        (-0.05, 0.00, 0.00, 0.00, 0.02, 0.00, 1.0),
        (0.00, -0.05, 0.00, 0.00, 0.00, 0.02, 1.0),
        (0.00, 0.00, -0.05, -0.02, 0.00, 0.00, 1.0),
    )
    return [
        np.asarray(pattern[index % len(pattern)], dtype=np.float64)
        for index in range(count)
    ]


def _copy_observation(obs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        str(name): np.asarray(value).copy()
        for name, value in obs.items()
        if isinstance(value, (np.ndarray, np.generic, int, float, bool))
    }


def _control_state_sha256(checkpoint: RobosuitePythonCheckpoint) -> str:
    """Hash controller, gripper, robot buffers, and RNG without rendered images."""

    control_only = RobosuitePythonCheckpoint(
        robots=checkpoint.robots,
        observables={},
        obs_cache={},
        rng_state=checkpoint.rng_state,
        python_random_state=checkpoint.python_random_state,
        numpy_random_state=checkpoint.numpy_random_state,
        torch_cpu_rng_state=checkpoint.torch_cpu_rng_state,
        torch_cuda_rng_states=checkpoint.torch_cuda_rng_states,
    )
    return _robosuite_python_state_sha256(control_only)


def _sample(
    env, obs, *, reward: float | None = None, done: bool | None = None
) -> ReplaySample:
    checkpoint = _capture_mujoco_integration_checkpoint(
        env,
        include_robosuite_python_state=True,
    )
    raw = _raw_env(env)
    return ReplaySample(
        integration_state=checkpoint.state.copy(),
        qpos=np.asarray(raw.sim.data.qpos).copy(),
        qvel=np.asarray(raw.sim.data.qvel).copy(),
        control_state_sha256=_control_state_sha256(checkpoint.robosuite_python_state),
        observation=_copy_observation(obs),
        reward=reward,
        done=done,
    )


def _numeric_observation_delta(
    reference: dict[str, np.ndarray], current: dict[str, np.ndarray]
) -> float:
    if set(reference) != set(current):
        return float("inf")
    deltas = []
    for name in reference:
        if "image" in name or "depth" in name or "segmentation" in name:
            continue
        left = reference[name]
        right = current[name]
        if np.issubdtype(left.dtype, np.number) or left.dtype == np.bool_:
            deltas.append(_max_abs_delta(left, right))
    return max(deltas, default=0.0)


def _rgb_exact(
    reference: dict[str, np.ndarray], current: dict[str, np.ndarray]
) -> bool:
    name = "agentview_image"
    return (
        name in reference
        and name in current
        and np.array_equal(reference[name], current[name])
    )


def _compare_sample(
    label: str,
    reference: ReplaySample,
    current: ReplaySample,
    *,
    state_atol: float,
) -> dict[str, Any]:
    deltas = {
        "integration_state": _max_abs_delta(
            reference.integration_state, current.integration_state
        ),
        "qpos": _max_abs_delta(reference.qpos, current.qpos),
        "qvel": _max_abs_delta(reference.qvel, current.qvel),
        "numeric_observation": _numeric_observation_delta(
            reference.observation, current.observation
        ),
    }
    if not np.array_equal(
        reference.integration_state, current.integration_state, equal_nan=True
    ):
        raise RuntimeError(
            f"Paired replay diverged at {label}: integration_state is not "
            f"bit-exact (max_abs_delta={deltas['integration_state']})"
        )
    for name, delta in deltas.items():
        if not np.isfinite(delta) or delta > state_atol:
            raise RuntimeError(
                f"Paired replay diverged at {label}: {name} delta={delta} "
                f"> state_atol={state_atol}"
            )
    if reference.reward != current.reward or reference.done != current.done:
        raise RuntimeError(
            f"Paired replay outcome diverged at {label}: "
            f"normal=({reference.reward}, {reference.done}) "
            f"replay=({current.reward}, {current.done})"
        )
    if reference.control_state_sha256 != current.control_state_sha256:
        raise RuntimeError(f"Paired replay control state diverged at {label}")
    return {
        **deltas,
        "control_state_exact": True,
        "agentview_rgb_exact": _rgb_exact(reference.observation, current.observation),
    }


def verify_runtime(args: argparse.Namespace) -> dict[str, Any]:
    from libero.libero import benchmark

    cfg = load_paired_release_config(args.config, {})
    suite_cls = benchmark.get_benchmark_dict()[cfg.env.task_suite_name]
    suite = suite_cls()
    if not 0 <= args.task_id < suite.n_tasks:
        raise ValueError(f"task-id must be in [0, {suite.n_tasks})")
    initial_states = suite.get_task_init_states(args.task_id)
    if not 0 <= args.trial_index < len(initial_states):
        raise ValueError(
            f"trial-index must be in [0, {len(initial_states)}) for task {args.task_id}"
        )
    task = suite.get_task(args.task_id)
    env, _, _ = _make_env(task, cfg.env.resolution, cfg.env.seed)
    pair_seed = _pair_seed(cfg.env.seed, args.task_id, args.trial_index)
    initial_state = np.asarray(initial_states[args.trial_index]).copy()
    actions = _actions(args.steps)

    try:
        normal_obs = _reset_to_initial_state(
            env, initial_state, pair_seed, cfg.env.num_steps_wait
        )
        warm_start = _capture_mujoco_integration_checkpoint(
            env,
            include_model_fingerprint=True,
            include_robosuite_python_state=True,
        )
        normal_samples = [_sample(env, normal_obs)]
        for action in actions:
            normal_obs, reward, done, _ = env.step(action)
            normal_samples.append(
                _sample(env, normal_obs, reward=float(reward), done=bool(done))
            )
            if done:
                raise RuntimeError(
                    "Runtime replay action pattern ended the normal episode"
                )

        _reset_to_initial_state(env, initial_state, pair_seed, cfg.env.num_steps_wait)
        (
            replay_obs,
            _,
            pre_forward_delta,
            post_forward_delta,
            post_python_state_delta,
        ) = _restore_mujoco_integration_checkpoint(
            env,
            warm_start,
            state_atol=args.state_atol,
        )
        comparisons = [
            _compare_sample(
                "warm_start",
                normal_samples[0],
                _sample(env, replay_obs),
                state_atol=args.state_atol,
            )
        ]
        for index, action in enumerate(actions, start=1):
            replay_obs, reward, done, _ = env.step(action)
            comparisons.append(
                _compare_sample(
                    f"post_step_{index}",
                    normal_samples[index],
                    _sample(
                        env,
                        replay_obs,
                        reward=float(reward),
                        done=bool(done),
                    ),
                    state_atol=args.state_atol,
                )
            )

        return {
            "task_suite": cfg.env.task_suite_name,
            "task_id": args.task_id,
            "trial_index": args.trial_index,
            "pair_seed": pair_seed,
            "steps": args.steps,
            "state_atol": args.state_atol,
            "restore_deltas": {
                "pre_forward": pre_forward_delta,
                "post_forward": post_forward_delta,
                "post_python_state": post_python_state_delta,
            },
            "max_deltas": {
                name: max(item[name] for item in comparisons)
                for name in (
                    "integration_state",
                    "qpos",
                    "qvel",
                    "numeric_observation",
                )
            },
            "control_state_exact": all(
                item["control_state_exact"] for item in comparisons
            ),
            "agentview_rgb_exact": all(
                item["agentview_rgb_exact"] for item in comparisons
            ),
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/research/openvla/collect_libero_spatial_paired_release_layer31.yaml",
    )
    parser.add_argument("--task-id", type=int, default=4)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--state-atol", type=float, default=1.0e-6)
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if not np.isfinite(args.state_atol) or args.state_atol < 0:
        parser.error("--state-atol must be finite and non-negative")

    summary = verify_runtime(args)
    print(json.dumps(summary, indent=2))
    print(
        "PAIRED_REPLAY_RUNTIME_OK: "
        f"task={args.task_id} trial={args.trial_index} steps={args.steps}"
    )


if __name__ == "__main__":
    main()
