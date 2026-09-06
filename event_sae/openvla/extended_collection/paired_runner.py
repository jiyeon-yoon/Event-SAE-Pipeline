"""Paired normal/forced-release OpenVLA + LIBERO data collector.

This module is independent of Event-SAE's original collector and of the
normal-only extended collector.  For each candidate it runs the normal policy,
finds an objective stable-grasp-plus-lift trigger, then replays the exact same
initial state and seed while forcing only the gripper command open.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import time
from dataclasses import dataclass
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
from event_sae.openvla.extended_collection.policy import infer_action_with_uncertainty
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
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "userdata",
)


@dataclass
class PrefixTrace:
    model_input_rgb_sha256: str
    stateful_sim_sha256: str
    stateful_sim: dict[str, np.ndarray]
    raw_action: np.ndarray
    policy_action: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray


@dataclass
class PrefixComparison:
    steps_compared: int = 0
    rgb_exact: bool = True
    stateful_sim_exact: bool = True
    max_raw_action_abs_delta: float = 0.0
    max_policy_action_abs_delta: float = 0.0
    max_qpos_abs_delta: float = 0.0
    max_qvel_abs_delta: float = 0.0
    max_stateful_sim_abs_delta: float = 0.0
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
        self.max_stateful_sim_abs_delta = max(
            self.max_stateful_sim_abs_delta, state_delta
        )
        rgb_equal = normal.model_input_rgb_sha256 == forced.model_input_rgb_sha256
        sim_equal = normal.stateful_sim_sha256 == forced.stateful_sim_sha256
        self.rgb_exact = self.rgb_exact and rgb_equal
        self.stateful_sim_exact = self.stateful_sim_exact and sim_equal

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


def _reset_to_initial_state(
    env,
    initial_state: np.ndarray,
    pair_seed: int,
    wait: int,
):
    set_seed(pair_seed)
    env.seed(pair_seed)
    env.reset()
    obs = env.set_init_state(initial_state)
    for _ in range(wait):
        obs, _, _, _ = env.step(dummy_action())
    return obs


def _trace(
    model_input_rgb: np.ndarray,
    raw_action: np.ndarray,
    policy_action: np.ndarray,
    pre_vectors: dict[str, np.ndarray],
) -> PrefixTrace:
    return PrefixTrace(
        model_input_rgb_sha256=array_sha256(model_input_rgb),
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


def _update_post_detach_goal_state(
    *,
    step: int,
    detach_confirmed: int | None,
    goal_satisfied: bool,
    target_is_grasped: bool,
    candidate: int | None,
    consecutive_goal_steps: int,
    required_steps: int,
) -> tuple[int | None, int, int | None]:
    """Confirm that the released object remains in the goal state."""

    if (
        detach_confirmed is None
        or step < detach_confirmed
        or not goal_satisfied
        or target_is_grasped
    ):
        return None, 0, None
    if candidate is None:
        candidate = int(step)
    consecutive_goal_steps += 1
    confirmed = int(step) if consecutive_goal_steps >= required_steps else None
    return candidate, consecutive_goal_steps, confirmed


def _normal_post_success_stop_reason(
    *,
    step: int,
    success_step: int | None,
    t_cmd: int | None,
    t_detach_confirmed: int | None,
    t_goal_stable_after_detach: int | None,
    observation_steps: int,
    post_open_steps: int,
    stable_detach_steps: int,
    goal_stable_steps: int,
) -> str | None:
    """Return why a successful normal rollout may stop, or ``None`` to continue."""

    if success_step is None:
        return None
    if (
        t_cmd is not None
        and t_detach_confirmed is not None
        and t_goal_stable_after_detach is not None
    ):
        return "natural_release_goal_stable"

    # A gripper-open command near the first observation boundary starts one
    # final bounded window. This covers physical gripper-opening latency
    # without allowing an unbounded rollout.
    deadline = int(success_step) + int(observation_steps)
    if t_cmd is not None:
        deadline = max(deadline, int(t_cmd) + int(post_open_steps))
    if t_detach_confirmed is not None:
        deadline = max(
            deadline,
            int(t_detach_confirmed) + int(goal_stable_steps) - 1,
        )
    if step >= deadline:
        return "post_success_observation_timeout"
    return None


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
    reference_initial_z: float | None = None,
) -> tuple[dict[str, Any], dict[int, PrefixTrace], PrefixComparison, Exception | None]:
    if condition not in (NORMAL_CONDITION, FORCED_RELEASE_CONDITION):
        raise ValueError(f"Unsupported condition: {condition}")
    if condition == FORCED_RELEASE_CONDITION and (
        scheduled_trigger_step is None
        or normal_trace is None
        or reference_initial_z is None
    ):
        raise ValueError("Forced release requires the normal trigger and trace")

    # Reset before opening writer state.  A reset failure therefore cannot leave
    # an active half-episode in the durable output streams.
    obs = _reset_to_initial_state(env, initial_state, pair_seed, cfg.env.num_steps_wait)
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
    t_goal_stable_after_detach: int | None = None
    detach_candidate: int | None = None
    destination_contact_candidate: int | None = None
    post_detach_goal_candidate: int | None = None
    consecutive_ungrasped = 0
    consecutive_post_detach_goal = 0
    forced_open_steps = 0
    success_step: int | None = None
    normal_post_success_timeout = False

    # Do not let a late first success consume the post-success observation
    # window.  Before success the original LIBERO horizon is still strict;
    # only a successful normal rollout may use the bounded extension.
    normal_extension = (
        cfg.paired_release.normal_post_success_steps
        + cfg.paired_release.max_force_open_steps
        + cfg.paired_release.stable_detach_steps
        + cfg.paired_release.post_detach_goal_stable_steps
    )
    loop_limit = (
        max_steps + normal_extension if condition == NORMAL_CONDITION else max_steps
    )
    for step in range(loop_limit):
        if condition == NORMAL_CONDITION and success_step is None and step >= max_steps:
            break
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

            common = {
                "episode_num": episode_num,
                "pair_id": pair_id,
                "condition": condition,
                "pair_seed": pair_seed,
                "task_id": task_id,
                "task_episode_idx": trial_idx,
                "task_description": description,
                "step_in_episode": step,
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
            )
            raw_action = np.asarray(policy_out.raw_action).copy()
            policy_action = to_executed_libero_action(raw_action)
            current_trace = _trace(
                policy_out.model_input_rgb,
                raw_action,
                policy_action,
                pre.sim_vectors,
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
                reference = normal_trace.get(step)
                if step <= int(scheduled_trigger_step):
                    if reference is None:
                        invalid_reason = "normal_prefix_step_missing"
                    elif not comparison.compare(
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
            if t_goal_stable_after_detach is None:
                (
                    post_detach_goal_candidate,
                    consecutive_post_detach_goal,
                    goal_stable_confirmation,
                ) = _update_post_detach_goal_state(
                    step=step,
                    detach_confirmed=t_detach_confirmed,
                    goal_satisfied=goal_satisfied_post,
                    target_is_grasped=target_grasped(post.json_state, target_object),
                    candidate=post_detach_goal_candidate,
                    consecutive_goal_steps=consecutive_post_detach_goal,
                    required_steps=(cfg.paired_release.post_detach_goal_stable_steps),
                )
                if goal_stable_confirmation is not None:
                    t_goal_stable_after_detach = int(goal_stable_confirmation)
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
                pre_vectors=pre.sim_vectors,
                post_vectors=post.sim_vectors,
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
                    # LIBERO's On predicate can report success while the robot
                    # still holds the object. Continue the unmodified policy so
                    # natural open, detach, and stable placement are observable.
                    stop_reason = _normal_post_success_stop_reason(
                        step=step,
                        success_step=success_step,
                        t_cmd=t_cmd,
                        t_detach_confirmed=t_detach_confirmed,
                        t_goal_stable_after_detach=(t_goal_stable_after_detach),
                        observation_steps=(
                            cfg.paired_release.normal_post_success_steps
                        ),
                        post_open_steps=(cfg.paired_release.max_force_open_steps),
                        stable_detach_steps=(cfg.paired_release.stable_detach_steps),
                        goal_stable_steps=(
                            cfg.paired_release.post_detach_goal_stable_steps
                        ),
                    )
                    if stop_reason is not None:
                        normal_post_success_timeout = (
                            stop_reason == "post_success_observation_timeout"
                        )
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
        "t_goal_stable_after_detach": t_goal_stable_after_detach,
        "normal_post_success_timeout": normal_post_success_timeout,
        "detach_confirmation_steps": cfg.paired_release.stable_detach_steps,
        "post_detach_goal_confirmation_steps": (
            cfg.paired_release.post_detach_goal_stable_steps
        ),
        "forced_open_steps": forced_open_steps,
        "invalid_reason": invalid_reason,
    }
    final = writer.finish_episode(
        result=result,
        compress_npz=cfg.output.compress_npz,
    )
    activation_collector.flush_episode()
    return final, trace, comparison, fatal


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
    if cfg.env.task_suite_name not in _MAX_STEPS_PER_SUITE:
        raise ValueError(f"Unsupported task suite: {cfg.env.task_suite_name}")
    max_steps = _MAX_STEPS_PER_SUITE[cfg.env.task_suite_name]

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = _new_run_dir(
        cfg.output.root_dir,
        f"{cfg.env.task_suite_name}-paired-release",
    )
    manifest = {
        "schema_version": "extended_openvla_libero_paired_release_v3",
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
            "detach_confirmation_steps": cfg.paired_release.stable_detach_steps,
            "normal_post_success_steps": (cfg.paired_release.normal_post_success_steps),
            "post_detach_goal_stable_steps": (
                cfg.paired_release.post_detach_goal_stable_steps
            ),
            "primary_analysis_filter": (
                "valid_pair and normal_success and "
                "normal_natural_release_observed and "
                "normal_post_detach_goal_stable"
            ),
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
                                "normal_post_detach_goal_stable": bool(
                                    normal.get("t_goal_stable_after_detach") is not None
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
                        forced, _, comparison, forced_fatal = _run_episode(
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

                        invalid_reason = forced["invalid_reason"]
                        if (
                            normal["initial_state_sha256"]
                            != forced["initial_state_sha256"]
                        ):
                            invalid_reason = "initial_state_hash_mismatch"
                        elif (
                            normal["warm_start_sim_state_sha256"]
                            != forced["warm_start_sim_state_sha256"]
                        ):
                            invalid_reason = "warm_start_sim_state_mismatch"
                        elif (
                            normal["warm_start_source_rgb_sha256"]
                            != forced["warm_start_source_rgb_sha256"]
                        ):
                            invalid_reason = "warm_start_source_rgb_mismatch"
                        if forced_fatal is not None and invalid_reason is None:
                            invalid_reason = "forced_exception"
                        valid = invalid_reason is None
                        status = "valid" if valid else "invalid"
                        aggregate["valid_pairs" if valid else "invalid_pairs"] += 1
                        normal_natural_release = bool(
                            normal.get("t_cmd") is not None
                            and normal.get("t_detach_confirmed") is not None
                        )
                        normal_post_detach_goal_stable = bool(
                            normal.get("t_goal_stable_after_detach") is not None
                        )
                        primary = bool(
                            valid
                            and normal["success"]
                            and normal_natural_release
                            and normal_post_detach_goal_stable
                        )
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
                            "normal_post_detach_goal_stable": (
                                normal_post_detach_goal_stable
                            ),
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
