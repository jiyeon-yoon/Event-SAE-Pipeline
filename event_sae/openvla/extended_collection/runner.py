"""Independent extended OpenVLA + LIBERO rollout collector.

The original Event-SAE runner is intentionally left untouched.  This runner
keeps its proven layer-31 activation hook format, but owns rollout execution,
alignment, rich simulator telemetry, exact model-input images, and policy
uncertainty records.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from event_sae.openvla.extended_collection.activation import (
    Layer31ActivationCollector,
)
from event_sae.openvla.extended_collection.config import ExtendedRunConfig
from event_sae.openvla.extended_collection.policy import (
    _openvla_action_vocab_size,
    infer_action_with_uncertainty,
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
from event_sae.openvla.extended_collection.writer import ExtendedRunWriter, jsonable


_MAX_STEPS_PER_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def _git_state(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "dirty": bool(run("status", "--porcelain")),
            "remote": run("config", "--get", "remote.origin.url") or None,
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "remote": None}


def _resolve_task_ids(raw: str | list[int] | int | None, count: int) -> list[int]:
    if raw in (None, "", "all"):
        selected = list(range(count))
    elif isinstance(raw, int):
        selected = [raw]
    elif isinstance(raw, str):
        selected = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        selected = [int(value) for value in raw]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError(f"Invalid or duplicate task ids: {selected}")
    if min(selected) < 0 or max(selected) >= count:
        raise ValueError(f"Task ids must be in [0, {count}): {selected}")
    return selected


def _make_env(task, resolution: int, seed: int):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(seed)
    return env, task.language, bddl_path


def _resolve_unnorm_key(model, suite: str) -> str:
    key = suite
    if hasattr(model, "norm_stats"):
        fallback = f"{suite}_no_noops"
        if key not in model.norm_stats and fallback in model.norm_stats:
            key = fallback
        if key not in model.norm_stats:
            raise KeyError(f"Action un-normalization key not found: {suite!r}")
    return key


def _save_video(frames: list[np.ndarray], path: Path, fps: int) -> None:
    import imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps)
    try:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))
    finally:
        writer.close()


def _new_run_dir(root: str | Path, suite: str) -> Path:
    stamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    return Path(root).expanduser().resolve() / (
        f"EXTENDED-{suite}-openvla-{stamp}-{os.getpid()}"
    )


def _safe_info(info: Any) -> dict[str, Any]:
    if not isinstance(info, dict):
        return {"value": repr(info)}
    try:
        return jsonable(info)
    except TypeError:
        # Keep supported fields while exposing exactly which fields were not
        # serializable instead of silently dropping the whole info payload.
        safe: dict[str, Any] = {}
        for key, value in info.items():
            try:
                safe[str(key)] = jsonable(value)
            except TypeError:
                safe[str(key)] = {"unserializable_type": type(value).__name__}
        return safe


def collect_extended_libero(cfg: ExtendedRunConfig) -> dict[str, Any]:
    """Collect one extended dataset run and return its aggregate summary."""

    from libero.libero import benchmark

    set_seed(cfg.env.seed)
    model = load_openvla(cfg)
    processor = load_processor(cfg)
    unnorm_key = _resolve_unnorm_key(model, cfg.env.task_suite_name)
    action_vocab_size = _openvla_action_vocab_size(model)

    benchmark_cls = benchmark.get_benchmark_dict()[cfg.env.task_suite_name]
    suite = benchmark_cls()
    task_ids = _resolve_task_ids(cfg.env.task_ids, suite.n_tasks)
    if cfg.env.task_suite_name not in _MAX_STEPS_PER_SUITE:
        raise ValueError(f"Unsupported task suite: {cfg.env.task_suite_name}")
    max_steps = _MAX_STEPS_PER_SUITE[cfg.env.task_suite_name]

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = _new_run_dir(cfg.output.root_dir, cfg.env.task_suite_name)
    manifest = {
        "schema_version": "extended_openvla_libero_v2",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code": _git_state(repo_root),
        "config": cfg.to_dict(),
        "resolved_task_ids": task_ids,
        "alignment": "pre_state + raw_action/executed_action -> post_state",
        "activation_stream": {
            "layer": 31,
            "submodule": "post_mlp_residual",
            "format": "dense torch float32 shards + activation_index.jsonl",
            "forwards_per_policy_step": 7,
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
            "distribution": (
                "scalar summaries of the full vocabulary and the conditional "
                "action-token subset at each of 7 generated action dimensions"
            ),
        },
        "vision": {
            "model_input_rgb": "lossless uint8 arrays in per-episode NPZ",
            "rollout_video": bool(cfg.output.save_video),
            "depth": False,
            "segmentation": False,
        },
    }

    activation_collector = None
    aggregate = {"episodes": 0, "successes": 0, "failures": 0}
    with ExtendedRunWriter(
        run_dir, flush_every_step=cfg.output.flush_jsonl_every_step
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
                        f"Task {task_id} has {len(initial_states)} initial states, but "
                        f"{cfg.env.num_trials_per_task} trials were requested"
                    )
                env, description, bddl_path = _make_env(
                    task, cfg.env.resolution, cfg.env.seed
                )
                try:
                    env.reset()
                    preflight_obs = env.set_init_state(initial_states[0])
                    preflight_obs, _, _, _ = env.step(dummy_action())
                    capabilities = validate_preflight(
                        env, preflight_obs, strict=cfg.collection.strict_preflight
                    )
                    schema = build_simulator_schema(env, preflight_obs)
                    schema.update(
                        {
                            "task_id": task_id,
                            "task_description": description,
                            "bddl_file": bddl_path,
                            "capabilities": capabilities,
                        }
                    )
                    writer.write_schema(task_id, schema)

                    task_successes = 0
                    for trial_idx in range(cfg.env.num_trials_per_task):
                        episode_num = aggregate["episodes"] + 1
                        initial_state = initial_states[trial_idx]
                        prompt = writer.begin_episode(
                            prompt_record={
                                "episode_num": episode_num,
                                "task_id": task_id,
                                "task_episode_idx": trial_idx,
                                "task_description": description,
                                "suite_seed": cfg.env.seed,
                                "task_initial_state_index": trial_idx,
                                "bddl_file": bddl_path,
                            },
                            initial_state=initial_state,
                        )

                        env.reset()
                        obs = env.set_init_state(initial_state)
                        for _ in range(cfg.env.num_steps_wait):
                            obs, _, _, _ = env.step(dummy_action())

                        frames: list[np.ndarray] = []
                        success = False
                        caught_exception: str | None = None
                        fatal: BaseException | None = None
                        control_dt = 1.0 / float(getattr(env.env, "control_freq", 20))
                        step_count = 0
                        previous_pre = None

                        for step in range(max_steps):
                            try:
                                source_rgb = get_source_rgb(obs, 224)
                                pre = capture_snapshot(
                                    env, obs, previous=previous_pre, dt=control_dt
                                )
                                previous_pre = pre
                                model._sae_hook_context = {
                                    "episode_num": episode_num,
                                    "task_id": task_id,
                                    "task_episode_idx": trial_idx,
                                    "task_description": description,
                                    "step_in_episode": step,
                                }
                                policy_out = infer_action_with_uncertainty(
                                    model,
                                    processor,
                                    cfg,
                                    source_rgb,
                                    description,
                                    unnorm_key,
                                )
                                # Preserve OpenVLA's exact numpy dtype for execution; casting here
                                # would make this collector a subtly different policy baseline.
                                raw_action = np.asarray(policy_out.raw_action).copy()
                                executed_action = to_executed_libero_action(raw_action)
                                next_obs, reward, done, info = env.step(
                                    executed_action.tolist()
                                )
                                post = capture_snapshot(
                                    env, next_obs, previous=pre, dt=control_dt
                                )
                                policy_record = {
                                    "action_token_ids": policy_out.action_token_ids,
                                    "uncertainty": policy_out.uncertainty,
                                    "preprocessing": policy_out.preprocessing,
                                }
                                writer.write_step(
                                    common={
                                        "episode_num": episode_num,
                                        "task_id": task_id,
                                        "task_episode_idx": trial_idx,
                                        "task_description": description,
                                        "step_in_episode": step,
                                    },
                                    pre_json=pre.json_state,
                                    post_json=post.json_state,
                                    pre_vectors=pre.sim_vectors,
                                    post_vectors=post.sim_vectors,
                                    raw_action=raw_action,
                                    executed_action=executed_action,
                                    policy=policy_record,
                                    model_input_rgb=(
                                        policy_out.model_input_rgb
                                        if cfg.output.save_model_input_rgb
                                        else None
                                    ),
                                    reward=float(reward),
                                    done=bool(done),
                                    info=_safe_info(info),
                                )
                                if cfg.output.save_video:
                                    frames.append(source_rgb)
                                obs = next_obs
                                step_count += 1
                                if done:
                                    success = True
                                    break
                            except (
                                BaseException
                            ) as exc:  # save the partial episode before stopping
                                caught_exception = repr(exc)
                                fatal = exc
                                break

                        video_rel: str | None = None
                        if cfg.output.save_video and frames:
                            video_path = (
                                run_dir
                                / "videos"
                                / (
                                    f"episode_{episode_num:06d}--success={success}--task={task_id:02d}.mp4"
                                )
                            )
                            _save_video(frames, video_path, cfg.output.video_fps)
                            video_rel = str(video_path.relative_to(run_dir))
                        writer.finish_episode(
                            result={
                                "episode_num": episode_num,
                                "task_id": task_id,
                                "task_episode_idx": trial_idx,
                                "task_description": description,
                                "initial_state_sha256": prompt["initial_state_sha256"],
                                "success": success,
                                "num_actions": step_count,
                                "caught_exception": caught_exception,
                                "video_path": video_rel,
                            },
                            compress_npz=cfg.output.compress_npz,
                        )
                        aggregate["episodes"] += 1
                        if success:
                            aggregate["successes"] += 1
                            task_successes += 1
                        else:
                            aggregate["failures"] += 1
                        print(
                            f"episode={episode_num} task={task_id} trial={trial_idx} "
                            f"steps={step_count} success={success}",
                            flush=True,
                        )
                        if fatal is not None and cfg.collection.fail_fast:
                            raise RuntimeError(
                                f"Extended collection failed at episode {episode_num}, step "
                                f"{step_count}: {fatal!r}"
                            ) from fatal

                    print(
                        f"task={task_id} success_rate="
                        f"{task_successes / cfg.env.num_trials_per_task:.4f}",
                        flush=True,
                    )
                finally:
                    env.close()
        finally:
            if activation_collector is not None:
                activation_collector.close()

        aggregate["success_rate"] = (
            aggregate["successes"] / aggregate["episodes"]
            if aggregate["episodes"]
            else 0.0
        )
        aggregate["run_dir"] = str(run_dir)
        (run_dir / "summary.json").write_text(
            json.dumps(jsonable(aggregate), indent=2) + "\n", encoding="utf-8"
        )
    print(f"EXTENDED_COLLECTION_OK: {run_dir}")
    return aggregate
