"""LIBERO closed-loop evaluation loop for openVLA, with optional SAE
activation collection and user-supplied extra hooks.

Single-feature / zero-out interventions are not built in; supply them as an
``extra_hook_applier`` (see ``scripts/openvla/intervene.py``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import tqdm
from libero.libero import benchmark

from event_sae.openvla.activations import apply_collect_hooks, apply_sae_topk_collect_hooks
from event_sae.openvla.eval.config import RunConfig, resolve_task_ids
from event_sae.openvla.eval.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    quat2axisangle,
    save_rollout_video,
)
from event_sae.openvla.eval.logging_utils import (
    append_csv_row,
    create_run_dir,
    get_run_id,
    open_log_file,
    save_actions_json,
    write_csv_header,
)
from event_sae.openvla.eval.model import get_action, get_processor, load_model
from event_sae.openvla.intervene import SAEHookError
from event_sae.openvla.eval.utils import (
    get_resize_size,
    invert_gripper_action,
    normalize_gripper_action,
    set_seed_everywhere,
)


@dataclass
class EvalResult:
    run_dir: str
    success_rate: float
    prompt_records_path: str | None = None
    trajectory_records_path: str | None = None


def _parse_layer_idxs(raw: str | int) -> List[int]:
    if isinstance(raw, int):
        return [raw]
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


_MAX_STEPS_PER_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def eval_libero(
    cfg: RunConfig,
    extra_hook_applier: Optional[Callable[..., List[object]]] = None,
) -> EvalResult:
    set_seed_everywhere(cfg.env.seed)
    cfg_unnorm_key = cfg.env.task_suite_name

    model = load_model(cfg)
    if cfg.model.family == "openvla" and hasattr(model, "norm_stats"):
        if cfg_unnorm_key not in model.norm_stats and f"{cfg_unnorm_key}_no_noops" in model.norm_stats:
            cfg_unnorm_key = f"{cfg_unnorm_key}_no_noops"
        assert cfg_unnorm_key in model.norm_stats, (
            f"Action un-norm key {cfg_unnorm_key} not found in model norm stats."
        )

    processor = get_processor(cfg) if cfg.model.family == "openvla" else None

    run_id = get_run_id(cfg.env.task_suite_name, cfg.model.family)
    run_dir = create_run_dir(cfg.logging.root_dir, run_id)
    log_path = open_log_file(run_dir)
    log_file = open(log_path, "w")
    print(f"Logging to local log file: {log_path}")
    log_file.write(f"Logging to local log file: {log_path}\n")

    prompt_records_path: str | None = None
    prompt_records_file = None
    if cfg.logging.save_prompt_records:
        prompt_records_path = os.path.join(run_dir, "prompt_records.jsonl")
        prompt_records_file = open(prompt_records_path, "w", encoding="utf-8")

    trajectory_records_path: str | None = None
    trajectory_records_file = None
    if cfg.logging.save_trajectory_records:
        trajectory_records_path = os.path.join(run_dir, "trajectory_records.jsonl")
        trajectory_records_file = open(trajectory_records_path, "w", encoding="utf-8")

    hooks: List[object] = []
    if cfg.sae_collect.enabled:
        layer_idxs = _parse_layer_idxs(cfg.sae_collect.layer_idxs)
        if cfg.sae_collect.mode == "dense":
            sae_out_dir = Path(run_dir) / "sae_activations" / "post_mlp_residual"
            handle = apply_collect_hooks(model, layer_idxs, sae_out_dir, cfg.sae_collect.flush_every)
            hooks.append(handle)
            log_file.write(
                f"SAE collect (dense+metadata): layers={layer_idxs} "
                f"flush_every={cfg.sae_collect.flush_every} out_dir={sae_out_dir}\n"
            )
        elif cfg.sae_collect.mode == "topk":
            if not cfg.sae_collect.sae_checkpoint:
                raise ValueError("sae_collect.mode='topk' requires sae_collect.sae_checkpoint")
            if len(layer_idxs) != 1:
                raise ValueError(f"sae_collect.mode='topk' supports one layer; got {layer_idxs}")
            sae_out_dir = Path(run_dir) / "topk_activations"
            handle = apply_sae_topk_collect_hooks(
                model,
                layer_idx=layer_idxs[0],
                sae_checkpoint_path=cfg.sae_collect.sae_checkpoint,
                output_dir=sae_out_dir,
                topk=cfg.sae_collect.topk,
                rows_per_shard=cfg.sae_collect.rows_per_shard,
            )
            hooks.append(handle)
            log_file.write(
                f"SAE collect (online topk): layer={layer_idxs[0]} "
                f"sae_checkpoint={cfg.sae_collect.sae_checkpoint} topk={cfg.sae_collect.topk} "
                f"out_dir={sae_out_dir}\n"
            )
        else:
            raise ValueError(f"Unknown sae_collect.mode: {cfg.sae_collect.mode!r}")
    if extra_hook_applier is not None:
        hooks.extend(extra_hook_applier(model=model, cfg=cfg, run_dir=run_dir, log_file=log_file))

    csv_path = os.path.join(run_dir, "events.csv")
    write_csv_header(csv_path)

    actions_path = os.path.join(run_dir, "actions.json")
    all_actions_by_task = {}

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.env.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    selected_task_ids = resolve_task_ids(cfg.env.task_ids, num_tasks_in_suite)
    print(f"Task suite: {cfg.env.task_suite_name}")
    print(f"Selected task ids: {selected_task_ids}")
    log_file.write(f"Task suite: {cfg.env.task_suite_name}\n")
    log_file.write(f"Selected task ids: {selected_task_ids}\n")

    resize_size = get_resize_size(cfg.model.family)
    if cfg.env.task_suite_name not in _MAX_STEPS_PER_SUITE:
        raise ValueError(f"Unexpected task suite: {cfg.env.task_suite_name}")
    max_steps = _MAX_STEPS_PER_SUITE[cfg.env.task_suite_name]

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(selected_task_ids):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task, resolution=256)

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(cfg.env.num_trials_per_task)):
            episode_num = total_episodes + 1
            if prompt_records_file is not None:
                prompt_records_file.write(
                    json.dumps(
                        {
                            "episode_num": episode_num,
                            "task_id": task_id,
                            "task_episode_idx": episode_idx,
                            "task_description": task_description,
                        }
                    )
                    + "\n"
                )
                prompt_records_file.flush()

            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            env.reset()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            done = False
            replay_images = []

            print(f"Starting episode {task_episodes + 1}...")
            log_file.write(f"Starting episode {task_episodes + 1}...\n")
            current_episode_actions = []

            while t < max_steps + cfg.env.num_steps_wait:
                try:
                    if t < cfg.env.num_steps_wait:
                        obs, reward, done, info = env.step(get_libero_dummy_action())
                        t += 1
                        continue

                    img = get_libero_image(obs, resize_size)
                    replay_images.append(img)

                    observation = {
                        "full_image": img,
                        "state": np.concatenate(
                            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                        ),
                    }
                    # Per-step metadata read by activation collection + intervention hooks.
                    model._sae_hook_context = {
                        "episode_num": episode_num,
                        "task_id": task_id,
                        "task_episode_idx": episode_idx,
                        "task_description": task_description,
                        "step_in_episode": t - cfg.env.num_steps_wait,
                    }

                    action = get_action(
                        model,
                        processor,
                        cfg,
                        observation,
                        task_description,
                        unnorm_key=cfg_unnorm_key,
                    )

                    action = normalize_gripper_action(action, binarize=True)
                    if cfg.model.family == "openvla":
                        action = invert_gripper_action(action)

                    obs, reward, done, info = env.step(action.tolist())
                    current_episode_actions.append(action.tolist())
                    if trajectory_records_file is not None:
                        trajectory_records_file.write(
                            json.dumps(
                                {
                                    "episode_num": episode_num,
                                    "task_id": task_id,
                                    "task_episode_idx": episode_idx,
                                    "task_description": task_description,
                                    "step_in_episode": t - cfg.env.num_steps_wait,
                                    "eef_pos": [float(x) for x in obs["robot0_eef_pos"]],
                                    "eef_quat": [float(x) for x in obs["robot0_eef_quat"]],
                                    "gripper_qpos": [
                                        float(x)
                                        for x in np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
                                    ],
                                    "gripper_action": float(action[-1]),
                                    "done": bool(done),
                                }
                            )
                            + "\n"
                        )
                        trajectory_records_file.flush()

                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except SAEHookError as exc:
                    log_file.write(f"Fatal SAE hook error: {exc}\n")
                    log_file.flush()
                    raise
                except Exception as exc:
                    print(f"Caught exception: {exc}")
                    log_file.write(f"Caught exception: {exc}\n")
                    break

            task_episodes += 1
            total_episodes += 1

            all_actions_by_task.setdefault(task_description, {})[episode_idx] = current_episode_actions
            if cfg.logging.save_actions:
                save_actions_json(actions_path, all_actions_by_task)

            if cfg.logging.save_video:
                save_rollout_video(
                    replay_images,
                    total_episodes,
                    success=done,
                    task_description=task_description,
                    out_dir=os.path.join(run_dir, "videos"),
                    log_file=log_file,
                )

            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        task_success_rate = float(task_successes) / float(task_episodes)
        print(f"Current task success rate: {task_success_rate}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {task_success_rate}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()
        append_csv_row(csv_path, task_description, task_success_rate)

    if cfg.logging.save_actions:
        save_actions_json(actions_path, all_actions_by_task)

    for hook in hooks:
        hook.remove()

    if prompt_records_file is not None:
        prompt_records_file.close()
    if trajectory_records_file is not None:
        trajectory_records_file.close()
    log_file.close()
    return EvalResult(
        run_dir=run_dir,
        success_rate=float(total_successes) / float(total_episodes),
        prompt_records_path=prompt_records_path,
        trajectory_records_path=trajectory_records_path,
    )
