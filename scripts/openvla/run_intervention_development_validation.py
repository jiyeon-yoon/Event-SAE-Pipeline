"""Run a small paired intervention validation without replacing the full sweep.

This is an implementation check, not the paper's main statistical intervention
experiment.  It evaluates the same five initial states per LIBERO-Spatial task
under four conditions: raw, event feature alpha=1, the same event feature
alpha=0, and a random-alive feature alpha=0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


CONDITION_SPECS = (
    ("raw", "raw", None, None),
    ("event-alpha1", "intervention", "event", 1.0),
    ("event-alpha0", "intervention", "event", 0.0),
    ("random-alpha0", "intervention", "random", 0.0),
)
ACTION_TOLERANCE = 1e-6


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_provenance(root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "remote": git("remote", "get-url", "origin"),
    }


def _protocol_config(cfg, task_ids: list[int]) -> dict:
    payload = asdict(cfg)
    payload["logging"].pop("root_dir", None)
    payload["env"]["resolved_task_ids"] = task_ids
    return payload


def select_validation_candidates(rows: list[dict]) -> tuple[dict, dict]:
    def rank_one(ranking: str) -> dict:
        selected = [
            row
            for row in rows
            if row.get("ranking") == ranking and int(row.get("rank", -1)) == 1
        ]
        if len(selected) != 1:
            raise RuntimeError(
                f"Expected one rank-1 {ranking} candidate, found {len(selected)}"
            )
        feature_id = int(selected[0]["feature_id"])
        if not (0 <= feature_id < 32768):
            raise RuntimeError(f"Invalid {ranking} feature id: {feature_id}")
        return selected[0]

    event = rank_one("event_aligned")
    random = rank_one("random_alive")
    if int(event["feature_id"]) == int(random["feature_id"]):
        raise RuntimeError("Event and random validation candidates must differ")
    return event, random


def _run(command: list[str], *, repo_root: Path) -> None:
    print("RUN:", " ".join(command), flush=True)
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + old if old else "")
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def _expected_fingerprint(result: dict) -> str:
    keys = ["mode", "completed_rollouts", "run_config", "code"]
    if result.get("mode") == "intervention":
        keys.extend(
            [
                "feature_id",
                "alpha",
                "layer_idx",
                "hook_start_step",
                "sae_sha256",
            ]
        )
    return _fingerprint({key: result.get(key) for key in keys})


def _canonical_prompt_manifest(rows: list[dict]) -> list[dict]:
    manifest = []
    for row in rows:
        state_hash = row.get("initial_state_sha256")
        if not isinstance(state_hash, str) or len(state_hash) != 64:
            raise RuntimeError("Prompt record is missing initial_state_sha256")
        manifest.append(
            {
                "episode_num": int(row["episode_num"]),
                "task_id": int(row["task_id"]),
                "task_episode_idx": int(row["task_episode_idx"]),
                "task_description": str(row["task_description"]),
                "initial_state_sha256": state_hash,
            }
        )
    return manifest


def _manifest_sha256(manifest: list[dict]) -> str:
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _outcome_map(rows: list[dict]) -> dict[tuple[int, int], dict]:
    result = {}
    for row in rows:
        key = (int(row["task_id"]), int(row["task_episode_idx"]))
        if key in result:
            raise RuntimeError(f"Duplicate episode outcome: {key}")
        if row.get("caught_exception") is not None:
            raise RuntimeError(f"Episode {key} caught an exception")
        result[key] = row
    return result


def _action_map(
    actions: dict, prompt_manifest: list[dict]
) -> dict[tuple[int, int], list[list[float]]]:
    result = {}
    for prompt in prompt_manifest:
        key = (int(prompt["task_id"]), int(prompt["task_episode_idx"]))
        description = prompt["task_description"]
        task_actions = actions.get(description)
        if not isinstance(task_actions, dict):
            raise RuntimeError(f"Actions are missing task {description!r}")
        sequence = task_actions.get(str(key[1]))
        if not isinstance(sequence, list):
            raise RuntimeError(f"Actions are missing task/trial {key}")
        result[key] = sequence
    return result


def summarize_task_success(outcomes: dict[tuple[int, int], dict]) -> dict[str, dict]:
    grouped: dict[int, list[bool]] = defaultdict(list)
    for (task_id, _), row in sorted(outcomes.items()):
        grouped[task_id].append(bool(row["success"]))
    return {
        str(task_id): {
            "successes": sum(values),
            "rollouts": len(values),
            "success_rate": sum(values) / len(values),
        }
        for task_id, values in sorted(grouped.items())
    }


def compare_actions(
    baseline: dict[tuple[int, int], list[list[float]]],
    condition: dict[tuple[int, int], list[list[float]]],
    *,
    tolerance: float = ACTION_TOLERANCE,
) -> dict:
    if baseline.keys() != condition.keys():
        raise RuntimeError("Action task/trial keys do not match")

    total_values = 0
    absolute_delta_sum = 0.0
    maximum_delta = 0.0
    changed_episodes = 0
    length_mismatches = 0
    episode_rows = []
    per_task: dict[int, dict[str, int | float]] = defaultdict(
        lambda: {"episodes": 0, "changed_episodes": 0, "max_abs_delta": 0.0}
    )

    for key in sorted(baseline):
        raw_sequence = baseline[key]
        condition_sequence = condition[key]
        common_steps = min(len(raw_sequence), len(condition_sequence))
        first_changed_step = None
        episode_max = 0.0
        episode_sum = 0.0
        episode_values = 0
        for step in range(common_steps):
            raw_action = raw_sequence[step]
            condition_action = condition_sequence[step]
            if len(raw_action) != len(condition_action):
                raise RuntimeError(f"Action dimensions differ at {key}, step {step}")
            step_max = 0.0
            for raw_value, condition_value in zip(raw_action, condition_action):
                delta = abs(float(raw_value) - float(condition_value))
                if not math.isfinite(delta):
                    raise RuntimeError(f"Non-finite action delta at {key}, step {step}")
                episode_sum += delta
                episode_values += 1
                step_max = max(step_max, delta)
            episode_max = max(episode_max, step_max)
            if first_changed_step is None and step_max > tolerance:
                first_changed_step = step
        if len(raw_sequence) != len(condition_sequence):
            length_mismatches += 1
            if first_changed_step is None:
                first_changed_step = common_steps
        changed = first_changed_step is not None
        changed_episodes += int(changed)
        total_values += episode_values
        absolute_delta_sum += episode_sum
        maximum_delta = max(maximum_delta, episode_max)
        task_summary = per_task[key[0]]
        task_summary["episodes"] = int(task_summary["episodes"]) + 1
        task_summary["changed_episodes"] = int(task_summary["changed_episodes"]) + int(
            changed
        )
        task_summary["max_abs_delta"] = max(
            float(task_summary["max_abs_delta"]), episode_max
        )
        episode_rows.append(
            {
                "task_id": key[0],
                "task_episode_idx": key[1],
                "raw_steps": len(raw_sequence),
                "condition_steps": len(condition_sequence),
                "first_changed_step": first_changed_step,
                "max_abs_delta": episode_max,
                "mean_abs_delta": episode_sum / episode_values if episode_values else 0.0,
            }
        )
    return {
        "tolerance": tolerance,
        "episodes": len(baseline),
        "changed_episodes": changed_episodes,
        "length_mismatch_episodes": length_mismatches,
        "max_abs_delta": maximum_delta,
        "mean_abs_delta": (
            absolute_delta_sum / total_values if total_values else 0.0
        ),
        "all_actions_match": changed_episodes == 0,
        "per_task": {str(key): value for key, value in sorted(per_task.items())},
        "per_episode": episode_rows,
    }


def _validate_result(
    path: Path,
    *,
    condition_name: str,
    expected_mode: str,
    expected_rollouts: int,
    expected_commit: str,
    expected_run_config: dict,
    expected_feature_id: int | None,
    expected_alpha: float | None,
    expected_sae_sha256: str,
) -> dict:
    result = _json(path)
    expected_schema = (
        "event_sae_policy_evaluation_v1"
        if expected_mode == "raw"
        else "event_sae_feature_intervention_v1"
    )
    if result.get("schema_version") != expected_schema:
        raise RuntimeError(f"Invalid schema for {condition_name}")
    if result.get("mode") != expected_mode:
        raise RuntimeError(f"Invalid mode for {condition_name}")
    if int(result.get("completed_rollouts", -1)) != expected_rollouts:
        raise RuntimeError(f"Invalid rollout count for {condition_name}")
    if result.get("code", {}).get("commit") != expected_commit:
        raise RuntimeError(f"Code revision mismatch for {condition_name}")
    if result.get("code", {}).get("dirty") is not False:
        raise RuntimeError(f"Dirty-code result for {condition_name}")
    if result.get("run_config") != expected_run_config:
        raise RuntimeError(f"Run configuration mismatch for {condition_name}")
    if result.get("protocol_fingerprint") != _expected_fingerprint(result):
        raise RuntimeError(f"Protocol fingerprint mismatch for {condition_name}")

    run_dir = Path(result["run_dir"])
    stdout = run_dir / "stdout.log"
    if not stdout.is_file() or "Caught exception:" in stdout.read_text(encoding="utf-8"):
        raise RuntimeError(f"Invalid stdout for {condition_name}: {stdout}")
    required = {
        "prompt_records_path": run_dir / "prompt_records.jsonl",
        "episode_results_path": run_dir / "episode_results.jsonl",
        "actions_path": run_dir / "actions.json",
    }
    for field, expected_path in required.items():
        actual = result.get(field)
        if actual is None or Path(actual).resolve() != expected_path.resolve():
            raise RuntimeError(f"Invalid {field} for {condition_name}")
        if not expected_path.is_file():
            raise FileNotFoundError(expected_path)

    prompt_rows = _jsonl(required["prompt_records_path"])
    outcome_rows = _jsonl(required["episode_results_path"])
    if len(prompt_rows) != expected_rollouts or len(outcome_rows) != expected_rollouts:
        raise RuntimeError(f"Incomplete paired evidence for {condition_name}")
    manifest = _canonical_prompt_manifest(prompt_rows)
    outcomes = _outcome_map(outcome_rows)
    action_map = _action_map(_json(required["actions_path"]), manifest)

    manifest_keys = {
        (row["task_id"], row["task_episode_idx"], row["initial_state_sha256"])
        for row in manifest
    }
    outcome_keys = {
        (int(row["task_id"]), int(row["task_episode_idx"]), row["initial_state_sha256"])
        for row in outcome_rows
    }
    if manifest_keys != outcome_keys:
        raise RuntimeError(f"Prompt/outcome initial states differ for {condition_name}")

    if expected_mode == "intervention":
        expected_fields = {
            "feature_id": expected_feature_id,
            "alpha": expected_alpha,
            "layer_idx": 31,
            "hook_start_step": 0,
            "sae_sha256": expected_sae_sha256,
        }
        mismatches = {
            field: (result.get(field), value)
            for field, value in expected_fields.items()
            if result.get(field) != value
        }
        if mismatches:
            raise RuntimeError(
                f"Intervention protocol mismatch for {condition_name}: {mismatches}"
            )
        hook = result.get("hook_metrics", {})
        if int(hook.get("num_forwards", 0)) <= 0:
            raise RuntimeError(f"Hook did not run for {condition_name}")
        if int(hook.get("active_feature_values", 0)) <= 0:
            raise RuntimeError(f"Feature was never active for {condition_name}")
        if expected_alpha == 0.0:
            if int(hook.get("post_intervention_active_feature_values", -1)) != 0:
                raise RuntimeError(f"Feature was not fully zeroed for {condition_name}")
            if float(hook.get("absolute_feature_delta", 0.0)) <= 0.0:
                raise RuntimeError(f"Feature zeroing had no latent delta for {condition_name}")
        elif expected_alpha == 1.0:
            if hook.get("post_intervention_active_feature_values") != hook.get(
                "active_feature_values"
            ):
                raise RuntimeError("alpha=1 changed active feature count")
            if float(hook.get("absolute_feature_delta", math.inf)) != 0.0:
                raise RuntimeError("alpha=1 changed the target latent")

    return {
        "result": result,
        "manifest": manifest,
        "manifest_sha256": _manifest_sha256(manifest),
        "outcomes": outcomes,
        "actions": action_map,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    args = parser.parse_args()

    from event_sae.openvla.eval.config import load_config, resolve_task_ids

    repo_root = Path(__file__).resolve().parents[2]
    code = _git_provenance(repo_root)
    if code["dirty"]:
        raise RuntimeError("Refusing validation from a dirty Git worktree")
    if code["commit"] != args.expected_code_revision:
        raise RuntimeError("Code revision does not match --expected-code-revision")

    cfg = load_config(args.config)
    cfg.sae_collect.enabled = False
    task_ids = resolve_task_ids(cfg.env.task_ids, 10)
    if cfg.env.task_suite_name != "libero_spatial":
        raise RuntimeError("Development validation requires libero_spatial")
    if task_ids != list(range(10)) or int(cfg.env.num_trials_per_task) != 5:
        raise RuntimeError("Development validation requires 10 tasks and 5 trials/task")
    if not cfg.logging.save_actions or not cfg.logging.save_prompt_records:
        raise RuntimeError("Development validation requires action and prompt logging")
    expected_rollouts = 50
    expected_run_config = _protocol_config(cfg, task_ids)

    candidates_path = Path(args.candidates).expanduser().resolve()
    candidate_rows = _jsonl(candidates_path)
    event_row, random_row = select_validation_candidates(candidate_rows)
    feature_ids = {
        "event": int(event_row["feature_id"]),
        "random": int(random_row["feature_id"]),
    }
    sae_checkpoint = Path(args.sae_checkpoint).expanduser().resolve()
    sae_sha256 = _sha256(sae_checkpoint)
    root = (
        Path(args.work_dir).expanduser().resolve()
        / "intervention-development-validation"
    )
    root.mkdir(parents=True, exist_ok=True)
    plan = {
        "schema_version": "event_sae_intervention_development_plan_v1",
        "scope": "development_validation_not_paper_result",
        "code": code,
        "rollouts_per_condition": expected_rollouts,
        "initial_state_indices_per_task": list(range(5)),
        "candidate_source": {
            "path": str(candidates_path),
            "sha256": _sha256(candidates_path),
            "event_aligned_rank1": event_row,
            "random_alive_rank1": random_row,
        },
        "sae_checkpoint": str(sae_checkpoint),
        "sae_sha256": sae_sha256,
        "conditions": [item[0] for item in CONDITION_SPECS],
    }
    plan_path = root / "validation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    python = sys.executable
    config_path = str(Path(args.config).expanduser().resolve())
    result_paths = {}
    for name, mode, feature_kind, alpha in CONDITION_SPECS:
        condition_root = root / name
        result_path = condition_root / "result.json"
        result_paths[name] = result_path
        if result_path.is_file():
            print(f"RESUME: {result_path}")
            continue
        condition_root.mkdir(parents=True, exist_ok=True)
        if mode == "raw":
            command = [
                python,
                str(repo_root / "scripts/openvla/evaluate_policy.py"),
                "--config",
                config_path,
                "--mode",
                "raw",
                "--expected-rollouts",
                str(expected_rollouts),
                "--expected-code-revision",
                args.expected_code_revision,
                "--result-output",
                str(result_path),
                "--override",
                f"logging.root_dir={condition_root / 'runs'}",
            ]
        else:
            command = [
                python,
                str(repo_root / "scripts/openvla/intervene.py"),
                "--config",
                config_path,
                "--sae-checkpoint",
                str(sae_checkpoint),
                "--layer-idx",
                "31",
                "--feature-id",
                str(feature_ids[feature_kind]),
                "--alpha",
                str(alpha),
                "--hook-start-step",
                "0",
                "--expected-rollouts",
                str(expected_rollouts),
                "--expected-code-revision",
                args.expected_code_revision,
                "--result-output",
                str(result_path),
                "--override",
                f"logging.root_dir={condition_root / 'runs'}",
            ]
        _run(command, repo_root=repo_root)

    validated = {}
    for name, mode, feature_kind, alpha in CONDITION_SPECS:
        validated[name] = _validate_result(
            result_paths[name],
            condition_name=name,
            expected_mode=mode,
            expected_rollouts=expected_rollouts,
            expected_commit=args.expected_code_revision,
            expected_run_config=expected_run_config,
            expected_feature_id=(
                feature_ids[feature_kind] if feature_kind is not None else None
            ),
            expected_alpha=alpha,
            expected_sae_sha256=sae_sha256,
        )

    manifest_hashes = {
        name: value["manifest_sha256"] for name, value in validated.items()
    }
    if len(set(manifest_hashes.values())) != 1:
        raise RuntimeError(f"Conditions did not use identical initial states: {manifest_hashes}")
    baseline_outcomes = validated["raw"]["outcomes"]
    for name, value in validated.items():
        baseline_keys = {
            (key, row["initial_state_sha256"])
            for key, row in baseline_outcomes.items()
        }
        condition_keys = {
            (key, row["initial_state_sha256"])
            for key, row in value["outcomes"].items()
        }
        if baseline_keys != condition_keys:
            raise RuntimeError(f"Initial-state outcome evidence differs for {name}")

    action_comparisons = {
        name: compare_actions(validated["raw"]["actions"], validated[name]["actions"])
        for name in ("event-alpha1", "event-alpha0", "random-alpha0")
    }
    raw_success_vector = {
        key: bool(row["success"]) for key, row in baseline_outcomes.items()
    }
    alpha1_success_vector = {
        key: bool(row["success"])
        for key, row in validated["event-alpha1"]["outcomes"].items()
    }
    alpha1_identity_passed = (
        action_comparisons["event-alpha1"]["all_actions_match"]
        and raw_success_vector == alpha1_success_vector
    )
    if not alpha1_identity_passed:
        raise RuntimeError("alpha=1 identity control did not match raw policy")

    condition_summaries = {}
    for name, value in validated.items():
        result = value["result"]
        condition_summaries[name] = {
            "mode": result["mode"],
            "feature_id": result.get("feature_id"),
            "alpha": result.get("alpha"),
            "completed_rollouts": result["completed_rollouts"],
            "success_rate": result["success_rate"],
            "task_success_rates": summarize_task_success(value["outcomes"]),
            "hook_metrics": result.get("hook_metrics"),
            "result_path": str(result_paths[name]),
            "result_sha256": _sha256(result_paths[name]),
        }

    summary = {
        "schema_version": "event_sae_intervention_development_validation_v1",
        "scope": "development_validation_not_paper_result",
        "passed": True,
        "code": code,
        "shared_protocol": expected_run_config,
        "rollouts_per_condition": expected_rollouts,
        "total_rollouts": expected_rollouts * len(CONDITION_SPECS),
        "initial_state_manifest_sha256": next(iter(manifest_hashes.values())),
        "candidate_source": plan["candidate_source"],
        "sae_sha256": sae_sha256,
        "conditions": condition_summaries,
        "action_comparisons_against_raw": action_comparisons,
        "checks": {
            "identical_initial_states": True,
            "alpha1_identity_actions_and_outcomes": True,
            "event_alpha0_feature_zeroed": True,
            "random_alpha0_feature_zeroed": True,
            "event_alpha0_action_change_observed": (
                action_comparisons["event-alpha0"]["changed_episodes"] > 0
            ),
            "random_alpha0_action_change_observed": (
                action_comparisons["random-alpha0"]["changed_episodes"] > 0
            ),
        },
    }
    summary_path = root / "development_validation_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(json.dumps(summary["checks"], indent=2))
    print(f"INTERVENTION_DEVELOPMENT_VALIDATION_OK: {summary_path}")


if __name__ == "__main__":
    main()
