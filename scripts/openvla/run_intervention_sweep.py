"""Resumably run the deduplicated Spatial intervention sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


EXPECTED_RANKINGS = ("event_aligned", "window_mean", "task_mean", "random_alive")


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


def _protocol_config(cfg, resolved_task_ids: list[int]) -> dict:
    payload = asdict(cfg)
    payload["logging"].pop("root_dir", None)
    payload["env"]["resolved_task_ids"] = list(resolved_task_ids)
    return payload


def _fingerprint_payload(result: dict) -> dict:
    mode = result.get("mode")
    keys = ["mode", "completed_rollouts", "run_config", "code"]
    if mode == "intervention":
        keys.extend(
            [
                "feature_id",
                "alpha",
                "layer_idx",
                "hook_start_step",
                "sae_sha256",
            ]
        )
    return {key: result.get(key) for key in keys}


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


def build_plan(candidates: list[dict]) -> tuple[list[int], dict[int, list[dict]]]:
    counts = Counter(str(row["ranking"]) for row in candidates)
    if counts != Counter({name: 5 for name in EXPECTED_RANKINGS}):
        raise RuntimeError(f"Expected five candidates per ranking, got {dict(counts)}")
    memberships: dict[int, list[dict]] = defaultdict(list)
    order: list[int] = []
    for row in candidates:
        feature_id = int(row["feature_id"])
        if not (0 <= feature_id < 32768):
            raise RuntimeError(f"Invalid feature id: {feature_id}")
        if feature_id not in memberships:
            order.append(feature_id)
        memberships[feature_id].append(row)
    return order, dict(memberships)


def _run(command: list[str], *, repo_root: Path) -> None:
    print("RUN:", " ".join(command), flush=True)
    env = os.environ.copy()
    old = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + old if old else "")
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def _validate_result(
    path: Path,
    *,
    expected_rollouts: int,
    expected_commit: str,
    feature_id: int | None,
    expected_run_config: dict,
    expected_sae_sha256: str | None = None,
    expected_alpha: float | None = None,
    expected_hook_start_step: int | None = None,
    expected_layer_idx: int = 31,
) -> dict:
    result = _json(path)
    if int(result.get("completed_rollouts", -1)) != expected_rollouts:
        raise RuntimeError(f"Invalid rollout count in {path}")
    if result.get("code", {}).get("commit") != expected_commit:
        raise RuntimeError(f"Code revision mismatch in {path}")
    if result.get("code", {}).get("dirty"):
        raise RuntimeError(f"Dirty-code result is not accepted: {path}")
    if result.get("run_config") != expected_run_config:
        raise RuntimeError(f"Run configuration mismatch in {path}")
    recorded_fingerprint = result.get("protocol_fingerprint")
    if recorded_fingerprint != _fingerprint(_fingerprint_payload(result)):
        raise RuntimeError(f"Protocol fingerprint mismatch in {path}")
    if feature_id is None:
        if result.get("mode") != "raw":
            raise RuntimeError(f"Expected raw baseline in {path}")
    else:
        if result.get("mode") != "intervention":
            raise RuntimeError(f"Expected intervention result in {path}")
        if int(result.get("feature_id", -1)) != feature_id:
            raise RuntimeError(f"Feature mismatch in {path}")
        expected_fields = {
            "sae_sha256": expected_sae_sha256,
            "alpha": expected_alpha,
            "hook_start_step": expected_hook_start_step,
            "layer_idx": expected_layer_idx,
        }
        mismatches = {
            key: (result.get(key), expected)
            for key, expected in expected_fields.items()
            if result.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(f"Intervention protocol mismatch in {path}: {mismatches}")
        hook = result.get("hook_metrics", {})
        if int(hook.get("num_forwards", 0)) <= 0:
            raise RuntimeError(f"Intervention hook did not fire in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sae-checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--hook-start-step", type=int, default=0)
    args = parser.parse_args()

    # Lazy import keeps --help usable on a laptop without LIBERO.
    from event_sae.openvla.eval.config import load_config, resolve_task_ids

    repo_root = Path(__file__).resolve().parents[2]
    code = _git_provenance(repo_root)
    if code["dirty"]:
        raise RuntimeError("Refusing expensive rollouts from a dirty Git worktree")
    if code["commit"] != args.expected_code_revision:
        raise RuntimeError(
            f"Code revision {code['commit']} does not match --expected-code-revision"
        )
    cfg = load_config(args.config)
    cfg.sae_collect.enabled = False
    task_ids = resolve_task_ids(cfg.env.task_ids, 10)
    expected_rollouts = len(task_ids) * int(cfg.env.num_trials_per_task)
    if cfg.env.task_suite_name != "libero_spatial":
        raise RuntimeError("This sweep is restricted to libero_spatial")
    if len(task_ids) != 10 or int(cfg.env.num_trials_per_task) != 50:
        raise RuntimeError(
            "Paper main intervention protocol requires all 10 tasks and 50 trials/task"
        )
    expected_run_config = _protocol_config(cfg, task_ids)
    sae_checkpoint = Path(args.sae_checkpoint).expanduser().resolve()
    sae_sha256 = _sha256(sae_checkpoint)
    candidates_path = Path(args.candidates).expanduser().resolve()
    candidates = _jsonl(candidates_path)
    feature_order, memberships = build_plan(candidates)
    total_conditions = 1 + len(feature_order)
    print(
        json.dumps(
            {
                "candidate_rows": len(candidates),
                "unique_features": len(feature_order),
                "rollouts_per_condition": expected_rollouts,
                "conditions_including_raw_baseline": total_conditions,
                "total_rollouts_for_full_sweep": total_conditions * expected_rollouts,
                "feature_order": feature_order,
            },
            indent=2,
        )
    )
    work = Path(args.work_dir).expanduser().resolve() / "intervention"
    work.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    baseline_path = work / "baseline.json"
    if not baseline_path.is_file():
        _run(
            [
                python,
                str(repo_root / "scripts/openvla/evaluate_policy.py"),
                "--config",
                str(Path(args.config).resolve()),
                "--mode",
                "raw",
                "--expected-rollouts",
                str(expected_rollouts),
                "--expected-code-revision",
                args.expected_code_revision,
                "--result-output",
                str(baseline_path),
                "--override",
                f"logging.root_dir={work / 'runs' / 'baseline'}",
            ],
            repo_root=repo_root,
        )
    baseline = _validate_result(
        baseline_path,
        expected_rollouts=expected_rollouts,
        expected_commit=args.expected_code_revision,
        feature_id=None,
        expected_run_config=expected_run_config,
    )

    results: dict[int, dict] = {}
    for feature_id in feature_order:
        feature_root = work / f"feature-{feature_id}"
        result_path = feature_root / "result.json"
        if not result_path.is_file():
            _run(
                [
                    python,
                    str(repo_root / "scripts/openvla/intervene.py"),
                    "--config",
                    str(Path(args.config).resolve()),
                    "--sae-checkpoint",
                    str(Path(args.sae_checkpoint).resolve()),
                    "--layer-idx",
                    "31",
                    "--feature-id",
                    str(feature_id),
                    "--alpha",
                    str(args.alpha),
                    "--hook-start-step",
                    str(args.hook_start_step),
                    "--expected-rollouts",
                    str(expected_rollouts),
                    "--expected-code-revision",
                    args.expected_code_revision,
                    "--result-output",
                    str(result_path),
                    "--override",
                    f"logging.root_dir={feature_root / 'runs'}",
                ],
                repo_root=repo_root,
            )
        results[feature_id] = _validate_result(
            result_path,
            expected_rollouts=expected_rollouts,
            expected_commit=args.expected_code_revision,
            feature_id=feature_id,
            expected_run_config=expected_run_config,
            expected_sae_sha256=sae_sha256,
            expected_alpha=args.alpha,
            expected_hook_start_step=args.hook_start_step,
        )

    baseline_sr = float(baseline["success_rate"])
    completed_rows = []
    for feature_id, result in results.items():
        sr = float(result["success_rate"])
        completed_rows.append(
            {
                "feature_id": feature_id,
                "success_rate": sr,
                "delta_from_raw": sr - baseline_sr,
                "active_feature_values": int(
                    result["hook_metrics"]["active_feature_values"]
                ),
                "ranking_memberships": memberships[feature_id],
                "result_path": str(work / f"feature-{feature_id}" / "result.json"),
            }
        )
    summary = {
        "schema_version": "event_sae_intervention_sweep_v1",
        "code": code,
        "candidate_path": str(candidates_path),
        "raw_baseline_success_rate": baseline_sr,
        "rollouts_per_condition": expected_rollouts,
        "candidate_rows": len(candidates),
        "unique_features_total": len(feature_order),
        "features_completed": len(results),
        "alpha": args.alpha,
        "hook_start_step": args.hook_start_step,
        "results": completed_rows,
    }
    summary_path = work / "sweep_summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(summary_path)
    print(f"INTERVENTION_SWEEP_OK: {summary_path}")


if __name__ == "__main__":
    main()
