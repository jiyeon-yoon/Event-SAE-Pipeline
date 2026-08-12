"""Run the validated Event-SAE discovery pipeline on Spatial-500 artifacts.

This command never starts new LIBERO rollouts. It reuses the five collected
dataset parts and the already-trained layer-31 SAE. Existing valid stage
outputs are skipped; incomplete outputs stop the run instead of being deleted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch


STAGES = (
    "merge",
    "keyframes",
    "media",
    "topk_fidelity",
    "event_features",
    "clusters",
    "scores",
    "rankings",
)


def _git_provenance(repo_root: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), *args], text=True
        ).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "remote": git("remote", "get-url", "origin"),
    }


def _require_free_space(path: Path, min_free_gib: float) -> dict:
    usage = shutil.disk_usage(path)
    free_gib = usage.free / (1024**3)
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"Only {free_gib:.1f} GiB free under {path}; "
            f"at least {min_free_gib:.1f} GiB is required before GPU processing"
        )
    return {
        "path": str(path),
        "total_gib": usage.total / (1024**3),
        "used_gib": usage.used / (1024**3),
        "free_gib": free_gib,
        "required_free_gib": min_free_gib,
    }


def _jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run(stage: str, command: list[str], log_dir: Path, *, dry_run: bool) -> None:
    print(f"\n[{stage}] {' '.join(command)}", flush=True)
    if dry_run:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    repo_root = Path(__file__).resolve().parents[2]
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=child_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Stage {stage} failed with exit code {return_code}; see {log_path}")


def _validate_merge(run_dir: Path) -> dict:
    manifest = _json(run_dir / "merge_manifest.json")
    expected = {
        "num_episodes": 500,
        "num_videos": 500,
        "num_activation_shards": 369,
    }
    for key, value in expected.items():
        if int(manifest[key]) != value:
            raise RuntimeError(f"Merged {key}={manifest[key]}, expected {value}")
    if manifest["task_episode_counts"] != {str(task): 50 for task in range(10)}:
        raise RuntimeError("Merged task counts are not exactly 50 for tasks 0..9")
    return manifest


def _validate_keyframes(path: Path) -> dict:
    summary = _json(path)
    if int(summary["num_selected_episodes"]) != 500 or len(summary["episodes"]) != 500:
        raise RuntimeError("AWE output does not contain all 500 episodes")
    waypoint_count = sum(int(item["num_waypoints"]) for item in summary["episodes"])
    if waypoint_count <= 0:
        raise RuntimeError("AWE produced no waypoints")
    return {"episodes": 500, "waypoints": waypoint_count, "mean": waypoint_count / 500}


def _validate_media(report_path: Path, expected_samples: int) -> dict:
    report = _json(report_path)
    if int(report["num_skipped"]) != 0:
        raise RuntimeError(f"Media packaging skipped {report['num_skipped']} samples")
    if int(report["num_samples"]) != expected_samples:
        raise RuntimeError(
            f"Media samples={report['num_samples']}, expected {expected_samples}"
        )
    if len(_jsonl(Path(report["samples_path"]))) != expected_samples:
        raise RuntimeError("samples.jsonl line count does not match packaging report")
    return {"samples": expected_samples, "skipped": 0}


def _validate_topk(topk_dir: Path, fidelity_path: Path) -> dict:
    manifest = _json(topk_dir / "manifest.json")
    fidelity = _json(fidelity_path)
    if int(manifest["num_shards"]) != 369 or len(manifest["shards"]) != 369:
        raise RuntimeError("Sparse Top-K output does not contain 369 shards")
    if int(fidelity["data"]["num_shards_evaluated"]) != 369:
        raise RuntimeError("Fidelity output did not evaluate all 369 shards")
    if int(fidelity["data"]["num_rows"]) != int(manifest["total_rows"]):
        raise RuntimeError("Fidelity and Top-K row counts differ")
    metrics = fidelity["metrics"]
    if not all(math.isfinite(float(value)) for value in metrics.values()):
        raise RuntimeError("Fidelity output contains non-finite metrics")
    return {"shards": 369, "rows": int(manifest["total_rows"]), **metrics}


def _validate_event_features(samples_path: Path, features_path: Path) -> dict:
    samples = len(_jsonl(samples_path))
    features = len(_jsonl(features_path))
    if samples != features:
        raise RuntimeError(f"event_features={features}, samples={samples}")
    return {"event_features": features}


def _validate_clusters(cluster_dir: Path) -> dict:
    summary = _json(cluster_dir / "summary.json")
    clusters = _jsonl(cluster_dir / "clusters.jsonl")
    if int(summary["num_tasks"]) != 10:
        raise RuntimeError(f"Cluster output has {summary['num_tasks']} tasks, expected 10")
    if summary.get("coverage_denominator") != "all_attempted_episodes":
        raise RuntimeError("Cluster coverage did not use all attempted episodes")
    recurring = sum(bool(item["is_canonical"]) for item in clusters)
    recurring_tasks = len(
        {str(item["task_description"]) for item in clusters if bool(item["is_canonical"])}
    )
    return {
        "clusters": len(clusters),
        "recurring_clusters": recurring,
        "tasks_with_recurring_cluster": recurring_tasks,
        "paper_reference_clusters": 48,
        "paper_reference_recurring": 36,
    }


def _validate_scores(path: Path, *, expected_rows: int) -> dict:
    payload = torch.load(path, map_location="cpu")
    counts = payload["selection_counts"]
    if int(counts["skipped_missing_window_vectors"]) != 0:
        raise RuntimeError(
            f"Scoring missed {counts['skipped_missing_window_vectors']} event windows"
        )
    if tuple(payload["matrix_raw"].shape)[1] != 32768:
        raise RuntimeError("Score matrix dictionary dimension is not 32768")
    rows = int(payload["matrix_raw"].shape[0])
    if rows != expected_rows:
        raise RuntimeError(
            f"Score rows={rows}, but recurring cluster count={expected_rows}"
        )
    selected = int(counts["selected_events_after_activation_filter"])
    if selected <= 0 or len(payload["selected_events"]) != selected:
        raise RuntimeError("Score artifact contains no valid selected events")
    task_counts = {
        int(task_id): int(count)
        for task_id, count in counts.get("task_timestep_counts", {}).items()
    }
    if set(task_counts) != set(range(10)) or any(count <= 0 for count in task_counts.values()):
        raise RuntimeError(
            f"Score artifact must contain positive timestep counts for tasks 0..9: {task_counts}"
        )
    return {
        "rows": rows,
        "selected_events": selected,
        "task_timestep_counts": task_counts,
        "skipped_missing_window_vectors": 0,
    }


def _validate_rankings(path: Path) -> dict:
    candidates = _jsonl(path)
    counts = Counter(str(item["ranking"]) for item in candidates)
    expected = {name: 5 for name in ("event_aligned", "window_mean", "task_mean", "random_alive")}
    if dict(counts) != expected or len(candidates) != 20:
        raise RuntimeError(f"Invalid ranking candidates: {dict(counts)}")
    unique = len({int(item["feature_id"]) for item in candidates})
    return {"candidate_rows": 20, "unique_feature_ids": unique, "ranking_counts": dict(counts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, help="Output root from the download command")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--expected-code-revision", required=True)
    parser.add_argument("--min-free-gib", type=float, default=60.0)
    parser.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    start = STAGES.index(args.start_at)
    stop = STAGES.index(args.stop_after)
    if start > stop:
        parser.error("--start-at must not come after --stop-after")
    selected = set(STAGES[start : stop + 1])
    required = set(STAGES[: stop + 1])
    repo_root = Path(__file__).resolve().parents[2]
    code = _git_provenance(repo_root)
    if code["dirty"]:
        raise RuntimeError("Refusing to run expensive processing from a dirty Git worktree")
    if code["commit"] != args.expected_code_revision:
        raise RuntimeError(
            f"Code revision {code['commit']} does not match "
            f"--expected-code-revision {args.expected_code_revision}"
        )
    python = sys.executable
    input_root = Path(args.input_root).expanduser().resolve()
    input_manifest = _json(input_root / "input_manifest.json")
    input_dirs = [Path(item["local_dir"]) for item in input_manifest["pairs"]]
    checkpoint = Path(input_manifest["sae"]["checkpoint_path"])
    work = Path(args.work_dir).expanduser().resolve()
    run = work / "merged"
    pipeline = work / "pipeline"
    logs = work / "logs"
    keyframes = pipeline / "keyframes" / "waypoint_summary.json"
    events = pipeline / "events"
    samples = events / "samples.jsonl"
    event_features = events / "event_features.jsonl"
    topk = pipeline / "topk"
    fidelity = pipeline / "offline_fidelity.json"
    clusters = pipeline / "clusters"
    scores = pipeline / "scores" / "event_feature_scores.pt"
    rankings = pipeline / "rankings" / "candidates.jsonl"
    work.mkdir(parents=True, exist_ok=True)
    disk_preflight = _require_free_space(work, args.min_free_gib)
    vision_model = input_manifest["vision_model"]
    stage_results: dict[str, dict] = {}
    started = time.time()

    if "merge" in selected and not (run / "merge_manifest.json").is_file():
        command = [python, str(repo_root / "scripts/openvla/merge_runs.py")]
        for path in input_dirs:
            command += ["--input-dir", str(path)]
        command += [
            "--output-dir", str(run), "--trials-per-task", "50",
            "--expected-tasks", "10", "--expected-shards", "369", "--link-mode", "symlink",
        ]
        _run("merge", command, logs, dry_run=args.dry_run)
    if not args.dry_run and "merge" in required:
        stage_results["merge"] = _validate_merge(run)

    if "keyframes" in selected and not keyframes.is_file():
        _run(
            "keyframes",
            [
                python, str(repo_root / "scripts/extract_keyframes.py"),
                "--trajectory-records-path", str(run / "trajectory_records.jsonl"),
                "--output-dir", str(keyframes.parent), "--waypoint-mode", "pos_only",
                "--err-threshold", "0.05", "--success-filter", "all",
            ],
            logs,
            dry_run=args.dry_run,
        )
    if not args.dry_run and "keyframes" in required:
        stage_results["keyframes"] = _validate_keyframes(keyframes)

    if "media" in selected and not (events / "packaging_report.json").is_file():
        _run(
            "media",
            [
                python, str(repo_root / "scripts/extract_keyframe_media.py"),
                "--waypoint-summary-path", str(keyframes), "--output-dir", str(events),
                "--frame-offsets", "-4", "-2", "0", "2", "4",
            ],
            logs,
            dry_run=args.dry_run,
        )
    if not args.dry_run and "media" in required:
        stage_results["media"] = _validate_media(
            events / "packaging_report.json", stage_results["keyframes"]["waypoints"]
        )

    if "topk_fidelity" in selected:
        command = [
            python, str(repo_root / "scripts/extract_topk.py"),
            "--dense-dir", str(run / "sae_activations/post_mlp_residual"),
            "--sae-checkpoint", str(checkpoint), "--layer-idx", "31", "--topk", "64",
            "--batch-size", str(args.batch_size), "--output-dir", str(topk),
            "--device", args.device, "--fidelity-output", str(fidelity),
        ]
        if (topk / "manifest.json").is_file():
            command.append("--resume")
        _run("topk_fidelity", command, logs, dry_run=args.dry_run)
    if not args.dry_run and "topk_fidelity" in required:
        stage_results["topk_fidelity"] = _validate_topk(topk, fidelity)

    if "event_features" in selected and not event_features.is_file():
        _run(
            "event_features",
            [
                python, str(repo_root / "scripts/build_event_features.py"),
                "--samples-path", str(samples), "--output-path", str(event_features),
                "--vision-model-name-or-path", str(vision_model["repo_id"]),
                "--vision-model-revision", str(vision_model["revision"]),
                "--device", args.device,
            ],
            logs,
            dry_run=args.dry_run,
        )
    if not args.dry_run and "event_features" in required:
        stage_results["event_features"] = _validate_event_features(samples, event_features)

    if "clusters" in selected and not (clusters / "summary.json").is_file():
        _run(
            "clusters",
            [
                python, str(repo_root / "scripts/cluster_events.py"),
                "--event-features-path", str(event_features),
                "--prompt-records-path", str(run / "prompt_records.jsonl"),
                "--output-dir", str(clusters), "--vision-weight", "1.0",
                "--state-weight", "0.5", "--progress-weight", "0.4",
                "--distance-threshold", "0.18", "--min-coverage", "0.5",
                "--num-exemplars", "5",
            ],
            logs,
            dry_run=args.dry_run,
        )
    if not args.dry_run and "clusters" in required:
        stage_results["clusters"] = _validate_clusters(clusters)

    if "scores" in selected and not scores.is_file():
        _run(
            "scores",
            [
                python, str(repo_root / "scripts/score_cluster_features.py"),
                "--topk-run-dir", str(topk), "--event-features-path", str(event_features),
                "--cluster-assignments-path", str(clusters / "cluster_assignments.jsonl"),
                "--clusters-path", str(clusters / "clusters.jsonl"),
                "--prompt-records-path", str(run / "prompt_records.jsonl"),
                "--output-path", str(scores), "--window-size", "5", "--top-n", "20",
                "--step-mapping", "inference_step",
            ],
            logs,
            dry_run=args.dry_run,
        )
    if not args.dry_run and "scores" in required:
        stage_results["scores"] = _validate_scores(
            scores,
            # The score artifact retains every task-local cluster. Canonical
            # coverage filtering happens when suite-level rankings are built.
            expected_rows=stage_results["clusters"]["clusters"],
        )

    if "rankings" in selected and not rankings.is_file():
        _run(
            "rankings",
            [
                python, str(repo_root / "scripts/build_feature_rankings.py"),
                "--scores-pt", str(scores), "--topk-run-dir", str(topk),
                "--output-dir", str(rankings.parent), "--top-k", "5",
                "--top-n-per-row", "20", "--seed", "0", "--min-coverage", "0.5",
            ],
            logs,
            dry_run=args.dry_run,
        )
    if not args.dry_run and "rankings" in required:
        stage_results["rankings"] = _validate_rankings(rankings)
        result = {
            "schema_version": "event_sae_spatial_500_discovery_v1",
            "input_manifest": str(input_root / "input_manifest.json"),
            "work_dir": str(work),
            "stages": stage_results,
            "elapsed_seconds_this_invocation": time.time() - started,
            "scope": "Spatial-500 layer-31 discovery/ranking; no new rollout intervention",
            "code": code,
            "disk_preflight": disk_preflight,
            "vision_model": vision_model,
        }
        result_path = work / "pipeline_summary.json"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(stage_results, indent=2))
        print(f"DISCOVERY_PIPELINE_OK: {result_path}")


if __name__ == "__main__":
    main()
