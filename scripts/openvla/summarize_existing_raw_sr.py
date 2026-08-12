"""Summarize raw Spatial success from an existing merged collection run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-run-dir", required=True)
    parser.add_argument("--trials-per-task", type=int, default=10)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    run_dir = Path(args.merged_run_dir).expanduser().resolve()
    prompts = [
        json.loads(line)
        for line in (run_dir / "prompt_records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected = {
        int(row["episode_num"]): row
        for row in prompts
        if int(row["task_episode_idx"]) < args.trials_per_task
    }
    expected = 10 * args.trials_per_task
    if len(selected) != expected:
        raise RuntimeError(f"Selected {len(selected)} raw episodes, expected {expected}")

    final_done: dict[int, bool] = {}
    with (run_dir / "trajectory_records.jsonl").open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            episode = int(row["episode_num"])
            if episode in selected:
                final_done[episode] = bool(row["done"])
    if set(final_done) != set(selected):
        raise RuntimeError("Some selected raw episodes have no trajectory records")

    by_task: dict[int, list[bool]] = defaultdict(list)
    for episode, prompt in selected.items():
        by_task[int(prompt["task_id"])].append(final_done[episode])
    if set(by_task) != set(range(10)) or any(
        len(values) != args.trials_per_task for values in by_task.values()
    ):
        raise RuntimeError("Raw baseline does not contain the expected 10 Spatial tasks")
    successes = sum(sum(values) for values in by_task.values())
    result = {
        "schema_version": "event_sae_existing_raw_sr_v1",
        "merged_run_dir": str(run_dir),
        "selection": f"task_episode_idx < {args.trials_per_task}",
        "completed_rollouts": expected,
        "successes": successes,
        "success_rate": successes / expected,
        "per_task": {
            str(task_id): {
                "rollouts": len(by_task[task_id]),
                "successes": sum(by_task[task_id]),
                "success_rate": sum(by_task[task_id]) / len(by_task[task_id]),
            }
            for task_id in sorted(by_task)
        },
        "note": (
            "This reuses the matching first init-state trials from the existing raw collection; "
            "no additional raw-policy rollouts were run."
        ),
    }
    output = Path(args.output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2))
    print(f"RAW_SR_OK: {output}")


if __name__ == "__main__":
    main()
