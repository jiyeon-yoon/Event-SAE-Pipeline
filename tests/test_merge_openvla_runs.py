import csv
import json
from pathlib import Path

from event_sae.openvla.merge_runs import MergeConfig, merge_openvla_runs


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def _make_run(root: Path, task_id: int) -> None:
    root.mkdir()
    description = f"task {task_id}"
    prompts = []
    trajectories = []
    index = []
    actions = {description: {}}
    for local_episode, trial in ((1, 0), (2, 1)):
        prompts.append(
            {
                "episode_num": local_episode,
                "task_id": task_id,
                "task_episode_idx": trial,
                "task_description": description,
            }
        )
        actions[description][str(trial)] = [[0.0] * 7, [1.0] * 7]
        for step in range(2):
            trajectories.append(
                {
                    "episode_num": local_episode,
                    "task_id": task_id,
                    "task_episode_idx": trial,
                    "task_description": description,
                    "step_in_episode": step,
                    "eef_pos": [0.0, 0.0, float(step)],
                    "done": step == 1,
                }
            )
        index.append(
            {
                "layer_idx": 31,
                "row_start": (local_episode - 1) * 2,
                "row_end": local_episode * 2,
                "episode_num": local_episode,
                "step_in_episode": local_episode - 1,
                "task_id": task_id,
                "task_episode_idx": trial,
                "global_forward_idx": local_episode,
                "shard_path": "layer_31_shard_000000.pt",
            }
        )
        videos = root / "videos"
        videos.mkdir(exist_ok=True)
        (videos / f"run--episode={local_episode}--task={task_id}.mp4").write_bytes(b"video")

    _write_jsonl(root / "prompt_records.jsonl", prompts)
    _write_jsonl(root / "trajectory_records.jsonl", trajectories)
    (root / "actions.json").write_text(json.dumps(actions), encoding="utf-8")
    with (root / "events.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Task Description", "Task Success Rate"])
        writer.writeheader()
        writer.writerow({"Task Description": description, "Task Success Rate": "1.0"})
    dense = root / "sae_activations" / "post_mlp_residual"
    dense.mkdir(parents=True)
    (dense / "layer_31_shard_000000.pt").write_bytes(b"tensor")
    _write_jsonl(dense / "activation_index.jsonl", index)


def test_merge_rewrites_every_episode_reference(tmp_path: Path):
    run0, run1 = tmp_path / "run0", tmp_path / "run1"
    _make_run(run0, 0)
    _make_run(run1, 1)
    output = tmp_path / "merged"
    result = merge_openvla_runs(
        MergeConfig(
            input_dirs=(str(run0), str(run1)),
            output_dir=str(output),
            trials_per_task=2,
            expected_tasks=2,
            expected_shards=2,
        )
    )

    prompts = [json.loads(line) for line in (output / "prompt_records.jsonl").read_text().splitlines()]
    trajectories = [
        json.loads(line) for line in (output / "trajectory_records.jsonl").read_text().splitlines()
    ]
    activation_index = [
        json.loads(line)
        for line in (
            output / "sae_activations" / "post_mlp_residual" / "activation_index.jsonl"
        ).read_text().splitlines()
    ]
    assert [row["episode_num"] for row in prompts] == [1, 2, 3, 4]
    assert sorted({row["episode_num"] for row in trajectories}) == [1, 2, 3, 4]
    assert sorted({row["episode_num"] for row in activation_index}) == [1, 2, 3, 4]
    assert [row["global_forward_idx"] for row in activation_index] == [1, 2, 3, 4]
    assert len(list((output / "videos").glob("*.mp4"))) == 4
    assert len(list((output / "videos").glob("*episode=3--*.mp4"))) == 1
    for record in activation_index:
        linked = output / "sae_activations" / "post_mlp_residual" / record["shard_path"]
        assert linked.is_symlink()
        assert linked.read_bytes() == b"tensor"
    assert result["num_episodes"] == 4
    assert result["num_activation_shards"] == 2
    assert json.loads((output / "merge_manifest.json").read_text())["num_videos"] == 4
