import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.config import (
    load_extended_config,
    parse_overrides,
)
from event_sae.openvla.extended_collection.writer import (
    ExtendedRunWriter,
    array_sha256,
)


def test_config_is_layer31_only(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("collection:\n  layer_idx: 31\n", encoding="utf-8")
    cfg = load_extended_config(
        config_path,
        parse_overrides(["env.task_ids=0,1", "output.save_video=false"]),
    )
    assert cfg.collection.layer_idx == 31
    assert cfg.env.task_ids == "0,1"
    assert cfg.output.save_video is False

    with pytest.raises(ValueError, match="fixed to layer 31"):
        load_extended_config(config_path, parse_overrides(["collection.layer_idx=16"]))

    with pytest.raises(ValueError, match="requires every low-volume telemetry stream"):
        load_extended_config(
            config_path,
            parse_overrides(["collection.save_contacts_and_grasp=false"]),
        )


def test_writer_preserves_pre_action_post_alignment(tmp_path: Path):
    with ExtendedRunWriter(tmp_path / "run") as writer:
        prompt = writer.begin_episode(
            prompt_record={"episode_num": 1, "task_id": 0, "task_episode_idx": 0},
            initial_state=np.asarray([1.0, 2.0], dtype=np.float32),
        )
        assert prompt["initial_state_sha256"] == array_sha256(
            np.asarray([1.0, 2.0], dtype=np.float32)
        )
        writer.write_step(
            common={"episode_num": 1, "task_id": 0, "step_in_episode": 0},
            pre_json={"eef": [0, 0, 0]},
            post_json={"eef": [1, 0, 0]},
            pre_vectors={"qpos": np.asarray([0.0, 1.0])},
            post_vectors={"qpos": np.asarray([0.1, 1.1])},
            raw_action=np.zeros(7),
            executed_action=np.ones(7),
            policy={"entropy_mean_nats": 1.0},
            model_input_rgb=np.zeros((2, 2, 3), dtype=np.uint8),
            reward=0.0,
            done=False,
            info={},
        )
        writer.finish_episode(
            result={"episode_num": 1, "success": False}, compress_npz=True
        )

    trajectory = json.loads(
        (tmp_path / "run" / "trajectory_records.jsonl").read_text().strip()
    )
    assert trajectory["alignment"] == "pre_state + action -> post_state"
    sim = np.load(tmp_path / "run" / "sim_state" / "episode_000001.npz")
    assert sim["pre_qpos"].shape == (1, 2)
    assert sim["post_qpos"].tolist() == [[0.1, 1.1]]
    vision = np.load(tmp_path / "run" / "vision" / "episode_000001.npz")
    assert vision["model_input_rgb"].shape == (1, 2, 2, 3)
