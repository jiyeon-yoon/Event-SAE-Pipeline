import json
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.activation import Layer31ActivationCollector


def test_layer31_collector_writes_all_seven_policy_forwards(tmp_path: Path):
    model = SimpleNamespace(
        language_model=SimpleNamespace(
            model=SimpleNamespace(
                layers=torch.nn.ModuleList([torch.nn.Identity() for _ in range(32)])
            )
        ),
        _sae_hook_context={
            "episode_num": 1,
            "task_id": 0,
            "task_episode_idx": 0,
            "task_description": "test",
            "step_in_episode": 0,
        },
    )
    collector = Layer31ActivationCollector(model, tmp_path, flush_every=3)
    for _ in range(7):
        model.language_model.model.layers[31](torch.zeros((1, 1, 4096)))
    collector.close()
    collector.close()

    records = [
        json.loads(line)
        for line in (tmp_path / "activation_index.jsonl").read_text().splitlines()
    ]
    assert len(records) == 7
    assert [row["global_forward_idx"] for row in records] == list(range(1, 8))
    assert {row["step_in_episode"] for row in records} == {0}
    assert (
        sum(
            torch.load(path, map_location="cpu", weights_only=True).shape[0]
            for path in sorted(tmp_path.glob("layer_31_shard_*.pt"))
        )
        == 7
    )
