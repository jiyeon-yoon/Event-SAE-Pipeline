import json
import sys
import types
from pathlib import Path

import numpy as np
import torch

# The clustering test does not call Gemini, but event_sae.events currently
# re-exports its optional annotation module at package import time.
if "google.genai" not in sys.modules:
    genai_stub = types.ModuleType("google.genai")
    genai_stub.types = types.SimpleNamespace()
    sys.modules["google.genai"] = genai_stub

from event_sae.events.cluster import cluster_events
from event_sae.events.io import load_jsonl
from scripts.extract_topk import (
    _aggregate_fidelity_shards,
    _encode_topk_and_fidelity_in_batches,
    _encode_topk_in_batches,
)


class _IdentitySAE:
    def encode(self, value):
        return value


class _IdentityReconstructionSAE:
    threshold = torch.tensor(0.0)

    def __call__(self, value, *, output_features=False):
        assert output_features
        return value, value


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_encode_topk_batches_matches_single_batch():
    dense = torch.tensor(
        [[1.0, 4.0, 2.0, 3.0], [8.0, 5.0, 7.0, 6.0], [0.0, -1.0, 2.0, 1.0]]
    )
    expected_values, expected_indices = torch.topk(dense, k=2, dim=-1)
    values, indices = _encode_topk_in_batches(
        _IdentitySAE(), dense, device="cpu", topk=2, batch_size=2
    )
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(indices.to(torch.int64), expected_indices)


def test_combined_topk_and_fidelity_is_exact_for_identity(tmp_path: Path):
    dense = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0], [3.0, 4.0, 0.0, 0.0], [5.0, 0.0, 6.0, 7.0]]
    )
    values, indices, stats = _encode_topk_and_fidelity_in_batches(
        _IdentityReconstructionSAE(),
        dense,
        device="cpu",
        topk=2,
        batch_size=2,
        dict_size=4,
    )
    expected_values, expected_indices = torch.topk(dense, k=2, dim=-1)
    torch.testing.assert_close(values, expected_values)
    torch.testing.assert_close(indices.to(torch.int64), expected_indices)

    stats_path = tmp_path / "fidelity_shard_000000.pt"
    torch.save(stats, stats_path)
    checkpoint_path = tmp_path / "ae.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    output_path = tmp_path / "fidelity.json"
    manifest = {
        "sae_sha256": "test-hash",
        "dense_dir": str(tmp_path),
        "num_shards": 1,
        "shards": [{"fidelity_path": stats_path.name}],
    }
    result = _aggregate_fidelity_shards(
        output_dir=tmp_path,
        manifest=manifest,
        fidelity_output=output_path,
        trainer_cfg={
            "activation_dim": 4,
            "dict_size": 4,
            "layer": 31,
            "submodule_name": "post_mlp_residual",
            "k": 2,
        },
        checkpoint_path=checkpoint_path,
        sae=_IdentityReconstructionSAE(),
        batch_size=2,
        device="cpu",
    )
    assert result["data"]["num_rows"] == 3
    assert result["metrics"]["frac_variance_explained"] == 1.0
    assert result["metrics"]["reconstruction_mse"] == 0.0
    assert result["metrics"]["fraction_alive"] == 1.0
    assert np.isclose(result["metrics"]["average_l0"], 7 / 3)


def test_cluster_coverage_uses_all_prompt_episodes(tmp_path: Path):
    event_path = tmp_path / "event_features.jsonl"
    prompt_path = tmp_path / "prompt_records.jsonl"
    out = tmp_path / "clusters"
    base = {
        "task_description": "task zero",
        "task_id": 0,
        "task_episode_idx": 0,
        "waypoint_rank": 0,
        "waypoint_step": 1,
        "progress_percent": 0.5,
        "num_steps": 10,
        "vision_embedding": [1.0, 0.0],
        "state_vector": [0.0, 1.0],
        "clip_path": "clip.mp4",
        "frame_paths": ["frame.png"],
    }
    rows = []
    for episode in (1, 2):
        row = dict(base)
        row.update(sample_id=f"sample-{episode}", episode_num=episode)
        rows.append(row)
    _write_jsonl(event_path, rows)
    _write_jsonl(
        prompt_path,
        [
            {
                "episode_num": episode,
                "task_id": 0,
                "task_episode_idx": episode - 1,
                "task_description": "task zero",
            }
            for episode in range(1, 5)
        ],
    )

    cluster_events(
        event_path,
        out,
        prompt_records_path=prompt_path,
        distance_threshold=0.18,
        min_coverage=0.75,
    )
    clusters = load_jsonl(out / "clusters.jsonl")
    assert len(clusters) == 1
    assert clusters[0]["total_task_episodes"] == 4
    assert np.isclose(clusters[0]["episode_coverage"], 0.5)
    assert clusters[0]["is_canonical"] is False
