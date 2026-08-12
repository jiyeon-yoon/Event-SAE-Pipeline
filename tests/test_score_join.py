import json

import torch

from event_sae.scoring.rankings import alive_feature_ids
from event_sae.scoring.score_matrix import _load_timestep_vectors, join_cluster_events


def test_invalid_annotation_does_not_remove_authoritative_cluster():
    result = join_cluster_events(
        event_features=[
            {
                "sample_id": "s1",
                "task_description": "task",
                "task_id": 0,
                "task_episode_idx": 0,
                "episode_num": 1,
                "waypoint_rank": 0,
                "waypoint_step": 3,
                "progress_percent": 0.3,
                "num_steps": 10,
            }
        ],
        cluster_assignments=[
            {"sample_id": "s1", "task_description": "task", "cluster_id": "c0"}
        ],
        clusters=[
            {
                "cluster_id": "c0",
                "task_description": "task",
                "episode_coverage": 0.8,
            }
        ],
        cluster_annotations=[
            {
                "cluster_id": "c0",
                "task_description": "task",
                "api_error": "quota",
            }
        ],
    )
    assert len(result.selected_events) == 1
    assert result.selected_events[0]["cluster_id"] == "c0"
    assert result.cluster_metadata_by_id["c0"]["annotation_valid"] is False
    assert result.counts["clusters_without_valid_annotation"] == 1


def test_alive_feature_ids_excludes_zero_value_topk_ties(tmp_path):
    torch.save(
        {
            "top_feature_ids": torch.tensor([[5, 7], [9, 11]], dtype=torch.int32),
            "top_feature_vals": torch.tensor([[1.0, 0.0], [0.5, 0.0]]),
        },
        tmp_path / "shard_000000.pt",
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "token_topk_sparse_v1",
                "shards": [{"path": "shard_000000.pt"}],
            }
        ),
        encoding="utf-8",
    )
    assert alive_feature_ids(tmp_path) == {5, 9}


def test_openvla_streaming_score_loader_keeps_only_required_windows(tmp_path):
    def write_shard(name, episodes, steps, feature_ids, feature_values):
        torch.save(
            {
                "episode_num": torch.tensor(episodes),
                "step_in_episode": torch.tensor(steps),
                "top_feature_ids": torch.tensor(feature_ids),
                "top_feature_vals": torch.tensor(feature_values),
            },
            tmp_path / name,
        )

    write_shard("s0.pt", [1, 1], [0, 1], [[0], [1]], [[2.0], [4.0]])
    write_shard("s1.pt", [1, 2], [1, 0], [[1], [2]], [[8.0], [6.0]])
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "token_topk_sparse_v1",
                "dict_size": 3,
                "capture_target": None,
                "shards": [{"path": "s0.pt"}, {"path": "s1.pt"}],
            }
        ),
        encoding="utf-8",
    )
    vectors, task_means, task_counts, _, counters = _load_timestep_vectors(
        tmp_path,
        step_mapping="inference_step",
        episode_to_task_id={1: 0, 2: 0},
        task_id_set={0},
        dict_size=3,
        required_timestep_keys={(1, 1)},
    )
    assert set(vectors) == {(1, 1)}
    torch.testing.assert_close(vectors[(1, 1)], torch.tensor([0.0, 6.0, 0.0]))
    torch.testing.assert_close(task_means[0], torch.tensor([2 / 3, 2.0, 2.0]))
    assert task_counts == {0: 3}
    assert counters["timestep_vectors_retained"] == 1
    assert counters["timestep_vectors_total"] == 3
