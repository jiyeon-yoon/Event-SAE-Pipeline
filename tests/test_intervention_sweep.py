import json

import pytest

from scripts.openvla.run_intervention_sweep import (
    _fingerprint,
    _fingerprint_payload,
    _validate_result,
    build_plan,
)


def _candidates():
    rows = []
    for ranking_index, ranking in enumerate(
        ("event_aligned", "window_mean", "task_mean", "random_alive")
    ):
        for rank in range(5):
            # Feature 7 is shared by every ranking; all other ids are unique.
            feature_id = 7 if rank == 0 else ranking_index * 10 + rank
            rows.append(
                {
                    "ranking": ranking,
                    "rank": rank + 1,
                    "feature_id": feature_id,
                }
            )
    return rows


def test_build_plan_deduplicates_features_across_rankings():
    order, memberships = build_plan(_candidates())
    assert order[0] == 7
    assert len(order) == 17
    assert len(memberships[7]) == 4


def test_build_plan_requires_five_candidates_per_ranking():
    with pytest.raises(RuntimeError, match="five candidates"):
        build_plan(_candidates()[:-1])


def _intervention_result(run_config: dict) -> dict:
    result = {
        "mode": "intervention",
        "completed_rollouts": 500,
        "feature_id": 17,
        "alpha": 0.0,
        "layer_idx": 31,
        "hook_start_step": 0,
        "sae_sha256": "sae-hash",
        "run_config": run_config,
        "code": {"commit": "source", "dirty": False},
        "hook_metrics": {"num_forwards": 1},
    }
    result["protocol_fingerprint"] = _fingerprint(_fingerprint_payload(result))
    return result


def test_resume_validation_rejects_changed_intervention_protocol(tmp_path):
    run_config = {"model": {"revision": "weights"}, "env": {"seed": 0}}
    result = _intervention_result(run_config)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")

    validated = _validate_result(
        path,
        expected_rollouts=500,
        expected_commit="source",
        feature_id=17,
        expected_run_config=run_config,
        expected_sae_sha256="sae-hash",
        expected_alpha=0.0,
        expected_hook_start_step=0,
    )
    assert validated["feature_id"] == 17

    with pytest.raises(RuntimeError, match="protocol mismatch"):
        _validate_result(
            path,
            expected_rollouts=500,
            expected_commit="source",
            feature_id=17,
            expected_run_config=run_config,
            expected_sae_sha256="sae-hash",
            expected_alpha=0.5,
            expected_hook_start_step=0,
        )


def test_resume_validation_rejects_tampered_fingerprint(tmp_path):
    run_config = {"model": {"revision": "weights"}, "env": {"seed": 0}}
    result = _intervention_result(run_config)
    result["alpha"] = 0.5
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(RuntimeError, match="fingerprint"):
        _validate_result(
            path,
            expected_rollouts=500,
            expected_commit="source",
            feature_id=17,
            expected_run_config=run_config,
            expected_sae_sha256="sae-hash",
            expected_alpha=0.5,
            expected_hook_start_step=0,
        )
