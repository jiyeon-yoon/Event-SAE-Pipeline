import pytest

from scripts.openvla.run_intervention_development_validation import (
    compare_actions,
    select_validation_candidates,
    summarize_task_success,
)


def test_selects_rank_one_event_and_random_candidates():
    rows = [
        {"ranking": "event_aligned", "rank": 1, "feature_id": 30729},
        {"ranking": "event_aligned", "rank": 2, "feature_id": 4},
        {"ranking": "random_alive", "rank": 1, "feature_id": 1589},
    ]
    event, random = select_validation_candidates(rows)
    assert event["feature_id"] == 30729
    assert random["feature_id"] == 1589


def test_action_comparison_detects_identity_and_first_divergence():
    baseline = {
        (0, 0): [[0.0, 1.0], [2.0, 3.0]],
        (1, 0): [[4.0, 5.0]],
    }
    identity = compare_actions(baseline, baseline)
    assert identity["all_actions_match"] is True
    assert identity["changed_episodes"] == 0

    changed = {
        (0, 0): [[0.0, 1.0], [2.5, 3.0]],
        (1, 0): [[4.0, 5.0], [6.0, 7.0]],
    }
    comparison = compare_actions(baseline, changed)
    assert comparison["all_actions_match"] is False
    assert comparison["changed_episodes"] == 2
    assert comparison["length_mismatch_episodes"] == 1
    assert comparison["max_abs_delta"] == pytest.approx(0.5)
    assert comparison["per_episode"][0]["first_changed_step"] == 1
    assert comparison["per_episode"][1]["first_changed_step"] == 1


def test_task_success_summary_is_per_task():
    outcomes = {
        (0, 0): {"success": True},
        (0, 1): {"success": False},
        (1, 0): {"success": True},
        (1, 1): {"success": True},
    }
    summary = summarize_task_success(outcomes)
    assert summary["0"] == {"successes": 1, "rollouts": 2, "success_rate": 0.5}
    assert summary["1"] == {"successes": 2, "rollouts": 2, "success_rate": 1.0}
