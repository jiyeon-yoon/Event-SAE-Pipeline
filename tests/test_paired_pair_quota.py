from __future__ import annotations

import pytest

from event_sae.openvla.extended_collection.paired_runner import (
    TaskPairQuota,
    _normal_post_success_stop_reason,
    _require_single_task_run,
    _update_post_detach_goal_state,
)


def test_task_pair_quota_stops_only_after_both_targets():
    quota = TaskPairQuota(
        task_id=3,
        max_attempts=5,
        target_valid_pairs=2,
        target_primary_pairs=1,
    )

    quota.begin_attempt(0)
    quota.record_ineligible()
    quota.begin_attempt(1)
    quota.record_eligible(valid=False, primary=False)
    quota.begin_attempt(2)
    quota.record_eligible(valid=True, primary=False)
    assert quota.can_attempt is True

    quota.begin_attempt(3)
    quota.record_eligible(valid=True, primary=True)

    assert quota.target_reached is True
    assert quota.can_attempt is False
    assert quota.as_dict() == {
        "task_id": 3,
        "max_attempts": 5,
        "target_valid_pairs": 2,
        "target_primary_pairs": 1,
        "attempts": 4,
        "attempts_remaining": 1,
        "eligible_pairs": 3,
        "valid_pairs": 2,
        "invalid_pairs": 1,
        "ineligible_pairs": 1,
        "primary_analysis_pairs": 1,
        "target_reached": True,
    }


def test_task_pair_quota_stops_at_max_attempts_when_target_is_short():
    quota = TaskPairQuota(
        task_id=4,
        max_attempts=2,
        target_valid_pairs=2,
        target_primary_pairs=1,
    )
    quota.begin_attempt(0)
    quota.record_eligible(valid=True, primary=True)
    quota.begin_attempt(1)
    quota.record_ineligible()

    assert quota.target_reached is False
    assert quota.can_attempt is False
    with pytest.raises(RuntimeError, match="cannot start another attempt"):
        quota.begin_attempt(2)


def test_task_pair_quota_rejects_primary_without_valid_pair():
    quota = TaskPairQuota(
        task_id=0,
        max_attempts=1,
        target_valid_pairs=1,
        target_primary_pairs=1,
    )
    quota.begin_attempt(0)
    with pytest.raises(ValueError, match="must also be valid"):
        quota.record_eligible(valid=False, primary=True)


def test_paired_collection_requires_one_task_per_run():
    assert _require_single_task_run([4]) == 4
    with pytest.raises(ValueError, match="exactly one task"):
        _require_single_task_run([4, 5])


def test_success_before_open_waits_for_natural_release_and_stable_goal():
    common = {
        "success_step": 10,
        "observation_steps": 20,
        "post_open_steps": 20,
        "stable_detach_steps": 2,
        "goal_stable_steps": 2,
    }
    assert (
        _normal_post_success_stop_reason(
            step=10,
            t_cmd=None,
            t_detach_confirmed=None,
            t_goal_stable_after_detach=None,
            **common,
        )
        is None
    )
    assert (
        _normal_post_success_stop_reason(
            step=14,
            t_cmd=12,
            t_detach_confirmed=13,
            t_goal_stable_after_detach=14,
            **common,
        )
        == "natural_release_goal_stable"
    )


def test_success_without_natural_release_stops_at_bounded_timeout():
    common = {
        "success_step": 10,
        "t_cmd": None,
        "t_detach_confirmed": None,
        "t_goal_stable_after_detach": None,
        "observation_steps": 20,
        "post_open_steps": 20,
        "stable_detach_steps": 2,
        "goal_stable_steps": 2,
    }
    assert _normal_post_success_stop_reason(step=29, **common) is None
    assert (
        _normal_post_success_stop_reason(step=30, **common)
        == "post_success_observation_timeout"
    )


def test_late_open_extends_timeout_for_detach_and_goal_confirmation():
    common = {
        "success_step": 10,
        "t_cmd": 30,
        "t_detach_confirmed": None,
        "t_goal_stable_after_detach": None,
        "observation_steps": 20,
        "post_open_steps": 20,
        "stable_detach_steps": 2,
        "goal_stable_steps": 2,
    }
    assert _normal_post_success_stop_reason(step=30, **common) is None
    assert _normal_post_success_stop_reason(step=49, **common) is None
    assert (
        _normal_post_success_stop_reason(step=50, **common)
        == "post_success_observation_timeout"
    )


def test_post_detach_goal_requires_consecutive_ungrasped_steps():
    candidate, count, confirmed = _update_post_detach_goal_state(
        step=12,
        detach_confirmed=12,
        goal_satisfied=True,
        target_is_grasped=False,
        candidate=None,
        consecutive_goal_steps=0,
        required_steps=2,
    )
    assert (candidate, count, confirmed) == (12, 1, None)

    candidate, count, confirmed = _update_post_detach_goal_state(
        step=13,
        detach_confirmed=12,
        goal_satisfied=True,
        target_is_grasped=False,
        candidate=candidate,
        consecutive_goal_steps=count,
        required_steps=2,
    )
    assert (candidate, count, confirmed) == (12, 2, 13)

    assert _update_post_detach_goal_state(
        step=14,
        detach_confirmed=12,
        goal_satisfied=False,
        target_is_grasped=False,
        candidate=candidate,
        consecutive_goal_steps=count,
        required_steps=2,
    ) == (None, 0, None)
