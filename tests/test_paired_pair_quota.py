from __future__ import annotations

import pytest

from event_sae.openvla.extended_collection.paired_runner import (
    TaskPairQuota,
    _require_single_task_run,
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
