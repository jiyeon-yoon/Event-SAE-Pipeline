from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.openvla.verify_paired_replay_runtime import (  # noqa: E402
    ReplaySample,
    _actions,
    _compare_sample,
)


def _sample(*, state_delta=0.0, control_hash="same", rgb_value=7):
    return ReplaySample(
        integration_state=np.asarray([1.0 + state_delta, 2.0]),
        qpos=np.asarray([3.0 + state_delta]),
        qvel=np.asarray([4.0]),
        control_state_sha256=control_hash,
        observation={
            "robot0_joint_pos": np.asarray([3.0 + state_delta]),
            "agentview_image": np.full((2, 2, 3), rgb_value, dtype=np.uint8),
        },
        reward=0.0,
        done=False,
    )


def test_action_pattern_exercises_motion_rotation_and_both_gripper_signs():
    actions = _actions(12)

    assert len(actions) == 12
    assert all(action.shape == (7,) for action in actions)
    assert any(np.any(action[:3] != 0) for action in actions)
    assert any(np.any(action[3:6] != 0) for action in actions)
    assert {float(action[-1]) for action in actions} == {-1.0, 1.0}


def test_compare_sample_reports_rgb_audit_when_control_state_is_exact():
    result = _compare_sample(
        "step",
        _sample(rgb_value=7),
        _sample(rgb_value=8),
        state_atol=1e-6,
    )

    assert result["integration_state"] == 0.0
    assert result["control_state_exact"] is True
    assert result["agentview_rgb_exact"] is False


def test_compare_sample_rejects_even_tiny_integration_state_drift():
    with pytest.raises(RuntimeError, match="not bit-exact"):
        _compare_sample(
            "step",
            _sample(),
            _sample(state_delta=1e-8),
            state_atol=1e-6,
        )


def test_compare_sample_rejects_physics_divergence():
    with pytest.raises(RuntimeError, match="integration_state"):
        _compare_sample(
            "step",
            _sample(),
            _sample(state_delta=1e-3),
            state_atol=1e-6,
        )


def test_compare_sample_rejects_control_state_divergence():
    with pytest.raises(RuntimeError, match="control state"):
        _compare_sample(
            "step",
            _sample(control_hash="normal"),
            _sample(control_hash="replay"),
            state_atol=1e-6,
        )
