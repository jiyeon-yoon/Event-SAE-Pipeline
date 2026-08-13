from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.telemetry import (  # noqa: E402
    TelemetrySnapshot,
    _finite_difference_eef_velocity,
    _classify_contact,
    _goal_predicates,
    _observation_state,
    validate_preflight,
)


def test_contact_classification_distinguishes_grasp_from_collision_candidate():
    assert _classify_contact(["gripper:0"], ["object:bowl"]) == (
        "gripper_object_manipulation",
        False,
    )
    assert _classify_contact(["robot:0"], ["fixture:table"]) == (
        "robot_fixture_contact",
        True,
    )


def test_observation_state_excludes_all_camera_arrays():
    result = _observation_state(
        {
            "agentview_image": np.zeros((4, 4, 3), dtype=np.uint8),
            "agentview_depth": np.zeros((4, 4), dtype=np.float32),
            "robot0_joint_pos": np.asarray([1.0, 2.0]),
            "label": "not numeric",
        }
    )
    assert result == {"robot0_joint_pos": [1.0, 2.0]}


def test_eef_velocity_finite_difference_handles_quaternion_sign():
    linear, angular = _finite_difference_eef_velocity(
        np.zeros(3),
        np.asarray([0.0, 0.0, 0.0, 1.0]),
        np.asarray([0.1, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0, -1.0]),
        0.1,
    )
    np.testing.assert_allclose(linear, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(angular, [0.0, 0.0, 0.0], atol=1e-10)


def test_goal_predicates_are_saved_individually_and_unordered():
    class Raw:
        parsed_problem = {"goal_state": [("on", "bowl", "plate"), ("open", "drawer")]}

        def _eval_predicate(self, predicate):
            return predicate[0] == "on"

        def _check_success(self):
            return False

    result = _goal_predicates(SimpleNamespace(env=Raw()))
    assert [row["satisfied"] for row in result["goal_predicates"]] == [True, False]
    assert result["goal_fraction"] == 0.5
    assert "unordered" in result["predicate_semantics"]


def test_strict_preflight_reports_missing_fields():
    data = SimpleNamespace(qpos=np.zeros(1))
    env = SimpleNamespace(env=SimpleNamespace(), sim=SimpleNamespace(data=data))
    with pytest.raises(RuntimeError, match="preflight failed"):
        validate_preflight(env, {}, strict=True)
    capabilities = validate_preflight(env, {}, strict=False)
    assert capabilities["required_observation_keys"] is False
    assert capabilities["simulator_core_vectors"] is False
