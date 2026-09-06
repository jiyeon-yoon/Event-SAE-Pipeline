from __future__ import annotations

import numpy as np
import pytest

from event_sae.openvla.extended_collection.controlled_release import (
    ReleaseTriggerDetector,
    force_gripper_open,
    only_gripper_was_overridden,
    resolve_goal_destination,
    resolve_target_object,
    target_destination_contact,
)


def _state(*, grasped=False, z=0.0):
    return {
        "objects": {
            "bowl": {"position_world": [0.0, 0.0, z]},
            "plate": {"position_world": [0.1, 0.0, 0.0]},
        },
        "fixtures": {"table": {"position_world": [0.0, 0.0, 0.0]}},
        "contact_and_grasp": {
            "grasped_objects": {"bowl": grasped, "plate": False},
            "contacts": [],
        },
        "goals": {
            "goal_predicates": [
                {"predicate": ["On", "bowl", "plate"], "satisfied": False}
            ]
        },
    }


def test_resolves_target_and_object_destination_from_bddl():
    state = _state()
    assert resolve_target_object(state) == "bowl"
    assert resolve_goal_destination(state, "bowl") == "plate"
    assert resolve_target_object(state, explicit="BOWL") == "bowl"


def test_trigger_requires_one_continuous_stable_grasp_lift_window():
    detector = ReleaseTriggerDetector(
        target_object="bowl",
        initial_z=0.0,
        stable_grasp_steps=2,
        min_lift_delta_m=0.02,
        trigger_delay_steps=1,
    )
    assert not detector.observe(0, _state(grasped=True, z=0.03))
    assert not detector.observe(1, _state(grasped=False, z=0.03))
    assert detector.t_grasp is None
    assert not detector.observe(2, _state(grasped=True, z=0.03))
    assert not detector.observe(3, _state(grasped=True, z=0.03))
    assert detector.observe(4, _state(grasped=True, z=0.03))
    assert detector.as_dict()["trigger_step"] == 4


def test_force_open_changes_only_gripper_and_does_not_mutate_input():
    policy = np.asarray([0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 1.0])
    original = policy.copy()
    executed = force_gripper_open(policy)
    assert np.array_equal(policy, original)
    assert np.array_equal(executed[:6], policy[:6])
    assert executed[6] == -1.0
    assert only_gripper_was_overridden(policy, executed, atol=0.0)

    with pytest.raises(ValueError, match="7D"):
        force_gripper_open(np.zeros(6))
    with pytest.raises(ValueError, match="must be -1.0"):
        force_gripper_open(policy, forced_value=0.0)


def test_destination_contact_requires_exact_target_and_destination():
    state = _state()
    state["contact_and_grasp"]["contacts"] = [
        {
            "geom1_owners": ["object:bowl"],
            "geom2_owners": ["object:plate"],
        }
    ]
    assert target_destination_contact(state, "bowl", "plate")
    assert not target_destination_contact(state, "bowl", "table")
    assert not target_destination_contact(state, "bowl", None)
