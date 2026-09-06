"""Pure helpers for objective normal/forced-release paired rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


NORMAL_CONDITION = "normal"
FORCED_RELEASE_CONDITION = "forced_release"
FORCED_OPEN_VALUE = -1.0


def _objects(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("objects", {}))


def _fixtures(state: dict[str, Any]) -> dict[str, Any]:
    return dict(state.get("fixtures", {}))


def _match_object_name(value: Any, names: set[str]) -> str | None:
    text = str(value)
    if text in names:
        return text
    matches = [name for name in names if name.lower() == text.lower()]
    return matches[0] if len(matches) == 1 else None


def resolve_target_object(
    state: dict[str, Any],
    *,
    explicit: str | None = None,
) -> str:
    """Resolve the manipulated object from the first object goal argument.

    LIBERO goal predicates put the manipulated object in their first argument.
    An explicit per-task mapping is required when that rule is ambiguous.
    """

    names = set(_objects(state))
    if explicit is not None:
        matched = _match_object_name(explicit, names)
        if matched is None:
            raise ValueError(
                f"Configured target object {explicit!r} is absent; objects={sorted(names)}"
            )
        return matched

    candidates: list[str] = []
    rows = state.get("goals", {}).get("goal_predicates", [])
    for row in rows:
        predicate = list(row.get("predicate", []))
        if len(predicate) < 2:
            continue
        matched = _match_object_name(predicate[1], names)
        if matched is not None:
            candidates.append(matched)
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0]
    if not unique and len(names) == 1:
        return next(iter(names))
    raise ValueError(
        "Could not resolve one target object from BDDL goals; "
        f"candidates={unique} objects={sorted(names)}. "
        "Set paired_release.target_object_by_task for this task."
    )

def resolve_goal_destination(
    state: dict[str, Any],
    target_object: str,
    *,
    explicit: str | None = None,
) -> str | None:
    """Resolve the target's BDDL destination object/fixture when one exists."""

    names = set(_objects(state)) | set(_fixtures(state))
    if explicit is not None:
        matched = _match_object_name(explicit, names)
        if matched is None:
            raise ValueError(
                f"Configured destination {explicit!r} is absent; "
                f"objects/fixtures={sorted(names)}"
            )
        return matched

    candidates: list[str] = []
    for row in state.get("goals", {}).get("goal_predicates", []):
        predicate = list(row.get("predicate", []))
        arguments = predicate[1:]
        target_indexes = [
            index
            for index, value in enumerate(arguments)
            if _match_object_name(value, {target_object}) == target_object
        ]
        for target_index in target_indexes:
            for value in arguments[target_index + 1 :]:
                matched = _match_object_name(value, names - {target_object})
                if matched is not None:
                    candidates.append(matched)
                    break
    unique = sorted(set(candidates))
    return unique[0] if len(unique) == 1 else None


def target_grasped(state: dict[str, Any], target_object: str) -> bool:
    value = (
        state.get("contact_and_grasp", {})
        .get("grasped_objects", {})
        .get(target_object)
    )
    return value is True


def target_z(state: dict[str, Any], target_object: str) -> float:
    record = _objects(state).get(target_object)
    if record is None:
        raise KeyError(f"Missing target object state: {target_object}")
    position = np.asarray(record.get("position_world"), dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError(f"Invalid target position for {target_object}: {position}")
    return float(position[2])


def target_destination_contact(
    state: dict[str, Any],
    target_object: str,
    destination: str | None,
) -> bool:
    """Return true only for physical contact with the resolved goal destination."""

    if destination is None:
        return False
    target_owner = f"object:{target_object}"
    destination_owners = {
        f"object:{destination}",
        f"fixture:{destination}",
    }
    for contact in state.get("contact_and_grasp", {}).get("contacts", []):
        owners = set(contact.get("geom1_owners", [])) | set(
            contact.get("geom2_owners", [])
        )
        if target_owner in owners and owners.intersection(destination_owners):
            return True
    return False


@dataclass
class ReleaseTriggerDetector:
    """Detect stable grasp plus lift without using subjective progress labels."""

    target_object: str
    initial_z: float
    stable_grasp_steps: int
    min_lift_delta_m: float
    trigger_delay_steps: int
    consecutive_grasp: int = 0
    eligible_since: int | None = None
    t_grasp: int | None = None
    t_stable_grasp: int | None = None
    t_lift: int | None = None
    trigger_step: int | None = None

    def observe(self, step: int, state: dict[str, Any]) -> bool:
        if self.trigger_step is not None:
            return False
        grasped = target_grasped(state, self.target_object)
        lifted = (
            target_z(state, self.target_object) - self.initial_z
            >= self.min_lift_delta_m
        )

        if grasped:
            if self.consecutive_grasp == 0:
                self.t_grasp = int(step)
                self.t_stable_grasp = None
                self.t_lift = None
            self.consecutive_grasp += 1
            if (
                self.consecutive_grasp >= self.stable_grasp_steps
                and self.t_stable_grasp is None
            ):
                self.t_stable_grasp = int(step)
            if lifted and self.t_lift is None:
                self.t_lift = int(step)
        else:
            # These timestamps describe one continuous grasp segment.  Reset
            # them after a loss so a later trigger cannot mix two attempts.
            self.consecutive_grasp = 0
            self.eligible_since = None
            self.t_grasp = None
            self.t_stable_grasp = None
            self.t_lift = None

        eligible = grasped and lifted and (
            self.consecutive_grasp >= self.stable_grasp_steps
        )
        if eligible:
            if self.eligible_since is None:
                self.eligible_since = int(step)
            if step - self.eligible_since >= self.trigger_delay_steps:
                self.trigger_step = int(step)
                return True
        else:
            self.eligible_since = None
        return False

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_object": self.target_object,
            "initial_object_z": self.initial_z,
            "t_grasp": self.t_grasp,
            "t_stable_grasp": self.t_stable_grasp,
            "t_lift": self.t_lift,
            "trigger_step": self.trigger_step,
        }


def force_gripper_open(
    policy_libero_action: np.ndarray,
    *,
    forced_value: float = FORCED_OPEN_VALUE,
) -> np.ndarray:
    """Copy an action and change only LIBERO's gripper dimension."""

    if forced_value != FORCED_OPEN_VALUE:
        raise ValueError("LIBERO forced-open value must be -1.0")
    policy = np.asarray(policy_libero_action)
    if policy.shape != (7,):
        raise ValueError(f"Expected one 7D action, got {policy.shape}")
    executed = policy.copy()
    executed[-1] = forced_value
    return executed


def only_gripper_was_overridden(
    policy_libero_action: Any,
    executed_libero_action: Any,
    *,
    atol: float,
) -> bool:
    policy = np.asarray(policy_libero_action, dtype=np.float64)
    executed = np.asarray(executed_libero_action, dtype=np.float64)
    return (
        policy.shape == executed.shape == (7,)
        and np.allclose(policy[:6], executed[:6], rtol=0.0, atol=atol)
        and abs(float(executed[6]) - FORCED_OPEN_VALUE) <= atol
    )
