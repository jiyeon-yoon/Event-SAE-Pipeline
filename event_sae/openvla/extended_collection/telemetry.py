"""Version-tolerant LIBERO / robosuite / MuJoCo telemetry extraction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

import numpy as np


_SIM_VECTOR_NAMES = (
    "time",
    "qpos",
    "qvel",
    "act",
    "qacc",
    "ctrl",
    "mocap_pos",
    "mocap_quat",
    "userdata",
    "qfrc_actuator",
    "qfrc_applied",
    "qfrc_bias",
    "qfrc_constraint",
    "qfrc_passive",
    "actuator_force",
    "xfrc_applied",
    "cfrc_ext",
    "sensordata",
)
_IMAGE_KEY_PARTS = ("image", "depth", "segmentation")


@dataclass(frozen=True)
class TelemetrySnapshot:
    json_state: dict[str, Any]
    sim_vectors: dict[str, np.ndarray]
    eef_pos: np.ndarray | None
    eef_quat: np.ndarray | None


def _raw_env(env):
    return getattr(env, "env", env)


def _loaded_mujoco():
    """Return the simulator's already-loaded MuJoCo module without importing it.

    Constructing a real LIBERO environment loads MuJoCo before telemetry starts.
    Avoiding a fresh import here keeps preflight side-effect free and also lets
    lightweight schema/unit checks run on machines without a working GL runtime.
    """

    return sys.modules.get("mujoco")


def _numeric(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if array.dtype.kind not in "biufc":
        return None
    return array.copy()


def _observation_state(obs: dict[str, Any]) -> dict[str, Any]:
    state: dict[str, Any] = {}
    for key, value in obs.items():
        lowered = key.lower()
        if any(part in lowered for part in _IMAGE_KEY_PARTS):
            continue
        array = _numeric(value)
        if array is not None:
            state[key] = array.tolist() if array.ndim else array.item()
    return state


def _sim_vectors(env) -> dict[str, np.ndarray]:
    data = env.sim.data
    vectors: dict[str, np.ndarray] = {}
    for name in _SIM_VECTOR_NAMES:
        if hasattr(data, name):
            vectors[name] = np.asarray(getattr(data, name)).copy()
    return vectors


def _id2name(model, kind: str, index: int) -> str | None:
    method = getattr(model, f"{kind}_id2name", None)
    if callable(method):
        try:
            return method(index)
        except Exception:
            pass
    try:
        mujoco = _loaded_mujoco()
        if mujoco is None:
            return None
        enum = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
        native = getattr(model, "_model", model)
        return mujoco.mj_id2name(native, enum, index)
    except Exception:
        return None


def _name2id(model, kind: str, name: str) -> int | None:
    method = getattr(model, f"{kind}_name2id", None)
    if callable(method):
        try:
            return int(method(name))
        except Exception:
            pass
    try:
        mujoco = _loaded_mujoco()
        if mujoco is None:
            return None
        enum = getattr(mujoco.mjtObj, f"mjOBJ_{kind.upper()}")
        native = getattr(model, "_model", model)
        value = int(mujoco.mj_name2id(native, enum, name))
        return value if value >= 0 else None
    except Exception:
        return None


def _native_model_data(env):
    model = env.sim.model
    data = env.sim.data
    return getattr(model, "_model", model), getattr(data, "_data", data)


def _body_velocity(env, body_id: int) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        mujoco = _loaded_mujoco()
        if mujoco is None:
            raise RuntimeError("MuJoCo is not loaded")
        native_model, native_data = _native_model_data(env)
        velocity = np.zeros(6, dtype=np.float64)
        mujoco.mj_objectVelocity(
            native_model,
            native_data,
            mujoco.mjtObj.mjOBJ_BODY,
            int(body_id),
            velocity,
            0,
        )
        # MuJoCo spatial velocity order is angular, then linear.
        return (
            velocity[3:].copy(),
            velocity[:3].copy(),
            "mujoco.mj_objectVelocity(world)",
        )
    except Exception:
        data = env.sim.data
        if hasattr(data, "body_xvelp") and hasattr(data, "body_xvelr"):
            return (
                np.asarray(data.body_xvelp[body_id]).copy(),
                np.asarray(data.body_xvelr[body_id]).copy(),
                "body_xvelp/body_xvelr",
            )
        if hasattr(data, "cvel"):
            value = np.asarray(data.cvel[body_id]).copy()
            return value[3:], value[:3], "cvel(angular,linear)"
    return np.full(3, np.nan), np.full(3, np.nan), "unavailable"


def _joint_value(data, method_name: str, joint_name: str) -> Any:
    method = getattr(data, method_name, None)
    if not callable(method):
        return None
    try:
        value = np.asarray(method(joint_name)).copy()
        return value.tolist() if value.ndim else value.item()
    except Exception:
        return None


def _object_and_fixture_state(env) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = _raw_env(env)
    model, data = env.sim.model, env.sim.data
    body_ids = getattr(raw, "obj_body_id", {})

    def collect(items: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, item in items.items():
            body_id = body_ids.get(name)
            if body_id is None:
                root_body = getattr(item, "root_body", name)
                body_id = _name2id(model, "body", root_body)
            record: dict[str, Any] = {
                "root_body": getattr(item, "root_body", None),
                "body_id": body_id,
                "contact_geoms": list(getattr(item, "contact_geoms", []) or []),
                "joint_names": list(getattr(item, "joints", []) or []),
            }
            if body_id is not None:
                record["position_world"] = np.asarray(data.body_xpos[body_id]).tolist()
                record["quaternion_wxyz_world"] = np.asarray(
                    data.body_xquat[body_id]
                ).tolist()
                linear, angular, method = _body_velocity(env, int(body_id))
                record["linear_velocity_world"] = linear.tolist()
                record["angular_velocity_world"] = angular.tolist()
                record["velocity_source"] = method
            record["joint_state"] = {
                joint: {
                    "qpos": _joint_value(data, "get_joint_qpos", joint),
                    "qvel": _joint_value(data, "get_joint_qvel", joint),
                }
                for joint in record["joint_names"]
            }
            result[name] = record
        return result

    return collect(getattr(raw, "objects_dict", {})), collect(
        getattr(raw, "fixtures_dict", {})
    )


def _object_sites(env) -> dict[str, Any]:
    raw, model, data = _raw_env(env), env.sim.model, env.sim.data
    result: dict[str, Any] = {}
    for name, site in getattr(raw, "object_sites_dict", {}).items():
        site_id = _name2id(model, "site", name)
        record: dict[str, Any] = {
            "site_id": site_id,
            "parent_name": getattr(site, "parent_name", None),
        }
        if site_id is not None:
            record["position_world"] = np.asarray(data.site_xpos[site_id]).tolist()
            record["rotation_matrix_world"] = (
                np.asarray(data.site_xmat[site_id]).reshape(3, 3).tolist()
            )
        result[name] = record
    return result


def _contact_force(env, index: int) -> list[float] | None:
    try:
        mujoco = _loaded_mujoco()
        if mujoco is None:
            return None
        native_model, native_data = _native_model_data(env)
        force = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(native_model, native_data, index, force)
        return force.tolist()
    except Exception:
        return None


def _classify_contact(
    geom1_owners: list[str], geom2_owners: list[str]
) -> tuple[str, bool]:
    """Attach a transparent candidate label; never pretend it is ground truth."""

    owners = set(geom1_owners + geom2_owners)
    kinds = {value.split(":", 1)[0] for value in owners}
    if "gripper" in kinds and "object" in kinds:
        return "gripper_object_manipulation", False
    if "robot" in kinds and "fixture" in kinds:
        return "robot_fixture_contact", True
    if "gripper" in kinds and "fixture" in kinds:
        return "gripper_fixture_contact", True
    if "robot" in kinds and "object" in kinds:
        return "robot_object_contact", True
    if kinds == {"object", "fixture"}:
        return "object_fixture_contact", False
    if kinds == {"object"}:
        return "object_object_contact", False
    if kinds <= {"robot", "gripper"} and kinds:
        return "robot_self_contact", True
    return "unclassified_contact", False


def _geom_owner_map(env) -> dict[str, list[str]]:
    raw = _raw_env(env)
    cached = getattr(raw, "_extended_collection_geom_owners", None)
    if cached is not None:
        return cached
    owners: dict[str, list[str]] = {}

    def add(geoms, owner: str) -> None:
        for geom in geoms or []:
            owners.setdefault(str(geom), []).append(owner)

    for name, item in getattr(raw, "objects_dict", {}).items():
        add(getattr(item, "contact_geoms", []), f"object:{name}")
    for name, item in getattr(raw, "fixtures_dict", {}).items():
        add(getattr(item, "contact_geoms", []), f"fixture:{name}")
    for robot_index, robot in enumerate(getattr(raw, "robots", [])):
        add(getattr(robot.robot_model, "contact_geoms", []), f"robot:{robot_index}")
        gripper = getattr(robot, "gripper", None)
        add(getattr(gripper, "contact_geoms", []), f"gripper:{robot_index}")
        important = getattr(gripper, "important_geoms", {}) or {}
        for values in important.values():
            add(
                values if isinstance(values, (list, tuple)) else [values],
                f"gripper:{robot_index}",
            )
    setattr(raw, "_extended_collection_geom_owners", owners)
    return owners


def _contacts_and_grasp(env) -> dict[str, Any]:
    raw, model, data = _raw_env(env), env.sim.model, env.sim.data
    owners = _geom_owner_map(env)
    contacts: list[dict[str, Any]] = []
    for index in range(int(getattr(data, "ncon", 0))):
        contact = data.contact[index]
        geom1_id, geom2_id = int(contact.geom1), int(contact.geom2)
        geom1 = _id2name(model, "geom", geom1_id)
        geom2 = _id2name(model, "geom", geom2_id)
        geom1_owners = owners.get(str(geom1), [])
        geom2_owners = owners.get(str(geom2), [])
        semantic_class, potential_collision = _classify_contact(
            geom1_owners, geom2_owners
        )
        record = {
            "contact_index": index,
            "geom1_id": geom1_id,
            "geom2_id": geom2_id,
            "geom1_name": geom1,
            "geom2_name": geom2,
            "geom1_owners": geom1_owners,
            "geom2_owners": geom2_owners,
            "semantic_class": semantic_class,
            "potentially_unwanted_collision": potential_collision,
            "distance": float(getattr(contact, "dist", np.nan)),
            "position_world": np.asarray(getattr(contact, "pos", [])).tolist(),
            "contact_frame": np.asarray(getattr(contact, "frame", [])).tolist(),
            "friction": np.asarray(getattr(contact, "friction", [])).tolist(),
            "force_torque_contact_frame": _contact_force(env, index),
        }
        contacts.append(record)

    grasped: dict[str, bool | None] = {}
    robots = getattr(raw, "robots", [])
    checker = getattr(raw, "_check_grasp", None)
    if robots and callable(checker):
        gripper = getattr(robots[0], "gripper", None)
        for name, item in getattr(raw, "objects_dict", {}).items():
            try:
                grasped[name] = bool(checker(gripper, item))
            except Exception:
                grasped[name] = None
    return {
        # A MuJoCo contact is not automatically an unwanted collision.  Store
        # the physical contacts losslessly and leave semantic collision labels
        # to downstream analysis with the owner metadata above.
        "contact_count": len(contacts),
        "has_contact": bool(contacts),
        "potential_collision_count": sum(
            bool(row["potentially_unwanted_collision"]) for row in contacts
        ),
        "contacts": contacts,
        "grasped_objects": grasped,
        "grasp_definition": "both gripper fingerpad groups contact the object (robosuite _check_grasp)",
        "collision_semantics": "derive from contact geom owners; no contact is silently relabeled as a collision",
    }


def _goal_predicates(env) -> dict[str, Any]:
    raw = _raw_env(env)
    goals = list(getattr(raw, "parsed_problem", {}).get("goal_state", []))
    evaluator = getattr(raw, "_eval_predicate", None)
    rows: list[dict[str, Any]] = []
    for index, predicate in enumerate(goals):
        satisfied: bool | None
        error: str | None = None
        try:
            satisfied = bool(evaluator(predicate)) if callable(evaluator) else None
        except Exception as exc:
            satisfied, error = None, repr(exc)
        rows.append(
            {
                "predicate_index": index,
                "predicate": list(predicate),
                "satisfied": satisfied,
                "error": error,
            }
        )
    known = [row["satisfied"] for row in rows if row["satisfied"] is not None]
    fraction = (
        float(sum(known) / len(rows)) if len(known) == len(rows) and rows else None
    )
    try:
        success = bool(env.check_success())
    except Exception:
        try:
            success = bool(raw._check_success())
        except Exception:
            success = None
    return {
        "goal_predicates": rows,
        "goal_fraction": fraction,
        "all_goal_predicates_satisfied": success,
        "predicate_semantics": "unordered BDDL conjunction; not an ordered semantic subgoal sequence",
    }


def _robot_state(
    env, obs: dict[str, Any], previous, dt: float | None
) -> dict[str, Any]:
    raw = _raw_env(env)
    robot = raw.robots[0] if getattr(raw, "robots", []) else None
    eef_pos = _numeric(obs.get("robot0_eef_pos"))
    eef_quat = _numeric(obs.get("robot0_eef_quat"))
    linear = _numeric(getattr(robot, "_hand_vel", None)) if robot is not None else None
    angular = (
        _numeric(getattr(robot, "_hand_ang_vel", None)) if robot is not None else None
    )
    velocity_source = "robosuite robot Jacobian"
    if (linear is None or angular is None) and previous is not None and dt and dt > 0:
        linear, angular = _finite_difference_eef_velocity(
            previous.eef_pos, previous.eef_quat, eef_pos, eef_quat, dt
        )
        velocity_source = "finite difference"
    return {
        "joint_position": _numeric(getattr(robot, "_joint_positions", None)),
        "joint_velocity": _numeric(getattr(robot, "_joint_velocities", None)),
        "joint_torque_command": _numeric(getattr(robot, "torques", None)),
        "eef_position": eef_pos,
        "eef_quaternion_xyzw": eef_quat,
        "eef_linear_velocity": linear,
        "eef_angular_velocity": angular,
        "eef_velocity_source": (
            velocity_source
            if linear is not None and angular is not None
            else "unavailable"
        ),
        "gripper_qpos": _numeric(obs.get("robot0_gripper_qpos")),
        "gripper_qvel": _numeric(obs.get("robot0_gripper_qvel")),
    }


def _quat_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return np.asarray(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def _finite_difference_eef_velocity(
    previous_pos: np.ndarray | None,
    previous_quat: np.ndarray | None,
    current_pos: np.ndarray | None,
    current_quat: np.ndarray | None,
    dt: float,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if previous_pos is None or current_pos is None:
        linear = None
    else:
        linear = (np.asarray(current_pos) - np.asarray(previous_pos)) / dt
    if previous_quat is None or current_quat is None:
        return linear, None
    q0 = np.asarray(previous_quat, dtype=np.float64)
    q1 = np.asarray(current_quat, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    delta = _quat_multiply_xyzw(q1, np.asarray([-q0[0], -q0[1], -q0[2], q0[3]]))
    vector, scalar = delta[:3], float(np.clip(delta[3], -1.0, 1.0))
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        angular = np.zeros(3)
    else:
        angle = 2.0 * np.arctan2(norm, scalar)
        angular = (vector / norm) * (angle / dt)
    return linear, angular


def capture_snapshot(
    env,
    obs: dict[str, Any],
    *,
    previous: TelemetrySnapshot | None,
    dt: float | None,
) -> TelemetrySnapshot:
    """Capture one state without rendering or stepping the environment."""

    objects, fixtures = _object_and_fixture_state(env)
    robot = _robot_state(env, obs, previous, dt)
    json_state = {
        "observation_state": _observation_state(obs),
        "robot": robot,
        "objects": objects,
        "fixtures": fixtures,
        "object_sites": _object_sites(env),
        "contact_and_grasp": _contacts_and_grasp(env),
        "goals": _goal_predicates(env),
    }
    return TelemetrySnapshot(
        json_state=json_state,
        sim_vectors=_sim_vectors(env),
        eef_pos=_numeric(obs.get("robot0_eef_pos")),
        eef_quat=_numeric(obs.get("robot0_eef_quat")),
    )


def _named_elements(model, kind: str, count: int) -> list[dict[str, Any]]:
    return [
        {"id": index, "name": _id2name(model, kind, index)}
        for index in range(int(count))
    ]


def build_simulator_schema(env, obs: dict[str, Any]) -> dict[str, Any]:
    raw, model = _raw_env(env), env.sim.model
    joints: list[dict[str, Any]] = []
    qpos_adr = np.asarray(getattr(model, "jnt_qposadr", []), dtype=int)
    dof_adr = np.asarray(getattr(model, "jnt_dofadr", []), dtype=int)
    for index in range(int(getattr(model, "njnt", 0))):
        q_start = int(qpos_adr[index])
        q_end = int(qpos_adr[index + 1]) if index + 1 < len(qpos_adr) else int(model.nq)
        v_start = int(dof_adr[index])
        v_end = int(dof_adr[index + 1]) if index + 1 < len(dof_adr) else int(model.nv)
        joints.append(
            {
                "id": index,
                "name": _id2name(model, "joint", index),
                "qpos_slice": [q_start, q_end],
                "qvel_slice": [v_start, v_end],
            }
        )
    sensors: list[dict[str, Any]] = []
    sensor_adr = np.asarray(getattr(model, "sensor_adr", []), dtype=int)
    sensor_dim = np.asarray(getattr(model, "sensor_dim", []), dtype=int)
    for index in range(int(getattr(model, "nsensor", 0))):
        start, width = int(sensor_adr[index]), int(sensor_dim[index])
        sensors.append(
            {
                "id": index,
                "name": _id2name(model, "sensor", index),
                "sensordata_slice": [start, start + width],
            }
        )
    return {
        "mujoco_dimensions": {
            key: int(getattr(model, key, 0))
            for key in (
                "nq",
                "nv",
                "na",
                "nu",
                "nbody",
                "njnt",
                "ngeom",
                "nsite",
                "nsensor",
            )
        },
        "sim_vector_shapes": {
            key: list(value.shape) for key, value in _sim_vectors(env).items()
        },
        "joints": joints,
        "actuators": _named_elements(model, "actuator", getattr(model, "nu", 0)),
        "bodies": _named_elements(model, "body", getattr(model, "nbody", 0)),
        "geometries": _named_elements(model, "geom", getattr(model, "ngeom", 0)),
        "sites": _named_elements(model, "site", getattr(model, "nsite", 0)),
        "sensors": sensors,
        "observation": {
            key: {
                "shape": list(np.asarray(value).shape),
                "dtype": str(np.asarray(value).dtype),
            }
            for key, value in obs.items()
        },
        "objects": {
            name: {
                "root_body": getattr(item, "root_body", None),
                "contact_geoms": list(getattr(item, "contact_geoms", []) or []),
                "joints": list(getattr(item, "joints", []) or []),
            }
            for name, item in getattr(raw, "objects_dict", {}).items()
        },
        "fixtures": {
            name: {
                "root_body": getattr(item, "root_body", None),
                "contact_geoms": list(getattr(item, "contact_geoms", []) or []),
                "joints": list(getattr(item, "joints", []) or []),
            }
            for name, item in getattr(raw, "fixtures_dict", {}).items()
        },
        "bddl_goal_state": [
            list(value)
            for value in getattr(raw, "parsed_problem", {}).get("goal_state", [])
        ],
        "control_frequency_hz": float(getattr(raw, "control_freq", np.nan)),
    }


def validate_preflight(
    env,
    obs: dict[str, Any],
    *,
    strict: bool,
) -> dict[str, Any]:
    """Fail before rollout collection if a required pinned API is unavailable."""

    raw, data = _raw_env(env), env.sim.data
    required_obs = (
        "agentview_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    )
    capabilities = {
        "required_observation_keys": all(key in obs for key in required_obs),
        "simulator_core_vectors": all(
            hasattr(data, key)
            for key in (
                "time",
                "qpos",
                "qvel",
                "act",
                "qacc",
                "ctrl",
                "mocap_pos",
                "mocap_quat",
                "userdata",
            )
        ),
        "simulator_force_vectors": all(
            hasattr(data, key)
            for key in (
                "qfrc_actuator",
                "qfrc_applied",
                "qfrc_bias",
                "qfrc_constraint",
                "qfrc_passive",
                "actuator_force",
                "xfrc_applied",
                "cfrc_ext",
            )
        ),
        "simulator_sensors": hasattr(data, "sensordata"),
        "contact_records": hasattr(data, "ncon") and hasattr(data, "contact"),
        "object_pose_mapping": bool(getattr(raw, "obj_body_id", {})),
        "object_velocity": hasattr(data, "cvel") or hasattr(data, "body_xvelp"),
        "grasp_predicate": callable(getattr(raw, "_check_grasp", None)),
        "bddl_goal_predicates": bool(
            getattr(raw, "parsed_problem", {}).get("goal_state", [])
        )
        and callable(getattr(raw, "_eval_predicate", None)),
        "robot_joint_state": bool(getattr(raw, "robots", [])),
    }
    try:
        mujoco = _loaded_mujoco()
        if mujoco is None:
            raise RuntimeError("MuJoCo is not loaded")
        capabilities["mujoco_contact_force"] = hasattr(mujoco, "mj_contactForce")
        capabilities["mujoco_body_velocity"] = hasattr(mujoco, "mj_objectVelocity")
    except Exception:
        capabilities["mujoco_contact_force"] = False
        capabilities["mujoco_body_velocity"] = False
    capabilities["object_velocity"] = bool(
        capabilities["object_velocity"] or capabilities["mujoco_body_velocity"]
    )
    try:
        dt = 1.0 / float(getattr(raw, "control_freq", 20))
        probe = capture_snapshot(env, obs, previous=None, dt=dt)

        def finite(value: Any) -> bool:
            array = _numeric(value)
            return array is not None and bool(np.all(np.isfinite(array)))

        named_bodies = list(probe.json_state["objects"].values()) + list(
            probe.json_state["fixtures"].values()
        )
        capabilities["object_pose_velocity_values"] = bool(named_bodies) and all(
            all(
                finite(record.get(field))
                for field in (
                    "position_world",
                    "quaternion_wxyz_world",
                    "linear_velocity_world",
                    "angular_velocity_world",
                )
            )
            for record in named_bodies
        )
        required_robot_fields = (
            "joint_position",
            "joint_velocity",
            "joint_torque_command",
            "eef_position",
            "eef_quaternion_xyzw",
            "eef_linear_velocity",
            "eef_angular_velocity",
            "gripper_qpos",
            "gripper_qvel",
        )
        capabilities["robot_state_values"] = all(
            finite(probe.json_state["robot"].get(field))
            for field in required_robot_fields
        )
        capabilities["simulator_vector_values"] = all(
            bool(np.all(np.isfinite(value))) for value in probe.sim_vectors.values()
        )
        goal_rows = probe.json_state["goals"]["goal_predicates"]
        capabilities["goal_predicate_values"] = bool(goal_rows) and all(
            isinstance(row.get("satisfied"), bool) and row.get("error") is None
            for row in goal_rows
        )
        grasp_values = probe.json_state["contact_and_grasp"]["grasped_objects"]
        capabilities["grasp_values"] = bool(grasp_values) and all(
            isinstance(value, bool) for value in grasp_values.values()
        )
        contacts = probe.json_state["contact_and_grasp"]["contacts"]
        capabilities["contact_force_values"] = all(
            finite(row.get("force_torque_contact_frame")) for row in contacts
        )
    except Exception:
        for name in (
            "object_pose_velocity_values",
            "robot_state_values",
            "simulator_vector_values",
            "goal_predicate_values",
            "grasp_values",
            "contact_force_values",
        ):
            capabilities[name] = False
    if strict:
        missing = [name for name, available in capabilities.items() if not available]
        if missing:
            raise RuntimeError(
                "Extended collector preflight failed; unavailable required fields: "
                + ", ".join(missing)
            )
    return capabilities
