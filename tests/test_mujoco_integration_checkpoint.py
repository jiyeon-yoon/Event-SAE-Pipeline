from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import event_sae.openvla.extended_collection.paired_runner as paired_runner
from event_sae.openvla.extended_collection.paired_runner import (
    _capture_mujoco_integration_checkpoint,
    _replay_values_equal,
    _reset_to_initial_state,
    _restore_mujoco_integration_checkpoint,
    _robosuite_python_state_sha256,
)


class FakeModel:
    nq = 2
    nv = 2
    na = 1
    nu = 2
    nmocap = 0
    nuserdata = 1
    neq = 0
    npluginstate = 0

    def __init__(self, xml="<mujoco model='paired'/>"):
        self.xml = xml

    def get_xml(self):
        return self.xml


class FakeInterpolator:
    def __init__(self):
        self.start = np.asarray([0.1, 0.2])
        self.goal = np.asarray([0.3, 0.4])
        self.step = 3


class OtherController:
    pass


class FakeController:
    def __init__(self, state_ref=None, update_delta=0.0):
        self.initial_joint = np.asarray([0.2, -0.1])
        self.initial_ee_pos = np.asarray([0.3, 0.4, 0.5])
        self.initial_ee_ori_mat = np.eye(3)
        self.goal_pos = np.asarray([0.6, 0.7, 0.8])
        self.goal_ori = np.eye(3) * 2
        self.relative_ori = np.asarray([0.01, 0.02, 0.03])
        self.ori_ref = np.eye(3) * 3
        self.kp = np.asarray([150.0] * 6)
        self.kd = np.asarray([20.0] * 6)
        self.torques = np.asarray([0.4, 0.5])
        self.action_scale = np.asarray([0.05, 0.05])
        self.action_input_transform = np.asarray([1.0, 1.0])
        self.action_output_transform = np.asarray([0.0, 0.0])
        self.new_update = True
        self.interpolator_pos = FakeInterpolator()
        self.interpolator_ori = None
        self.update_calls = 0
        self._state_ref = state_ref
        self._update_delta = float(update_delta)

    def update(self, *, force):
        assert force is True
        self.update_calls += 1
        if self._state_ref is not None:
            self._state_ref.state[-1] += self._update_delta


class FakeBuffer:
    def __init__(self, value):
        self.dim = 2
        self.last = np.asarray([value, value + 1.0])
        self.current = np.asarray([value + 2.0, value + 3.0])


class FakeRingBuffer:
    def __init__(self, value):
        self.dim = 2
        self.length = 3
        self._size = 2
        self.ptr = 1
        self.buf = np.full((3, 2), value, dtype=np.float64)


class FakeGripper:
    def __init__(self):
        self.current_action = np.asarray([-0.75, 0.75])


class FakeRobot:
    def __init__(self, state_ref=None, update_delta=0.0):
        self.controller = FakeController(state_ref, update_delta)
        self.gripper = FakeGripper()
        self.torques = np.asarray([0.8, 0.9])
        self.recent_qpos = FakeBuffer(1.0)
        self.recent_actions = FakeBuffer(2.0)
        self.recent_torques = FakeBuffer(3.0)
        self.recent_ee_forcetorques = FakeBuffer(4.0)
        self.recent_ee_pose = FakeBuffer(5.0)
        self.recent_ee_vel = FakeBuffer(6.0)
        self.recent_ee_vel_buffer = FakeRingBuffer(7.0)
        self.recent_ee_acc = FakeBuffer(8.0)


class FakeObservable:
    def __init__(self):
        self.name = "agentview_image"
        self._sensor = lambda cache: cache[self.name]
        self._corrupter = lambda value: value
        self._filter = lambda value: value
        self._delayer = lambda: 0.0
        self._sampling_timestep = 0.05
        self._enabled = True
        self._active = True
        self._is_number = False
        self._data_shape = (2, 2, 3)
        self._time_since_last_sample = 0.025
        self._current_delay = 0.0
        self._current_observed_value = np.full((2, 2, 3), 17, dtype=np.uint8)
        self._sampled = True


class FakeRawEnv:
    def __init__(self, sim, *, update_delta=0.0):
        self.sim = sim
        self.timestep = 10
        self.cur_time = 0.5
        self.done = False
        self.hard_reset = True
        self.robots = [FakeRobot(sim.data, update_delta)]
        observable = FakeObservable()
        self._observables = {observable.name: observable}
        self._obs_cache = {
            observable.name: observable._current_observed_value.copy(),
            "robot0_joint_pos": np.asarray([0.1, 0.2]),
        }
        self.rng = np.random.default_rng(123)

    def _get_observations(self):
        return {
            name: observable._current_observed_value.copy()
            for name, observable in self._observables.items()
            if observable._active
        }


def _fake_env(state, *, xml="<mujoco model='paired'/>", update_delta=0.0):
    model = FakeModel(xml)
    data = SimpleNamespace(state=np.asarray(state, dtype=np.float64).copy())
    sim = SimpleNamespace(model=model, data=data)
    raw = FakeRawEnv(sim, update_delta=update_delta)
    return SimpleNamespace(sim=sim, env=raw), model, data, raw


def _install_fake_mujoco(monkeypatch, *, set_delta=0.0, forward_delta=0.0):
    module = ModuleType("mujoco")
    module.mjtState = SimpleNamespace(mjSTATE_INTEGRATION=7)
    module.mj_stateSize = lambda model, spec: 8

    def get_state(model, data, output, spec):
        output[:] = data.state

    def set_state(model, data, values, spec):
        data.state = np.asarray(values, dtype=np.float64).copy()
        data.state[-1] += set_delta

    def forward(model, data):
        data.state = data.state.copy()
        data.state[-1] += forward_delta

    module.mj_getState = get_state
    module.mj_setState = set_state
    module.mj_forward = forward
    monkeypatch.setitem(sys.modules, "mujoco", module)


def _checkpoint(env):
    return _capture_mujoco_integration_checkpoint(
        env,
        include_model_fingerprint=True,
        include_robosuite_python_state=True,
    )


def test_mjstate_and_python_state_capture_restore_round_trip(monkeypatch):
    _install_fake_mujoco(monkeypatch)
    original = np.arange(8, dtype=np.float64)
    env, _, data, raw = _fake_env(original)
    checkpoint = _checkpoint(env)
    expected_python = checkpoint.robosuite_python_state
    expected_python_hash = _robosuite_python_state_sha256(expected_python)

    data.state[:] = -1
    raw.timestep = 99
    raw.cur_time = 9.9
    raw.done = True
    raw.robots[0].controller.initial_joint[:] = 99
    raw.robots[0].controller.interpolator_pos.goal[:] = 99
    raw.robots[0].gripper.current_action[:] = 0
    raw.robots[0].recent_qpos.current[:] = 99
    raw._observables["agentview_image"]._current_observed_value[:] = 0
    raw._observables["agentview_image"]._time_since_last_sample = 99
    raw._obs_cache["robot0_joint_pos"][:] = 99
    raw.rng.random(10)

    obs, restored, pre_delta, post_delta, python_delta = (
        _restore_mujoco_integration_checkpoint(env, checkpoint, state_atol=1e-12)
    )

    assert np.array_equal(data.state, original)
    assert restored.state_sha256 == checkpoint.state_sha256
    assert restored.model_xml_sha256 == checkpoint.model_xml_sha256
    assert pre_delta == post_delta == python_delta == 0.0
    assert (raw.timestep, raw.cur_time, raw.done) == (10, 0.5, False)
    assert _replay_values_equal(restored.robosuite_python_state, expected_python)
    assert (
        _robosuite_python_state_sha256(restored.robosuite_python_state)
        == expected_python_hash
    )
    assert np.all(obs["agentview_image"] == 17)
    assert raw.robots[0].controller.update_calls == 1


def test_restore_reinstates_global_random_generators(monkeypatch):
    torch = pytest.importorskip("torch")
    _install_fake_mujoco(monkeypatch)
    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(31)
    env, _, _, _ = _fake_env(np.arange(8))
    checkpoint = _checkpoint(env)

    expected = (random.random(), np.random.random(), torch.rand(1).item())
    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)

    _restore_mujoco_integration_checkpoint(env, checkpoint, state_atol=1e-12)
    restored = (random.random(), np.random.random(), torch.rand(1).item())

    assert restored == expected


def test_final_restore_is_exact_after_tolerated_forward_and_controller_drift(
    monkeypatch,
):
    _install_fake_mujoco(monkeypatch, forward_delta=5e-9)
    env, _, data, _ = _fake_env(np.arange(8), update_delta=7e-9)
    checkpoint = _checkpoint(env)

    _, restored, pre_delta, post_delta, python_delta = (
        _restore_mujoco_integration_checkpoint(env, checkpoint, state_atol=1e-6)
    )

    assert pre_delta == 0.0
    assert post_delta == pytest.approx(5e-9)
    assert python_delta == pytest.approx(12e-9)
    assert np.array_equal(data.state, checkpoint.state)
    assert restored.state_sha256 == checkpoint.state_sha256


def test_checkpoint_does_not_alias_python_buffers(monkeypatch):
    _install_fake_mujoco(monkeypatch)
    env, _, _, raw = _fake_env(np.arange(8))
    checkpoint = _checkpoint(env)
    before = _robosuite_python_state_sha256(checkpoint.robosuite_python_state)

    raw.robots[0].controller.goal_pos[:] = -7
    raw.robots[0].controller.interpolator_pos.start[:] = -7
    raw.robots[0].recent_ee_vel_buffer.buf[:] = -7
    raw.robots[0].gripper.current_action[:] = -7
    raw._observables["agentview_image"]._current_observed_value[:] = 0
    raw._obs_cache["robot0_joint_pos"][:] = -7

    assert _robosuite_python_state_sha256(checkpoint.robosuite_python_state) == before


def test_mjstate_integration_accepts_recreated_equivalent_model(monkeypatch):
    _install_fake_mujoco(monkeypatch)
    source, source_model, _, _ = _fake_env(np.arange(8))
    checkpoint = _checkpoint(source)
    target, target_model, target_data, _ = _fake_env(np.zeros(8))

    _, restored, pre_delta, post_delta, python_delta = (
        _restore_mujoco_integration_checkpoint(target, checkpoint, state_atol=1e-12)
    )

    assert source_model is not target_model
    assert np.array_equal(target_data.state, checkpoint.state)
    assert restored.model_xml_sha256 == checkpoint.model_xml_sha256
    assert pre_delta == post_delta == python_delta == 0.0


def test_mjstate_integration_rejects_different_model_xml(monkeypatch):
    _install_fake_mujoco(monkeypatch)
    source, _, _, _ = _fake_env(np.arange(8), xml="<mujoco model='source'/>")
    checkpoint = _checkpoint(source)
    target, _, _, _ = _fake_env(np.zeros(8), xml="<mujoco model='target'/>")

    with pytest.raises(RuntimeError, match="different MuJoCo model XML"):
        _restore_mujoco_integration_checkpoint(target, checkpoint, state_atol=1e-12)


def test_mjstate_integration_rejects_changed_model_dimensions(monkeypatch):
    _install_fake_mujoco(monkeypatch)
    source, _, _, _ = _fake_env(np.arange(8))
    checkpoint = _checkpoint(source)
    target, target_model, _, _ = _fake_env(np.zeros(8))
    target_model.nu = 3

    with pytest.raises(RuntimeError, match="model/state dimensions changed"):
        _restore_mujoco_integration_checkpoint(target, checkpoint, state_atol=1e-12)


def test_mjstate_integration_audits_setstate_before_forward(monkeypatch):
    _install_fake_mujoco(monkeypatch, set_delta=2e-6)
    env, _, _, _ = _fake_env(np.arange(8))
    checkpoint = _checkpoint(env)

    with pytest.raises(RuntimeError, match="mj_setState did not exactly restore"):
        _restore_mujoco_integration_checkpoint(env, checkpoint, state_atol=1e-6)


def test_mjstate_integration_audits_post_forward_drift(monkeypatch):
    _install_fake_mujoco(monkeypatch, forward_delta=2e-6)
    env, _, _, _ = _fake_env(np.arange(8))
    checkpoint = _checkpoint(env)

    with pytest.raises(RuntimeError, match="changed too much during mj_forward"):
        _restore_mujoco_integration_checkpoint(env, checkpoint, state_atol=1e-6)


def test_mjstate_integration_audits_post_python_state_drift(monkeypatch):
    _install_fake_mujoco(monkeypatch)
    env, _, _, _ = _fake_env(np.arange(8), update_delta=2e-6)
    checkpoint = _checkpoint(env)

    with pytest.raises(
        RuntimeError, match="changed too much during Python-state restore"
    ):
        _restore_mujoco_integration_checkpoint(env, checkpoint, state_atol=1e-6)


def test_restore_rejects_controller_type_mismatch(monkeypatch):
    _install_fake_mujoco(monkeypatch)
    source, _, _, _ = _fake_env(np.arange(8))
    checkpoint = _checkpoint(source)
    target, _, _, raw = _fake_env(np.zeros(8))
    raw.robots[0].controller = OtherController()

    with pytest.raises(RuntimeError, match="controller type changed"):
        _restore_mujoco_integration_checkpoint(target, checkpoint, state_atol=1e-12)


def _advance_fake_control(raw, data, action):
    robot = raw.robots[0]
    controller = robot.controller
    gripper = robot.gripper
    gripper.current_action = np.clip(
        gripper.current_action + np.asarray([-1.0, 1.0]) * action, -1.0, 1.0
    )
    influence = (
        controller.initial_joint.sum()
        + controller.goal_pos.sum()
        + controller.interpolator_pos.start.sum()
        + controller.interpolator_pos.goal.sum()
        + controller.interpolator_pos.step
        + gripper.current_action.sum()
        + robot.recent_qpos.current.sum()
    )
    data.state = data.state + influence * 1e-4
    robot.recent_qpos.last = robot.recent_qpos.current.copy()
    robot.recent_qpos.current = data.state[:2].copy()
    controller.interpolator_pos.step += 1


def test_same_actions_remain_identical_after_full_branch_restore(monkeypatch):
    _install_fake_mujoco(monkeypatch)
    normal, _, normal_data, normal_raw = _fake_env(np.arange(8))
    checkpoint = _checkpoint(normal)
    forced, _, forced_data, forced_raw = _fake_env(np.full(8, -3.0))
    forced_raw.robots[0].controller.initial_joint[:] = 50
    forced_raw.robots[0].gripper.current_action[:] = 0
    forced_raw.robots[0].recent_qpos.current[:] = 50

    _restore_mujoco_integration_checkpoint(forced, checkpoint, state_atol=1e-12)
    for action in (0.2, -0.1, 0.4, -0.3, 0.05):
        _advance_fake_control(normal_raw, normal_data, action)
        _advance_fake_control(forced_raw, forced_data, action)
        assert np.array_equal(normal_data.state, forced_data.state)
        assert np.array_equal(
            normal_raw.robots[0].gripper.current_action,
            forced_raw.robots[0].gripper.current_action,
        )
        assert np.array_equal(
            normal_raw.robots[0].recent_qpos.current,
            forced_raw.robots[0].recent_qpos.current,
        )


class FakeResetEnv:
    def __init__(self, *, hard_reset=True):
        self.env = SimpleNamespace(hard_reset=hard_reset, deterministic_reset=False)
        self.events = []
        self.models = []

    def seed(self, value):
        self.events.append(("seed", int(value)))

    def reset(self):
        self.events.append(("reset", self.env.hard_reset, self.env.deterministic_reset))
        self.models.append(object())
        return {}

    def set_init_state(self, state):
        self.events.append(("set_init_state", tuple(np.asarray(state).tolist())))
        return {"step": 0}

    def step(self, action):
        self.events.append(("dummy_step",))
        return {"step": len(self.events)}, 0.0, False, {}


def test_normal_and_forced_use_same_hard_reset_lifecycle(monkeypatch):
    seeded = []
    monkeypatch.setattr(paired_runner, "set_seed", lambda value: seeded.append(value))
    env = FakeResetEnv()
    initial_state = np.asarray([1.0, 2.0])

    _reset_to_initial_state(env, initial_state, pair_seed=17, wait=2)
    first_events = list(env.events)
    env.events.clear()
    _reset_to_initial_state(env, initial_state, pair_seed=17, wait=2)

    assert seeded == [17, 17]
    assert env.events == first_events
    assert first_events == [
        ("seed", 17),
        ("reset", True, False),
        ("set_init_state", (1.0, 2.0)),
        ("dummy_step",),
        ("dummy_step",),
    ]
    assert env.models[0] is not env.models[1]


def test_reset_rejects_hard_reset_false(monkeypatch):
    monkeypatch.setattr(paired_runner, "set_seed", lambda value: None)
    env = FakeResetEnv(hard_reset=False)

    with pytest.raises(RuntimeError, match="hard_reset=True"):
        _reset_to_initial_state(env, np.asarray([1.0, 2.0]), pair_seed=17, wait=2)
    assert env.events == []
