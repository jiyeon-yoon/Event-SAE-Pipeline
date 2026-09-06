from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.paired_runner import (
    PrefixComparison,
    _pair_invalid_reasons,
    _trace,
)


def _prefix_trace(
    *, observed_value: int, state_value: float, action_value: float = 0.0
):
    model_input = np.zeros((224, 224, 3), dtype=np.uint8)
    observed = np.full((224, 224, 3), observed_value, dtype=np.uint8)
    action = np.full(7, action_value, dtype=np.float64)
    state = np.asarray([state_value], dtype=np.float64)
    return _trace(
        model_input,
        observed,
        action,
        action,
        {"qpos": state, "qvel": state, "ctrl": state},
    )


def test_observed_rgb_difference_is_audit_only_when_prefix_is_controlled():
    normal = _prefix_trace(observed_value=0, state_value=0.0)
    forced = _prefix_trace(observed_value=1, state_value=5e-9)
    comparison = PrefixComparison()

    assert comparison.compare(0, normal, forced, action_atol=1e-6, state_atol=1e-6)
    assert comparison.rgb_exact is True
    assert comparison.observed_rgb_exact is False
    assert comparison.max_observed_rgb_abs_delta == 1.0
    assert comparison.stateful_sim_exact is False
    assert comparison.max_stateful_sim_abs_delta == 5e-9


def test_prefix_replay_still_rejects_material_state_drift():
    normal = _prefix_trace(observed_value=0, state_value=0.0)
    forced = _prefix_trace(observed_value=1, state_value=2e-6)

    comparison = PrefixComparison()
    assert not comparison.compare(0, normal, forced, action_atol=1e-6, state_atol=1e-6)
    assert comparison.mismatch_field == "pre_qpos"


def test_prefix_replay_still_rejects_policy_action_drift():
    normal = _prefix_trace(observed_value=0, state_value=0.0)
    forced = _prefix_trace(observed_value=1, state_value=0.0, action_value=2e-6)

    comparison = PrefixComparison()
    assert not comparison.compare(0, normal, forced, action_atol=1e-6, state_atol=1e-6)
    assert comparison.mismatch_field == "raw_openvla_action"


def test_primary_forced_failure_is_not_masked_by_warm_start_hashes():
    normal = {
        "initial_state_sha256": "initial",
        "warm_start_sim_state_sha256": "normal-sim",
        "warm_start_source_rgb_sha256": "normal-rgb",
    }
    forced = {
        "initial_state_sha256": "initial",
        "warm_start_sim_state_sha256": "forced-sim",
        "warm_start_source_rgb_sha256": "forced-rgb",
        "invalid_reason": "pretrigger_divergence",
    }

    reasons, audit = _pair_invalid_reasons(normal, forced, RuntimeError("boom"))

    assert reasons == ["pretrigger_divergence", "forced_exception"]
    assert audit == {
        "initial_state_exact": True,
        "warm_start_sim_state_exact": False,
        "warm_start_source_rgb_exact": False,
    }
