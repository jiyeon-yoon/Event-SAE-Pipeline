import pytest

from scripts.openvla.compare_hooked_sr import compare_results


def _result(mode: str) -> dict:
    return {
        "mode": mode,
        "completed_rollouts": 100,
        "success_rate": 0.9 if mode == "raw" else 0.88,
        "model_checkpoint": "openvla/model",
        "model_revision": "weights",
        "model_code_revision": "code",
        "task_suite": "libero_spatial",
        "seed": 0,
        "requested_task_ids": "",
        "num_trials_per_task": 10,
        "code": {"commit": "source", "dirty": False},
        "run_config": {"model": {"revision": "fixed"}, "env": {"seed": 0}},
        "hook_metrics": {"num_forwards": 4, "num_tokens": 8},
    }


def test_compare_hooked_sr_requires_matching_protocols():
    raw = _result("raw")
    reconstructed = _result("reconstruction")
    result = compare_results(raw, reconstructed)
    assert result["absolute_drop"] == pytest.approx(0.02)

    reconstructed["model_revision"] = "different"
    with pytest.raises(RuntimeError, match="protocols differ"):
        compare_results(raw, reconstructed)


def test_compare_hooked_sr_rejects_wrong_revision_and_dirty_reconstruction():
    raw = _result("raw")
    reconstructed = _result("reconstruction")
    with pytest.raises(RuntimeError, match="expected-code-revision"):
        compare_results(raw, reconstructed, expected_code_revision="other")

    reconstructed["code"] = {"commit": "source", "dirty": True}
    raw["code"] = reconstructed["code"]
    with pytest.raises(RuntimeError, match="clean Git"):
        compare_results(raw, reconstructed, expected_code_revision="source")


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("code", {"commit": "source"}, "clean Git"),
        ("run_config", None, "run_config"),
        ("run_config", {}, "run_config"),
    ],
)
def test_compare_hooked_sr_requires_complete_provenance(field, value, match):
    raw = _result("raw")
    reconstructed = _result("reconstruction")
    raw[field] = value
    reconstructed[field] = value
    with pytest.raises(RuntimeError, match=match):
        compare_results(raw, reconstructed, expected_code_revision="source")
