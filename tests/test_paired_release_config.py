from __future__ import annotations

from pathlib import Path

import pytest

from event_sae.openvla.extended_collection.config import (
    load_paired_release_config,
    parse_overrides,
)


BASE = """
model:
  revision: 962318cec55ac10993ff0f5f43eda9a270b4c873
  code_revision: 47a0ec7fc4ec123775a391911046cf33cf9ed83f
collection:
  layer_idx: 31
paired_release:
  stable_grasp_steps: 3
  min_lift_delta_m: 0.02
  max_force_open_steps: 20
  stable_detach_steps: 2
"""


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "paired.yaml"
    path.write_text(BASE, encoding="utf-8")
    return path


def test_paired_config_loads_mappings_and_overrides(tmp_path: Path):
    cfg = load_paired_release_config(
        _config(tmp_path),
        parse_overrides(
            [
                "env.task_ids=0,1",
                "paired_release.target_object_by_task={0: bowl}",
            ]
        ),
    )
    assert cfg.env.task_ids == "0,1"
    assert cfg.paired_release.target_object_by_task == {"0": "bowl"}
    assert cfg.paired_release.target_valid_pairs_per_task == 20
    assert cfg.paired_release.target_primary_pairs_per_task == 20
    assert cfg.paired_release.normal_post_success_steps == 20
    assert cfg.paired_release.post_detach_goal_stable_steps == 2


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("paired_release.stable_grasp_steps=0", "stable_grasp_steps"),
        ("paired_release.min_lift_delta_m=0", "min_lift_delta_m"),
        ("paired_release.trigger_delay_steps=-1", "trigger_delay_steps"),
        ("paired_release.max_force_open_steps=0", "max_force_open_steps"),
        ("paired_release.stable_detach_steps=0", "stable_detach_steps"),
        (
            "paired_release.normal_post_success_steps=0",
            "normal_post_success_steps",
        ),
        (
            "paired_release.post_detach_goal_stable_steps=0",
            "post_detach_goal_stable_steps",
        ),
        ("paired_release.forced_gripper_value=1", "forced_gripper_value"),
        ("paired_release.action_atol=-1", "tolerances"),
        (
            "paired_release.target_valid_pairs_per_task=0",
            "target_valid_pairs_per_task",
        ),
        (
            "paired_release.target_primary_pairs_per_task=-1",
            "target_primary_pairs_per_task",
        ),
        (
            "paired_release.target_valid_pairs_per_task=51",
            "maximum attempts",
        ),
        (
            "paired_release.target_primary_pairs_per_task=51",
            "maximum attempts",
        ),
        ("paired_release.min_free_disk_gb_at_start=0", "min_free_disk"),
        ("paired_release.abort_below_free_disk_gb=0", "abort_below"),
        ("paired_release.min_free_disk_gb_at_start=10", "disk guard"),
        ("paired_release.disk_check_every_steps=0", "disk_check"),
        ("output.save_model_input_rgb=false", "save_model_input_rgb"),
        ("collection.fail_fast=false", "fail_fast"),
    ],
)
def test_paired_config_rejects_unsafe_values(
    tmp_path: Path,
    override: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        load_paired_release_config(
            _config(tmp_path),
            parse_overrides([override]),
        )


def test_paired_config_rejects_primary_target_above_valid_target(
    tmp_path: Path,
):
    with pytest.raises(ValueError, match="cannot exceed target_valid"):
        load_paired_release_config(
            _config(tmp_path),
            parse_overrides(
                [
                    "paired_release.target_valid_pairs_per_task=10",
                    "paired_release.target_primary_pairs_per_task=11",
                ]
            ),
        )


def test_paired_config_requires_pinned_model_revisions(tmp_path: Path):
    path = _config(tmp_path)
    path.write_text(
        BASE.replace(
            "  revision: 962318cec55ac10993ff0f5f43eda9a270b4c873\n",
            "  revision: ''\n",
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pinned"):
        load_paired_release_config(path)
