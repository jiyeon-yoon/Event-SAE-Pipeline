import io
import json
from types import SimpleNamespace

import pytest
import torch

import event_sae.openvla.intervene as intervene


class _Layer(torch.nn.Module):
    def forward(self, hidden):
        return (hidden + 1.0,)


class _SAE:
    def encode(self, value):
        return value

    def decode(self, value):
        return value * 0.5


class _FeatureSAE:
    def encode(self, value):
        # Eight SAE features; feature 2 follows hidden dimension 0.
        result = torch.zeros(value.shape[0], 8, device=value.device)
        result[:, 2] = value[:, 0]
        return result

    def decode(self, value):
        result = torch.zeros(value.shape[0], 4, device=value.device)
        result[:, 0] = value[:, 2]
        return result


def _model():
    language_model = SimpleNamespace(
        model=SimpleNamespace(layers=torch.nn.ModuleList([_Layer(), _Layer()])),
        lm_head=torch.nn.Linear(4, 1, bias=False),
    )
    return SimpleNamespace(language_model=language_model)


def _checkpoint_config(layer=1):
    return {
        "trainer": {
            "layer": layer,
            "activation_dim": 4,
            "dict_size": 8,
            "submodule_name": "post_mlp_residual",
        }
    }


def test_reconstruction_hook_replaces_hidden_and_reports_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(
        intervene,
        "load_batch_topk_sae",
        lambda checkpoint_path, device: (_SAE(), _checkpoint_config()),
    )
    model = _model()
    model._sae_hook_context = {
        "episode_num": 3,
        "task_id": 0,
        "task_episode_idx": 2,
        "step_in_episode": 7,
    }
    handle = intervene.apply_resid_post_reconstruction_hook(
        model=model,
        layer_idx=1,
        sae_checkpoint_path="unused.pt",
        run_dir=str(tmp_path),
        log_file=io.StringIO(),
    )

    source = torch.zeros(1, 2, 4, dtype=torch.float32)
    hooked = model.language_model.model.layers[1](source)[0]
    torch.testing.assert_close(hooked, torch.full_like(source, 0.5))
    summary = handle.summary()
    assert summary["num_forwards"] == 1
    assert summary["num_environment_steps"] == 1
    assert summary["num_tokens"] == 2
    handle.remove()
    handle.remove()  # cleanup is idempotent

    record = json.loads((tmp_path / "sae_reconstruction_records.jsonl").read_text())
    assert record["episode_num"] == 3
    assert record["step_in_episode"] == 7


def test_hook_rejects_checkpoint_layer_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        intervene,
        "load_batch_topk_sae",
        lambda checkpoint_path, device: (_SAE(), _checkpoint_config(layer=0)),
    )
    with pytest.raises(ValueError, match="checkpoint is layer 0"):
        intervene.apply_resid_post_reconstruction_hook(
            model=_model(),
            layer_idx=1,
            sae_checkpoint_path="unused.pt",
            run_dir=str(tmp_path),
            log_file=io.StringIO(),
        )


def test_feature_intervention_preserves_residual_and_reports_activity(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        intervene,
        "load_batch_topk_sae",
        lambda checkpoint_path, device: (_FeatureSAE(), _checkpoint_config()),
    )
    model = _model()
    model._sae_hook_context = {
        "episode_num": 1,
        "task_id": 0,
        "task_episode_idx": 0,
        "step_in_episode": 3,
    }
    handle = intervene.apply_resid_post_feature_perturb_hook(
        model=model,
        layer_idx=1,
        sae_checkpoint_path="unused.pt",
        feature_idx=2,
        alpha=0.0,
        hook_start_step=0,
        run_dir=str(tmp_path),
        log_file=io.StringIO(),
    )
    source = torch.zeros(1, 2, 4)
    # The layer creates ones; zeroing feature 2 removes only dimension 0.
    hooked = model.language_model.model.layers[1](source)[0]
    expected = torch.ones_like(source)
    expected[..., 0] = 0.0
    torch.testing.assert_close(hooked, expected)
    summary = handle.summary()
    assert summary["num_forwards"] == 1
    assert summary["active_feature_values"] == 2
    assert summary["max_feature_activation"] == 1.0
    handle.remove()
    handle.remove()
