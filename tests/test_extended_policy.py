from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from event_sae.openvla.extended_collection.policy import (  # noqa: E402
    _clamp_probability,
    _summarize_action_scores,
    infer_action_with_uncertainty,
)


class FakeBatch(dict):
    def to(self, device, dtype=None):
        result = FakeBatch()
        for key, value in self.items():
            result[key] = value.to(
                device=device, dtype=dtype if value.is_floating_point() else None
            )
        return result


class FakeProcessor:
    def __call__(self, prompt, image):
        assert image.size == (224, 224)
        return FakeBatch(
            input_ids=torch.tensor([[1, 2]], dtype=torch.long),
            attention_mask=torch.ones((1, 2), dtype=torch.long),
            pixel_values=torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        )


class FakeModel:
    vocab_size = 100
    bin_centers = np.linspace(-0.75, 0.75, 4)
    config = SimpleNamespace(n_action_bins=5)

    def parameters(self):
        yield torch.zeros(1)

    def get_action_dim(self, key):
        return 7

    def get_action_stats(self, key):
        return {"q01": np.full(7, -2.0), "q99": np.full(7, 2.0)}

    def generate(self, input_ids, **kwargs):
        assert int(input_ids[0, -1]) == 29871
        assert kwargs["max_new_tokens"] == 7
        ids = torch.tensor([[99, 98, 97, 96, 95, 98, 97]], dtype=torch.long)
        sequences = torch.cat((input_ids, ids), dim=1)
        scores = []
        for token in ids[0]:
            score = torch.full((1, 104), -8.0)
            score[0, int(token)] = 4.0
            scores.append(score)
        return SimpleNamespace(sequences=sequences, scores=tuple(scores))


def test_single_generation_decodes_action_and_drops_logits():
    cfg = SimpleNamespace(
        model=SimpleNamespace(center_crop=False, checkpoint="openvla/test")
    )
    output = infer_action_with_uncertainty(
        FakeModel(),
        FakeProcessor(),
        cfg,
        np.zeros((224, 224, 3), dtype=np.uint8),
        "move object",
        "libero_spatial",
    )
    assert output.raw_action.shape == (7,)
    assert output.action_token_ids == [99, 98, 97, 96, 95, 98, 97]
    assert len(output.uncertainty["per_action_dimension"]) == 7
    assert output.uncertainty["action_vocab_start"] == 95
    assert output.uncertainty["action_vocab_size"] == 5
    assert all(
        row["selected_token_is_action_token"]
        for row in output.uncertainty["per_action_dimension"]
    )
    assert "logits" not in repr(output.uncertainty).lower()
    assert output.model_input_rgb.shape == (224, 224, 3)


def test_cached_normal_model_input_bypasses_preprocessing():
    cfg = SimpleNamespace(
        model=SimpleNamespace(center_crop=True, checkpoint="openvla/test")
    )
    cached = np.full((224, 224, 3), 17, dtype=np.uint8)
    output = infer_action_with_uncertainty(
        FakeModel(),
        FakeProcessor(),
        cfg,
        np.zeros((224, 224, 3), dtype=np.uint8),
        "move object",
        "libero_spatial",
        model_input_rgb_override=cached,
    )

    assert np.array_equal(output.model_input_rgb, cached)
    assert output.preprocessing["input_source"] == "normal_prefix_replay"


def test_uncertainty_is_over_action_vocabulary_only():
    scores = (torch.tensor([[1000.0, -2.0, -1.0, 3.0, 2.0]]),)
    result = _summarize_action_scores(
        scores, [3], action_vocab_end=5, action_vocab_size=2
    )
    row = result["per_action_dimension"][0]
    assert row["full_next_token"]["top1_token_id"] == 0
    assert row["conditional_action_token"]["top1_token_id"] == 3
    assert row["selected_token_conditional_rank"] == 1
    assert row["conditional_action_token"]["probability_mass_in_full_vocabulary"] < 1e-6
    assert result["action_vocab_start"] == 3


def test_lower_boundary_action_token_is_included():
    scores = (torch.zeros((1, 100)),)
    result = _summarize_action_scores(
        scores, [95], action_vocab_end=100, action_vocab_size=5
    )
    row = result["per_action_dimension"][0]
    assert result["action_vocab_start"] == 95
    assert row["selected_token_is_action_token"] is True
    assert row["selected_token_conditional_probability"] is not None


def test_inconsistent_openvla_bin_metadata_is_rejected():
    model = FakeModel()
    model.config = SimpleNamespace(n_action_bins=4)
    cfg = SimpleNamespace(
        model=SimpleNamespace(center_crop=False, checkpoint="openvla/test")
    )
    with pytest.raises(RuntimeError, match="action-bin metadata is inconsistent"):
        infer_action_with_uncertainty(
            model,
            FakeProcessor(),
            cfg,
            np.zeros((224, 224, 3), dtype=np.uint8),
            "move object",
            "libero_spatial",
        )


def test_probability_mass_roundoff_is_clamped():
    assert _clamp_probability(1.0 + torch.finfo(torch.float32).eps) == 1.0
    assert _clamp_probability(-torch.finfo(torch.float32).eps) == 0.0
