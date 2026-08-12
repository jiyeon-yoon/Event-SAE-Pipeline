import json
from pathlib import Path

import torch

import event_sae.evaluate as evaluate_module
from event_sae.evaluate import OfflineFidelityConfig, evaluate_offline_fidelity


class _HalfReconstructionSAE:
    threshold = torch.tensor(0.1)

    def __call__(self, batch, output_features=False):
        features = torch.stack(
            (batch[:, 0], torch.zeros_like(batch[:, 0]), batch[:, 1]), dim=1
        )
        return batch * 0.5, features


def test_offline_fidelity_aggregates_all_shards(monkeypatch, tmp_path: Path):
    data_dir = tmp_path / "dense"
    data_dir.mkdir()
    x = torch.tensor([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0], [8.0, 16.0]])
    torch.save(x[:2], data_dir / "layer_31_shard_000000.pt")
    torch.save(x[2:], data_dir / "layer_31_shard_000001.pt")
    checkpoint = tmp_path / "ae.pt"
    checkpoint.write_bytes(b"checkpoint")
    output = tmp_path / "metrics.json"

    monkeypatch.setattr(
        evaluate_module,
        "load_batch_topk_sae",
        lambda path, device: (
            _HalfReconstructionSAE(),
            {
                "trainer": {
                    "layer": 31,
                    "activation_dim": 2,
                    "dict_size": 3,
                    "k": 2,
                    "submodule_name": "post_mlp_residual",
                }
            },
        ),
    )
    result = evaluate_offline_fidelity(
        OfflineFidelityConfig(
            data_dirs=(str(data_dir),),
            sae_checkpoint_path=str(checkpoint),
            output_path=str(output),
            batch_size=2,
            device="cpu",
        )
    )

    residual = x * 0.5
    expected_fve = 1.0 - torch.var(residual, dim=0).sum() / torch.var(x, dim=0).sum()
    assert result["data"]["num_rows"] == 4
    assert result["data"]["num_shards_evaluated"] == 2
    assert abs(result["metrics"]["frac_variance_explained"] - float(expected_fve)) < 1e-7
    assert abs(result["metrics"]["reconstruction_mse"] - float(residual.square().mean())) < 1e-7
    assert result["metrics"]["alive_features"] == 2
    assert json.loads(output.read_text())["schema_version"] == "event_sae_offline_fidelity_v1"
