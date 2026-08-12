"""Streaming offline fidelity evaluation for trained Event-SAE checkpoints.

The reported metrics follow the diagnostics used by dictionary_learning:

* fraction of variance explained (FVE),
* mean squared reconstruction error (MSE),
* fraction of dictionary features that fire at least once, and
* average per-token L0.

Unlike the generic upstream helper, this module reads Event-SAE .pt
activation shards directly and aggregates FVE over every evaluated token.
It never concatenates shards in memory.

Training with normalize_activations=True rescales the final checkpoint
back to the original activation scale before it is saved. Therefore the raw
activation shards must be passed to the final checkpoint without applying the
training norm factor again.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import torch
from tqdm import tqdm

from event_sae.openvla.activations import load_batch_topk_sae


@dataclass(frozen=True)
class OfflineFidelityConfig:
    """Configuration for a single offline SAE fidelity evaluation."""

    data_dirs: tuple[str, ...]
    sae_checkpoint_path: str
    output_path: str
    layer_idx: int | None = None
    batch_size: int = 1024
    device: str = ""
    max_rows: int | None = None

    def resolved_device(self) -> str:
        return self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _find_shards(data_dirs: Iterable[str], layer_idx: int) -> list[Path]:
    pattern = f"layer_{layer_idx:02d}_shard_*.pt"
    found: dict[str, Path] = {}
    for raw_path in data_dirs:
        root = Path(raw_path).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"Activation path does not exist: {root}")
        candidates = [root] if root.is_file() else root.rglob(pattern)
        for candidate in candidates:
            if candidate.is_file() and candidate.match(pattern):
                resolved = candidate.resolve()
                found[str(resolved)] = resolved
    shards = [found[key] for key in sorted(found)]
    if not shards:
        roots = ", ".join(str(Path(path).expanduser()) for path in data_dirs)
        raise FileNotFoundError(f"No {pattern} files found under: {roots}")
    return shards


def _iter_batches(
    shard_paths: Iterable[Path],
    *,
    activation_dim: int,
    batch_size: int,
    max_rows: int | None,
    progress: dict[str, int],
) -> Iterator[torch.Tensor]:
    """Yield contiguous full batches across shard boundaries, plus one tail."""

    pending: torch.Tensor | None = None
    rows_read = 0

    for shard_path in tqdm(list(shard_paths), desc="Evaluating activation shards"):
        if max_rows is not None and rows_read >= max_rows:
            break
        shard = torch.load(shard_path, map_location="cpu")
        if not isinstance(shard, torch.Tensor):
            raise TypeError(f"Expected a tensor in {shard_path}, got {type(shard)!r}")
        if shard.ndim != 2 or shard.shape[1] != activation_dim:
            raise ValueError(
                f"Expected shape [rows, {activation_dim}] in {shard_path}, "
                f"got {tuple(shard.shape)}"
            )
        shard = shard.to(dtype=torch.float32)

        if max_rows is not None:
            remaining = max_rows - rows_read
            shard = shard[:remaining]
        if shard.shape[0] == 0:
            continue
        progress["num_shards_evaluated"] += 1
        progress["shard_bytes_evaluated"] += shard_path.stat().st_size
        rows_read += int(shard.shape[0])
        progress["num_rows_loaded"] = rows_read

        if pending is not None:
            needed = batch_size - int(pending.shape[0])
            if shard.shape[0] < needed:
                pending = torch.cat((pending, shard), dim=0)
                continue
            yield torch.cat((pending, shard[:needed]), dim=0)
            shard = shard[needed:]
            pending = None

        full_rows = (int(shard.shape[0]) // batch_size) * batch_size
        for start in range(0, full_rows, batch_size):
            yield shard[start : start + batch_size]
        if full_rows < shard.shape[0]:
            pending = shard[full_rows:]

        if max_rows is not None and rows_read >= max_rows:
            break

    if pending is not None and pending.shape[0] > 0:
        yield pending


@torch.inference_mode()
def evaluate_offline_fidelity(cfg: OfflineFidelityConfig) -> dict:
    """Evaluate a final BatchTopK SAE checkpoint on dense activation shards."""

    if cfg.batch_size < 2:
        raise ValueError("batch_size must be at least 2 for variance evaluation")
    if cfg.max_rows is not None and cfg.max_rows < 2:
        raise ValueError("max_rows must be at least 2 when provided")

    device = cfg.resolved_device()
    checkpoint_path = Path(cfg.sae_checkpoint_path).expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "ae.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAE checkpoint not found: {checkpoint_path}")

    sae, checkpoint_config = load_batch_topk_sae(checkpoint_path, device=device)
    trainer_cfg = checkpoint_config["trainer"]
    checkpoint_layer = int(trainer_cfg["layer"])
    layer_idx = checkpoint_layer if cfg.layer_idx is None else int(cfg.layer_idx)
    if layer_idx != checkpoint_layer:
        raise ValueError(
            f"Requested layer {layer_idx}, but checkpoint config is layer {checkpoint_layer}"
        )

    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    shard_paths = _find_shards(cfg.data_dirs, layer_idx)
    progress = {
        "num_shards_evaluated": 0,
        "shard_bytes_evaluated": 0,
        "num_rows_loaded": 0,
    }

    # Global sufficient statistics. Float64 prevents cancellation when the
    # dataset contains millions of high-norm activation vectors.
    sum_x = torch.zeros(activation_dim, dtype=torch.float64, device=device)
    sum_residual = torch.zeros_like(sum_x)
    sum_x_squared = torch.zeros((), dtype=torch.float64, device=device)
    sum_residual_squared = torch.zeros_like(sum_x_squared)
    alive = torch.zeros(dict_size, dtype=torch.bool, device=device)
    nonzero_count = torch.zeros((), dtype=torch.float64, device=device)
    finite_inputs = torch.ones((), dtype=torch.bool, device=device)
    finite_reconstructions = torch.ones((), dtype=torch.bool, device=device)
    finite_features = torch.ones((), dtype=torch.bool, device=device)
    num_rows = 0
    num_batches = 0

    for cpu_batch in _iter_batches(
        shard_paths,
        activation_dim=activation_dim,
        batch_size=cfg.batch_size,
        max_rows=cfg.max_rows,
        progress=progress,
    ):
        batch = cpu_batch.to(device=device, dtype=torch.float32, non_blocking=True)
        reconstruction, features = sae(batch, output_features=True)
        residual = batch - reconstruction
        finite_inputs &= torch.isfinite(batch).all()
        finite_reconstructions &= torch.isfinite(reconstruction).all()
        finite_features &= torch.isfinite(features).all()

        x64 = batch.to(dtype=torch.float64)
        residual64 = residual.to(dtype=torch.float64)
        sum_x += x64.sum(dim=0)
        sum_residual += residual64.sum(dim=0)
        sum_x_squared += x64.square().sum()
        sum_residual_squared += residual64.square().sum()

        active_mask = features != 0
        alive |= active_mask.any(dim=0)
        nonzero_count += active_mask.sum(dtype=torch.float64)
        num_rows += int(batch.shape[0])
        num_batches += 1

    if num_rows < 2:
        raise RuntimeError("Evaluation produced fewer than two activation rows")
    finite_status = {
        "inputs": bool(finite_inputs.item()),
        "reconstructions": bool(finite_reconstructions.item()),
        "features": bool(finite_features.item()),
    }
    if not all(finite_status.values()):
        failed = ", ".join(name for name, valid in finite_status.items() if not valid)
        raise FloatingPointError(f"NaN or Inf detected in: {failed}")

    count = float(num_rows)
    total_variation = sum_x_squared - sum_x.square().sum() / count
    residual_variation = sum_residual_squared - sum_residual.square().sum() / count
    if total_variation.item() <= 0:
        raise RuntimeError("Activation variance is zero; FVE is undefined")

    fve = 1.0 - residual_variation / total_variation
    # Element-wise MSE over the complete [num_rows, activation_dim] tensor.
    # sum_residual_squared is already required for global FVE, so reporting
    # MSE adds neither another dataset pass nor an extra reconstruction.
    reconstruction_mse = sum_residual_squared / (count * activation_dim)
    alive_count = int(alive.sum().item())
    result = {
        "schema_version": "event_sae_offline_fidelity_v1",
        "evaluation_scope": "provided_activation_shards",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
            "layer_idx": layer_idx,
            "submodule_name": trainer_cfg.get("submodule_name"),
            "activation_dim": activation_dim,
            "dict_size": dict_size,
            "k": int(trainer_cfg["k"]),
            "threshold": float(sae.threshold.item()),
        },
        "data": {
            "roots": [str(Path(path).expanduser().resolve()) for path in cfg.data_dirs],
            "num_shards_discovered": len(shard_paths),
            "num_shards_evaluated": progress["num_shards_evaluated"],
            "shard_bytes_evaluated": progress["shard_bytes_evaluated"],
            "num_rows": num_rows,
            "num_batches": num_batches,
            "batch_size": cfg.batch_size,
            "max_rows": cfg.max_rows,
            "selection": (
                "all_rows" if cfg.max_rows is None else "sorted_path_prefix_smoke_test"
            ),
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device": device,
            "activation_dtype": "float32",
            "statistics_dtype": "float64",
        },
        "metrics": {
            "frac_variance_explained": float(fve.item()),
            "reconstruction_mse": float(reconstruction_mse.item()),
            "fraction_alive": alive_count / dict_size,
            "alive_percent": 100.0 * alive_count / dict_size,
            "alive_features": alive_count,
            "average_l0": float((nonzero_count / count).item()),
        },
        "notes": [
            "Raw activation shards were evaluated without applying the training norm factor again.",
            "FVE is aggregated globally over every evaluated token, not averaged per shard.",
            "Reconstruction MSE is the element-wise mean of (activation - reconstruction)^2.",
            "If these are the training shards, this is in-sample rather than held-out fidelity.",
            "A max_rows result is a sorted-prefix smoke test and must not be used for paper comparison.",
            "Alive fraction is comparable only when the evaluation population is held fixed.",
        ],
    }

    output_path = Path(cfg.output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return result
