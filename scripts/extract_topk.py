"""CLI: offline extract top-k SAE activations from dense shards.

``--dense-dir`` is **the directory that contains ``activation_index.jsonl``**;
each record's ``shard_path`` is resolved relative to that directory.

Both layouts therefore work without code changes:

* **OpenPI**: pass the eval run root, e.g.
  ``logs/openpi/sae_collection/<run>/``. Index lives at the root and
  ``shard_path`` is ``sae_activations/post_mlp_residual/layer_NN_shard_*.pt``.
* **OpenVLA**: pass the per-target subdir, e.g.
  ``logs/openvla/<run>/sae_activations/post_mlp_residual/``. Index and
  shards are siblings inside it.

Reads:
  - Dense residual shards resolved from ``shard_path`` in the index.
  - Trained ``BatchTopKSAE`` checkpoint (``ae.pt`` with sibling ``config.json``).

Writes (under ``--output-dir``, default ``{dense_dir}/topk_activations``):
  - ``shard_NNNNNN.pt`` with sparse top-k rows + metadata
  - ``manifest.json`` in ``token_topk_sparse_v1`` format (same as online mode)

The output is byte-format-compatible with
``event_sae.openvla.activations.apply_sae_topk_collect_hooks``, so
downstream scoring can consume either online or offline shards uniformly.
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.openvla.activations import load_batch_topk_sae


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sparse_shard(path: Path, metadata: dict, *, topk: int) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Resume manifest references missing shard: {path}")
    expected_bytes = metadata.get("output_size_bytes")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise ValueError(f"Resume shard size mismatch: {path}")
    try:
        payload = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:  # PyTorch versions before mmap= support.
        payload = torch.load(path, map_location="cpu")
    required = {
        "episode_num",
        "step_in_episode",
        "global_forward_idx",
        "token_idx",
        "top_feature_ids",
        "top_feature_vals",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Resume shard {path} is missing keys: {sorted(missing)}")
    rows = int(metadata["num_rows"])
    if tuple(payload["top_feature_ids"].shape) != (rows, topk):
        raise ValueError(f"Resume shard has wrong top_feature_ids shape: {path}")
    if tuple(payload["top_feature_vals"].shape) != (rows, topk):
        raise ValueError(f"Resume shard has wrong top_feature_vals shape: {path}")
    for key in ("episode_num", "step_in_episode", "global_forward_idx", "token_idx"):
        if tuple(payload[key].shape) != (rows,):
            raise ValueError(f"Resume shard has wrong {key} shape: {path}")


def _encode_topk_in_batches(
    sae,
    dense: torch.Tensor,
    *,
    device: str,
    topk: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a shard without materializing its full SAE latent matrix."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    value_batches: list[torch.Tensor] = []
    index_batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, int(dense.shape[0]), batch_size):
            batch = dense[start : start + batch_size].to(device=device, dtype=torch.float32)
            encoded = sae.encode(batch)
            values, indices = torch.topk(encoded, k=topk, dim=-1)
            value_batches.append(values.to(dtype=torch.float32, device="cpu"))
            index_batches.append(indices.to(dtype=torch.int32, device="cpu"))
            del batch, encoded, values, indices
    return torch.cat(value_batches, dim=0), torch.cat(index_batches, dim=0)


def _merge_welford(
    count_a: int,
    mean_a: torch.Tensor,
    m2_a: torch.Tensor,
    count_b: int,
    mean_b: torch.Tensor,
    m2_b: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Merge vector means and their scalar, per-dimension centered sum of squares."""
    if count_a == 0:
        return count_b, mean_b, m2_b
    if count_b == 0:
        return count_a, mean_a, m2_a
    total = count_a + count_b
    delta = mean_b - mean_a
    mean = mean_a + delta * (count_b / total)
    m2 = m2_a + m2_b + delta.square().sum() * (count_a * count_b / total)
    return total, mean, m2


def _encode_topk_and_fidelity_in_batches(
    sae,
    dense: torch.Tensor,
    *,
    device: str,
    topk: int,
    batch_size: int,
    dict_size: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Encode sparse rows and collect stable fidelity sufficient statistics once."""
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    value_batches: list[torch.Tensor] = []
    index_batches: list[torch.Tensor] = []
    activation_dim = int(dense.shape[1])
    count = 0
    mean_x = torch.zeros(activation_dim, dtype=torch.float32, device=device)
    mean_residual = torch.zeros_like(mean_x)
    m2_x = torch.zeros((), dtype=torch.float32, device=device)
    m2_residual = torch.zeros_like(m2_x)
    alive = torch.zeros(dict_size, dtype=torch.bool, device=device)
    nonzero_count = torch.zeros((), dtype=torch.int64, device=device)
    finite_inputs = torch.ones((), dtype=torch.bool, device=device)
    finite_reconstructions = torch.ones_like(finite_inputs)
    finite_features = torch.ones_like(finite_inputs)

    with torch.inference_mode():
        for start in range(0, int(dense.shape[0]), batch_size):
            batch = dense[start : start + batch_size].to(device=device, dtype=torch.float32)
            reconstruction, encoded = sae(batch, output_features=True)
            values, indices = torch.topk(encoded, k=topk, dim=-1)
            value_batches.append(values.to(dtype=torch.float32, device="cpu"))
            index_batches.append(indices.to(dtype=torch.int32, device="cpu"))

            residual = batch - reconstruction
            batch_count = int(batch.shape[0])
            batch_mean_x = batch.mean(dim=0)
            batch_mean_residual = residual.mean(dim=0)
            batch_m2_x = (batch - batch_mean_x).square().sum()
            batch_m2_residual = (residual - batch_mean_residual).square().sum()
            count, mean_x, m2_x = _merge_welford(
                count,
                mean_x,
                m2_x,
                batch_count,
                batch_mean_x,
                batch_m2_x,
            )
            # Both streams contain the same number of rows, so merge against
            # the previous count rather than the count just updated above.
            previous_count = count - batch_count
            _, mean_residual, m2_residual = _merge_welford(
                previous_count,
                mean_residual,
                m2_residual,
                batch_count,
                batch_mean_residual,
                batch_m2_residual,
            )

            active = encoded != 0
            alive |= active.any(dim=0)
            nonzero_count += active.sum(dtype=torch.int64)
            finite_inputs &= torch.isfinite(batch).all()
            finite_reconstructions &= torch.isfinite(reconstruction).all()
            finite_features &= torch.isfinite(encoded).all()
            del batch, reconstruction, residual, encoded, values, indices, active

    metrics = {
        "num_rows": count,
        "mean_x": mean_x.to(dtype=torch.float64, device="cpu"),
        "m2_x": m2_x.to(dtype=torch.float64, device="cpu"),
        "mean_residual": mean_residual.to(dtype=torch.float64, device="cpu"),
        "m2_residual": m2_residual.to(dtype=torch.float64, device="cpu"),
        "alive": alive.to(device="cpu"),
        "nonzero_count": nonzero_count.to(device="cpu"),
        "finite_inputs": finite_inputs.to(device="cpu"),
        "finite_reconstructions": finite_reconstructions.to(device="cpu"),
        "finite_features": finite_features.to(device="cpu"),
    }
    return (
        torch.cat(value_batches, dim=0),
        torch.cat(index_batches, dim=0),
        metrics,
    )


def _validate_fidelity_shard(
    path: Path,
    metadata: dict,
    *,
    activation_dim: int,
    dict_size: int,
) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Resume manifest references missing fidelity shard: {path}")
    if path.stat().st_size != int(metadata["fidelity_size_bytes"]):
        raise ValueError(f"Resume fidelity shard size mismatch: {path}")
    payload = torch.load(path, map_location="cpu")
    if int(payload.get("num_rows", -1)) != int(metadata["num_rows"]):
        raise ValueError(f"Resume fidelity row count mismatch: {path}")
    for key in ("mean_x", "mean_residual"):
        if tuple(payload[key].shape) != (activation_dim,):
            raise ValueError(f"Resume fidelity shard has wrong {key} shape: {path}")
    if tuple(payload["alive"].shape) != (dict_size,):
        raise ValueError(f"Resume fidelity shard has wrong alive shape: {path}")


def _aggregate_fidelity_shards(
    *,
    output_dir: Path,
    manifest: dict,
    fidelity_output: Path,
    trainer_cfg: dict,
    checkpoint_path: Path,
    sae,
    batch_size: int,
    device: str,
) -> dict:
    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    count = 0
    mean_x = torch.zeros(activation_dim, dtype=torch.float64)
    mean_residual = torch.zeros_like(mean_x)
    m2_x = torch.zeros((), dtype=torch.float64)
    m2_residual = torch.zeros_like(m2_x)
    alive = torch.zeros(dict_size, dtype=torch.bool)
    nonzero_count = torch.zeros((), dtype=torch.int64)
    finite = {"inputs": True, "reconstructions": True, "features": True}

    for shard_meta in manifest["shards"]:
        stats = torch.load(output_dir / shard_meta["fidelity_path"], map_location="cpu")
        shard_count = int(stats["num_rows"])
        previous_count = count
        count, mean_x, m2_x = _merge_welford(
            previous_count,
            mean_x,
            m2_x,
            shard_count,
            stats["mean_x"],
            stats["m2_x"],
        )
        _, mean_residual, m2_residual = _merge_welford(
            previous_count,
            mean_residual,
            m2_residual,
            shard_count,
            stats["mean_residual"],
            stats["m2_residual"],
        )
        alive |= stats["alive"].to(dtype=torch.bool)
        nonzero_count += stats["nonzero_count"].to(dtype=torch.int64)
        finite["inputs"] &= bool(stats["finite_inputs"].item())
        finite["reconstructions"] &= bool(stats["finite_reconstructions"].item())
        finite["features"] &= bool(stats["finite_features"].item())

    if count < 2 or m2_x.item() <= 0:
        raise RuntimeError("Fidelity aggregation requires at least two non-constant rows")
    if not all(finite.values()):
        failed = ", ".join(name for name, valid in finite.items() if not valid)
        raise FloatingPointError(f"NaN or Inf detected in: {failed}")

    residual_sum_squared = m2_residual + count * mean_residual.square().sum()
    alive_count = int(alive.sum().item())
    result = {
        "schema_version": "event_sae_offline_fidelity_v1",
        "evaluation_scope": "all_rows_encoded_for_sparse_topk",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": manifest["sae_sha256"],
            "layer_idx": int(trainer_cfg["layer"]),
            "submodule_name": trainer_cfg.get("submodule_name"),
            "activation_dim": activation_dim,
            "dict_size": dict_size,
            "k": int(trainer_cfg["k"]),
            "threshold": float(sae.threshold.item()),
        },
        "data": {
            "root": str(Path(manifest["dense_dir"])),
            "num_shards_evaluated": int(manifest["num_shards"]),
            "num_rows": count,
            "batch_size": batch_size,
            "selection": "all_rows",
        },
        "runtime": {
            "torch_version": torch.__version__,
            "device": device,
            "activation_dtype": "float32",
            "statistics": "within-batch Welford float32; cross-shard Welford float64",
        },
        "metrics": {
            "frac_variance_explained": float((1.0 - m2_residual / m2_x).item()),
            "reconstruction_mse": float(
                (residual_sum_squared / (count * activation_dim)).item()
            ),
            "fraction_alive": alive_count / dict_size,
            "alive_percent": 100.0 * alive_count / dict_size,
            "alive_features": alive_count,
            "average_l0": float((nonzero_count.to(torch.float64) / count).item()),
        },
        "notes": [
            "Raw activation shards were evaluated without reapplying the training norm factor.",
            "FVE is aggregated globally over all encoded rows, not averaged per shard.",
            "This is in-sample fidelity because these activation shards trained the SAE.",
            "Sparse Top-K extraction and fidelity used the same SAE forward pass and data pass.",
        ],
    }
    fidelity_output.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest_atomic(fidelity_output, result)
    return result


def _load_index(index_path: Path, layer_idx: int) -> dict[str, list[dict]]:
    by_shard: dict[str, list[dict]] = defaultdict(list)
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if int(record["layer_idx"]) != layer_idx:
                continue
            by_shard[str(record["shard_path"])].append(record)
    for records in by_shard.values():
        records.sort(key=lambda r: int(r["row_start"]))
    return by_shard


def _write_manifest_atomic(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline SAE encode of dense activation shards.")
    parser.add_argument(
        "--dense-dir",
        required=True,
        help=(
            "Directory that contains activation_index.jsonl; shard paths in the "
            "index are resolved relative to it. For OpenPI eval runs pass the run "
            "root (the index sits at the root). For OpenVLA collection runs pass "
            "the per-target subdir (e.g. sae_activations/post_mlp_residual)."
        ),
    )
    parser.add_argument("--sae-checkpoint", required=True, help="Path to trained ae.pt")
    parser.add_argument("--layer-idx", type=int, required=True)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help=(
            "Rows encoded at once (default: 1024). This bounds the temporary "
            "[batch, dict_size] SAE latent tensor."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output dir (default: {dense_dir}/topk_activations).",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available else cpu).")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from an existing compatible manifest, validating every completed shard.",
    )
    parser.add_argument(
        "--fidelity-output",
        default=None,
        help=(
            "Optional JSON path. When set, FVE/MSE/alive/L0 are computed during "
            "the same SAE pass used for sparse Top-K extraction."
        ),
    )
    args = parser.parse_args()

    dense_dir = Path(args.dense_dir).resolve()
    index_path = dense_dir / "activation_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing {index_path}")

    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir is not None
        else (dense_dir / "topk_activations").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.sae_checkpoint).expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path = checkpoint_path / "ae.pt"
    sae, config = load_batch_topk_sae(checkpoint_path, device=device)
    trainer_cfg = config["trainer"]
    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    checkpoint_layer = int(trainer_cfg["layer"])
    if checkpoint_layer != args.layer_idx:
        raise ValueError(
            f"Requested layer {args.layer_idx}, but SAE checkpoint is layer {checkpoint_layer}"
        )
    if trainer_cfg.get("submodule_name") != "post_mlp_residual":
        raise ValueError(
            "Expected post_mlp_residual SAE, got "
            f"{trainer_cfg.get('submodule_name')!r}"
        )
    if not (1 <= args.topk <= dict_size):
        raise ValueError(f"topk must be in [1, {dict_size}], got {args.topk}")
    if args.batch_size <= 0:
        raise ValueError(f"batch-size must be positive, got {args.batch_size}")

    index = _load_index(index_path, layer_idx=args.layer_idx)
    if not index:
        raise RuntimeError(f"No index records found for layer {args.layer_idx} in {index_path}")

    # Probe a record for OpenPI-specific fields. ``capture_target`` and
    # ``executed_chunk_len`` propagate to the manifest so the scorer can
    # auto-detect step_mapping (action_executed for AE, chunk_executed for
    # PG, inference_step for OpenVLA legacy).
    probe_record = next(iter(index.values()))[0]
    capture_target = probe_record.get("capture_target")
    executed_chunk_len_seen = sorted({
        int(r["executed_chunk_len"])
        for records in index.values()
        for r in records
        if r.get("executed_chunk_len") is not None
    })

    fresh_manifest = {
        "format": "token_topk_sparse_v1",
        "layer": args.layer_idx,
        "sae_path": str(checkpoint_path),
        "sae_sha256": _sha256(checkpoint_path),
        "activation_index_sha256": _sha256(index_path),
        "dense_dir": str(dense_dir),
        "fidelity_enabled": args.fidelity_output is not None,
        "dict_size": dict_size,
        "activation_dim": activation_dim,
        "topk": args.topk,
        "capture_target": capture_target,
        "executed_chunk_lens_seen": executed_chunk_len_seen,
        "num_shards": 0,
        "total_rows": 0,
        "shards": [],
    }

    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{manifest_path} already exists. Pass --resume or choose a new --output-dir."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in (
            "format",
            "layer",
            "dict_size",
            "activation_dim",
            "topk",
            "sae_sha256",
            "activation_index_sha256",
            "fidelity_enabled",
        ):
            if manifest.get(key) != fresh_manifest.get(key):
                raise ValueError(
                    f"Cannot resume: manifest {key}={manifest.get(key)!r}, "
                    f"current={fresh_manifest.get(key)!r}"
                )
    else:
        manifest = fresh_manifest

    completed_sources: set[str] = set()
    expected_row_start = 0
    for shard_meta in manifest["shards"]:
        source_shard = shard_meta.get("source_shard")
        if source_shard is None:
            raise ValueError("Cannot resume a legacy manifest without per-shard source_shard fields.")
        out_path = output_dir / str(shard_meta["path"])
        if int(shard_meta["row_start"]) != expected_row_start:
            raise ValueError("Resume manifest row ranges are not contiguous.")
        source_path = dense_dir / str(source_shard)
        if not source_path.is_file():
            raise FileNotFoundError(f"Resume source shard is missing: {source_path}")
        if source_path.stat().st_size != int(shard_meta["source_size_bytes"]):
            raise ValueError(f"Resume source shard size mismatch: {source_path}")
        _validate_sparse_shard(out_path, shard_meta, topk=args.topk)
        if args.fidelity_output is not None:
            fidelity_path = shard_meta.get("fidelity_path")
            if fidelity_path is None:
                raise ValueError("Cannot resume: completed shard has no fidelity statistics.")
            _validate_fidelity_shard(
                output_dir / str(fidelity_path),
                shard_meta,
                activation_dim=activation_dim,
                dict_size=dict_size,
            )
        expected_row_start = int(shard_meta["row_end"])
        completed_sources.add(str(source_shard))
    manifest["num_shards"] = len(manifest["shards"])
    manifest["total_rows"] = expected_row_start

    shard_names = sorted(index.keys())
    total_rows = int(manifest["total_rows"])
    for src_shard_name in shard_names:
        if src_shard_name in completed_sources:
            print(f"Already encoded: {src_shard_name}")
            continue
        src_shard_path = dense_dir / src_shard_name
        if not src_shard_path.is_file():
            raise FileNotFoundError(f"Missing dense shard: {src_shard_path}")
        dense = torch.load(src_shard_path, map_location="cpu").to(torch.float32)
        if dense.ndim != 2 or int(dense.shape[1]) != activation_dim:
            raise ValueError(
                f"Unexpected dense shard shape {tuple(dense.shape)} in {src_shard_path}; "
                f"expected (N, {activation_dim})"
            )
        records = index[src_shard_name]
        n_rows = int(dense.shape[0])
        expected_row = 0
        for record in records:
            r0, r1 = int(record["row_start"]), int(record["row_end"])
            if r0 != expected_row or r1 <= r0 or r1 > n_rows:
                raise ValueError(
                    f"Non-contiguous or invalid index range [{r0}, {r1}) in "
                    f"{src_shard_name}; expected row_start={expected_row}, n_rows={n_rows}"
                )
            expected_row = r1
        if expected_row != n_rows:
            raise ValueError(
                f"Index covers {expected_row} of {n_rows} rows in {src_shard_name}"
            )

        episode_num = torch.zeros((n_rows,), dtype=torch.int64)
        step_in_episode = torch.zeros((n_rows,), dtype=torch.int64)
        global_forward_idx = torch.zeros((n_rows,), dtype=torch.int64)
        token_idx = torch.zeros((n_rows,), dtype=torch.int64)
        batch_idx = torch.zeros((n_rows,), dtype=torch.int64)
        # OpenPI-only: per-row chunk_start_step + executed_chunk_len so
        # `score_cluster_features` can compute `_effective_steps` under the
        # chosen step_mapping. OpenVLA records leave these at -1 (sentinel).
        chunk_start_step = torch.full((n_rows,), -1, dtype=torch.int64)
        executed_chunk_len = torch.full((n_rows,), -1, dtype=torch.int64)
        for record in records:
            r0, r1 = int(record["row_start"]), int(record["row_end"])
            episode_num[r0:r1] = int(record.get("episode_num") or 0)
            global_forward_idx[r0:r1] = int(record.get("global_forward_idx") or 0)
            # Per-token env-step mapping. OpenPI records each forward as one
            # chunked inference covering `seq_len` future tokens; the env step
            # a token corresponds to is `chunk_start + token_idx`. OpenVLA
            # records each forward as one env-step (no chunking), and all
            # rows of a record share the same `step_in_episode`. Matches
            # openpi-mech's `step_mapping="action_executed"` semantics.
            tokens_local = torch.arange(r1 - r0, dtype=torch.int64)
            token_idx[r0:r1] = tokens_local
            chunk_start = record.get("chunk_start_step")
            if chunk_start is None:
                chunk_start = record.get("action_chunk_start_step")
            if chunk_start is not None:
                chunk_start_step[r0:r1] = int(chunk_start)
                step_in_episode[r0:r1] = int(chunk_start) + tokens_local
            else:
                step_in_episode[r0:r1] = int(record.get("step_in_episode") or 0)
            if record.get("executed_chunk_len") is not None:
                executed_chunk_len[r0:r1] = int(record["executed_chunk_len"])

        fidelity_stats = None
        if args.fidelity_output is None:
            values, indices = _encode_topk_in_batches(
                sae,
                dense,
                device=device,
                topk=args.topk,
                batch_size=args.batch_size,
            )
        else:
            values, indices, fidelity_stats = _encode_topk_and_fidelity_in_batches(
                sae,
                dense,
                device=device,
                topk=args.topk,
                batch_size=args.batch_size,
                dict_size=dict_size,
            )

        out_shard_name = f"shard_{manifest['num_shards']:06d}.pt"
        output_shard_path = output_dir / out_shard_name
        torch.save(
            {
                "episode_num": episode_num,
                "step_in_episode": step_in_episode,
                "global_forward_idx": global_forward_idx,
                "batch_idx": batch_idx,
                "token_idx": token_idx,
                "chunk_start_step": chunk_start_step,
                "executed_chunk_len": executed_chunk_len,
                "top_feature_ids": indices,
                "top_feature_vals": values,
            },
            output_shard_path,
        )
        shard_metadata = {
                "shard_idx": manifest["num_shards"],
                "path": out_shard_name,
                "source_shard": src_shard_name,
                "source_size_bytes": src_shard_path.stat().st_size,
                "output_size_bytes": output_shard_path.stat().st_size,
                "num_rows": n_rows,
                "row_start": total_rows,
                "row_end": total_rows + n_rows,
        }
        if fidelity_stats is not None:
            fidelity_shard_name = f"fidelity_shard_{manifest['num_shards']:06d}.pt"
            fidelity_shard_path = output_dir / fidelity_shard_name
            torch.save(fidelity_stats, fidelity_shard_path)
            shard_metadata.update(
                fidelity_path=fidelity_shard_name,
                fidelity_size_bytes=fidelity_shard_path.stat().st_size,
            )
        manifest["shards"].append(shard_metadata)
        manifest["num_shards"] += 1
        total_rows += n_rows
        manifest["total_rows"] = total_rows
        _write_manifest_atomic(manifest_path, manifest)
        print(f"Encoded {src_shard_name} → {out_shard_name}  rows={n_rows}")

    manifest["total_rows"] = total_rows
    _write_manifest_atomic(manifest_path, manifest)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Total rows: {total_rows}")
    if args.fidelity_output is not None:
        fidelity_path = Path(args.fidelity_output).expanduser().resolve()
        result = _aggregate_fidelity_shards(
            output_dir=output_dir,
            manifest=manifest,
            fidelity_output=fidelity_path,
            trainer_cfg=trainer_cfg,
            checkpoint_path=checkpoint_path,
            sae=sae,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"Wrote fidelity: {fidelity_path}")
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
