"""Residual-preserving single-feature SAE intervention hook (paper Section 4.5).

For a target feature index ``i`` and scalar ``alpha``:

    z' = α · z  for i ∈ S, z'_j = z_j otherwise
    x' = x + Dec(z') − Dec(z)

The residual SAE-reconstruction error ``err(x) = x − Dec(Enc(x))`` is
preserved by construction: ``x' = Dec(z') + err(x)``. With α = 0 the
target feature is zeroed; α ∈ (0, 1) softly suppresses; α = 1 recovers
``x`` exactly; α > 1 amplifies.

The hook reads per-step metadata from ``model._sae_hook_context`` (set
by the runner) and only activates once ``step_in_episode ≥
hook_start_step`` so a warm-up phase can run unhooked.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import torch

from event_sae.openvla.activations import load_batch_topk_sae


class SAEHookError(RuntimeError):
    """Fatal hook error that must abort evaluation instead of counting a failure."""


def _load_and_validate_sae(*, model, layer_idx: int, sae_checkpoint_path: str):
    """Load an SAE and prove that it belongs on the requested model layer."""

    decoder_layers = model.language_model.model.layers
    if not (0 <= layer_idx < len(decoder_layers)):
        raise IndexError(f"layer_idx {layer_idx} out of range [0, {len(decoder_layers)})")

    device = str(model.language_model.lm_head.weight.device)
    sae, sae_config = load_batch_topk_sae(Path(sae_checkpoint_path), device=device)
    trainer_cfg = sae_config["trainer"]
    if trainer_cfg.get("submodule_name") != "post_mlp_residual":
        raise ValueError(
            "Expected a post_mlp_residual SAE checkpoint, got submodule_name="
            f"{trainer_cfg.get('submodule_name')!r}"
        )

    checkpoint_layer = int(trainer_cfg["layer"])
    if checkpoint_layer != layer_idx:
        raise ValueError(
            f"Requested decoder layer {layer_idx}, but SAE checkpoint is layer {checkpoint_layer}"
        )
    model_hidden_dim = int(model.language_model.lm_head.weight.shape[1])
    checkpoint_hidden_dim = int(trainer_cfg["activation_dim"])
    if checkpoint_hidden_dim != model_hidden_dim:
        raise ValueError(
            f"SAE activation_dim={checkpoint_hidden_dim}, but model hidden width={model_hidden_dim}"
        )
    return sae, trainer_cfg, decoder_layers


@dataclass
class ReconstructionHookHandle:
    """Removable hook plus globally aggregated reconstruction diagnostics."""

    _hook: object
    _records_file: TextIO
    _stats: dict[str, float | int]
    _removed: bool = False

    def summary(self) -> dict[str, float | int]:
        return {
            "num_forwards": int(self._stats["num_forwards"]),
            "num_environment_steps": int(self._stats["num_environment_steps"]),
            "num_tokens": int(self._stats["num_tokens"]),
            "num_values": int(self._stats["num_values"]),
        }

    def remove(self) -> None:
        if self._removed:
            return
        self._hook.remove()
        self._records_file.close()
        self._removed = True


@dataclass
class InterventionHookHandle:
    """Removable perturbation hook plus one-shot aggregate diagnostics."""

    _hook: object
    _records_file: TextIO
    _stats: dict
    feature_idx: int
    alpha: float
    _removed: bool = False

    def summary(self) -> dict:
        active = self._stats.get("active_values")
        maximum = self._stats.get("max_activation")
        return {
            "feature_id": self.feature_idx,
            "alpha": self.alpha,
            "num_forwards": int(self._stats["num_forwards"]),
            "num_environment_steps": int(self._stats["num_environment_steps"]),
            "num_tokens": int(self._stats["num_tokens"]),
            "active_feature_values": int(active.item()) if active is not None else 0,
            "max_feature_activation": float(maximum.item()) if maximum is not None else 0.0,
        }

    def remove(self) -> None:
        if self._removed:
            return
        self._hook.remove()
        self._records_file.close()
        self._removed = True


def apply_resid_post_reconstruction_hook(
    *,
    model,
    layer_idx: int,
    sae_checkpoint_path: str,
    run_dir: str,
    log_file,
) -> ReconstructionHookHandle:
    """Replace a decoder residual stream with ``Dec(Enc(x))``.

    This is the reconstruction-only hook required for the paper's Hooked SR
    fidelity check.  It deliberately does *not* preserve ``x - recon(x)``;
    preserving that residual would make the hook an identity operation.
    """

    sae, trainer_cfg, decoder_layers = _load_and_validate_sae(
        model=model,
        layer_idx=layer_idx,
        sae_checkpoint_path=sae_checkpoint_path,
    )
    activation_dim = int(trainer_cfg["activation_dim"])
    records_path = Path(run_dir) / "sae_reconstruction_records.jsonl"
    records_file = records_path.open("w", encoding="utf-8")
    stats: dict[str, float | int] = {
        "num_forwards": 0,
        "num_environment_steps": 0,
        "num_tokens": 0,
        "num_values": 0,
    }
    last_environment_step: dict[str, tuple[int | None, int | None] | None] = {
        "value": None
    }

    @torch.inference_mode()
    def hook_fn(module, inputs, output):
        del module, inputs
        try:
            if not isinstance(output, tuple):
                raise ValueError(f"Expected decoder layer output tuple, got {type(output)!r}")
            hidden = output[0]
            if hidden.ndim != 3 or hidden.shape[-1] != activation_dim:
                raise ValueError(
                    f"Hidden shape {tuple(hidden.shape)} incompatible with "
                    f"SAE activation_dim={activation_dim}"
                )

            flat = hidden.reshape(-1, hidden.shape[-1]).to(dtype=torch.float32)
            reconstructed = sae.decode(sae.encode(flat))
            stats["num_forwards"] = int(stats["num_forwards"]) + 1
            stats["num_tokens"] = int(stats["num_tokens"]) + int(flat.shape[0])
            stats["num_values"] = int(stats["num_values"]) + int(flat.numel())

            # One lightweight evidence record per environment step. Avoid
            # GPU-synchronizing .item() calls and per-token disk flushes here;
            # numerical reconstruction quality is measured by evaluate_sae.py.
            context = getattr(model, "_sae_hook_context", {}) or {}
            environment_step = (
                context.get("episode_num"),
                context.get("step_in_episode"),
            )
            if environment_step != last_environment_step["value"]:
                last_environment_step["value"] = environment_step
                stats["num_environment_steps"] = int(stats["num_environment_steps"]) + 1
                records_file.write(
                    json.dumps(
                        {
                            "episode_num": context.get("episode_num"),
                            "task_id": context.get("task_id"),
                            "task_episode_idx": context.get("task_episode_idx"),
                            "step_in_episode": context.get("step_in_episode"),
                            "first_forward_idx": int(stats["num_forwards"]),
                        }
                    )
                    + "\n"
                )
            updated = reconstructed.to(dtype=hidden.dtype).reshape_as(hidden)
            return (updated, *output[1:])
        except SAEHookError:
            raise
        except Exception as exc:
            raise SAEHookError(f"SAE reconstruction hook failed: {exc}") from exc

    hook = decoder_layers[layer_idx].register_forward_hook(hook_fn)
    log_file.write(
        f"SAE reconstruction hook: layer={layer_idx} "
        f"sae_checkpoint={sae_checkpoint_path}\n"
    )
    log_file.flush()
    return ReconstructionHookHandle(hook, records_file, stats)


def apply_resid_post_feature_perturb_hook(
    *,
    model,
    layer_idx: int,
    sae_checkpoint_path: str,
    feature_idx: int,
    alpha: float,
    hook_start_step: int,
    run_dir: str,
    log_file,
) -> InterventionHookHandle:
    sae, trainer_cfg, decoder_layers = _load_and_validate_sae(
        model=model,
        layer_idx=layer_idx,
        sae_checkpoint_path=sae_checkpoint_path,
    )
    activation_dim = int(trainer_cfg["activation_dim"])
    dict_size = int(trainer_cfg["dict_size"])
    if not (0 <= feature_idx < dict_size):
        raise IndexError(f"feature_idx {feature_idx} out of range [0, {dict_size})")

    records_path = Path(run_dir) / f"intervene_feat{feature_idx}_alpha{alpha}_records.jsonl"
    records_file = records_path.open("w", encoding="utf-8")
    stats = {
        "num_forwards": 0,
        "num_environment_steps": 0,
        "num_tokens": 0,
        "active_values": None,
        "max_activation": None,
    }
    last_environment_step = {"value": None}

    @torch.inference_mode()
    def hook_fn(module, inputs, output):
        del module, inputs
        try:
            context = getattr(model, "_sae_hook_context", {}) or {}
            episode_num = context.get("episode_num")
            step_in_episode = context.get("step_in_episode")
            if episode_num is None or step_in_episode is None:
                return output
            if step_in_episode < hook_start_step:
                return output

            if not isinstance(output, tuple):
                raise ValueError(f"Expected decoder layer output tuple, got {type(output)!r}")
            hidden = output[0]
            if hidden.ndim != 3 or hidden.shape[-1] != activation_dim:
                raise ValueError(
                    f"Hidden shape {tuple(hidden.shape)} incompatible with "
                    f"SAE activation_dim={activation_dim}"
                )

            flat = hidden.reshape(-1, hidden.shape[-1]).to(dtype=torch.float32)
            encoded = sae.encode(flat)
            perturbed = encoded.clone()
            feature_before = encoded[:, feature_idx]
            perturbed[:, feature_idx] = feature_before * float(alpha)
            updated = flat + (sae.decode(perturbed) - sae.decode(encoded))

            stats["num_forwards"] += 1
            stats["num_tokens"] += int(encoded.shape[0])
            active = torch.count_nonzero(feature_before)
            maximum = feature_before.max()
            stats["active_values"] = (
                active if stats["active_values"] is None else stats["active_values"] + active
            )
            stats["max_activation"] = (
                maximum
                if stats["max_activation"] is None
                else torch.maximum(stats["max_activation"], maximum)
            )

            environment_step = (episode_num, step_in_episode)
            if environment_step != last_environment_step["value"]:
                last_environment_step["value"] = environment_step
                stats["num_environment_steps"] += 1
                records_file.write(
                    json.dumps(
                        {
                            "episode_num": int(episode_num),
                            "task_id": context.get("task_id"),
                            "task_episode_idx": context.get("task_episode_idx"),
                            "step_in_episode": int(step_in_episode),
                            "first_forward_idx": int(stats["num_forwards"]),
                            "feature_id": int(feature_idx),
                            "alpha": float(alpha),
                        }
                    )
                    + "\n"
                )
            return (updated.to(dtype=hidden.dtype).reshape_as(hidden), *output[1:])
        except SAEHookError:
            raise
        except Exception as exc:
            raise SAEHookError(f"SAE feature intervention hook failed: {exc}") from exc

    hook = decoder_layers[layer_idx].register_forward_hook(hook_fn)
    log_file.write(
        f"Resid-post intervention hook: layer={layer_idx} feature_idx={feature_idx} "
        f"alpha={alpha} hook_start_step={hook_start_step} sae_checkpoint={sae_checkpoint_path}\n"
    )
    log_file.flush()
    return InterventionHookHandle(
        hook,
        records_file,
        stats,
        feature_idx=int(feature_idx),
        alpha=float(alpha),
    )
