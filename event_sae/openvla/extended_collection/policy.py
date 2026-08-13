"""Single-pass OpenVLA action inference with compact uncertainty summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


@dataclass(frozen=True)
class PolicyOutput:
    raw_action: np.ndarray
    action_token_ids: list[int]
    uncertainty: dict[str, Any]
    model_input_rgb: np.ndarray
    preprocessing: dict[str, Any]


def _build_prompt(task_label: str, checkpoint: str) -> str:
    if "openvla-v01" in checkpoint:
        return (
            f"{_OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take "
            f"to {task_label.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def _prepare_model_input_rgb(
    source_rgb: np.ndarray,
    *,
    center_crop: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the exact center-crop path used by the existing OpenVLA runner."""

    source = np.asarray(source_rgb, dtype=np.uint8)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {source.shape}")
    metadata: dict[str, Any] = {
        "source": "LIBERO agentview_image flipped 180 degrees then Lanczos3 resized",
        "source_shape": list(source.shape),
        "source_dtype": str(source.dtype),
        "center_crop": bool(center_crop),
        "crop_scale": 0.9 if center_crop else 1.0,
        "output_shape": [224, 224, 3],
        "output_dtype": "uint8",
    }
    if not center_crop:
        if source.shape != (224, 224, 3):
            raise ValueError(
                "source_rgb must already be the 224x224 LIBERO/OpenVLA source image"
            )
        return source.copy(), metadata

    import tensorflow as tf

    image = tf.convert_to_tensor(source)
    original_dtype = image.dtype
    image = tf.image.convert_image_dtype(image, tf.float32)
    crop_side = tf.sqrt(tf.constant(0.9, dtype=tf.float32))
    offset = (1.0 - crop_side) / 2.0
    boxes = tf.reshape(
        tf.stack([offset, offset, offset + crop_side, offset + crop_side]),
        (1, 4),
    )
    image = tf.image.crop_and_resize(
        tf.expand_dims(image, axis=0), boxes, tf.range(1), (224, 224)
    )[0]
    image = tf.clip_by_value(image, 0.0, 1.0)
    image = tf.image.convert_image_dtype(image, original_dtype, saturate=True)
    result = np.asarray(image.numpy(), dtype=np.uint8)
    if result.shape != (224, 224, 3):
        raise RuntimeError(f"Unexpected processed RGB shape: {result.shape}")
    return result, metadata


def _model_device(model):
    import torch

    try:
        return next(model.parameters()).device
    except (StopIteration, AttributeError, TypeError):
        try:
            return model.language_model.lm_head.weight.device
        except AttributeError:
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _summarize_action_scores(
    scores,
    token_ids: list[int],
    *,
    action_vocab_end: int,
    action_vocab_size: int,
) -> dict[str, Any]:
    """Summarize full-vocabulary and action-token uncertainty without logits."""

    import torch

    start = action_vocab_end - action_vocab_size
    per_dimension: list[dict[str, Any]] = []
    for dimension, (score, selected_id) in enumerate(zip(scores, token_ids)):
        if score.ndim != 2 or score.shape[0] != 1:
            raise ValueError(f"Expected one score row, got {tuple(score.shape)}")
        if score.shape[-1] < action_vocab_end:
            raise ValueError(
                f"Score vocabulary {score.shape[-1]} is smaller than action-vocab end {action_vocab_end}"
            )
        full_probabilities = torch.softmax(score[0].to(torch.float32), dim=-1)
        full_top_values, full_top_indices = torch.topk(full_probabilities, k=2)
        full_entropy = -(
            full_probabilities * torch.log(full_probabilities.clamp_min(1e-12))
        ).sum()
        action_probability_mass = full_probabilities[start:action_vocab_end].sum()
        conditional = full_probabilities[
            start:action_vocab_end
        ] / action_probability_mass.clamp_min(1e-12)
        action_top_values, action_top_indices = torch.topk(conditional, k=2)
        action_entropy = -(conditional * torch.log(conditional.clamp_min(1e-12))).sum()
        in_action_vocab = start <= selected_id < action_vocab_end
        if in_action_vocab:
            selected_index = selected_id - start
            selected_probability = float(full_probabilities[selected_id].item())
            selected_conditional_probability = float(conditional[selected_index].item())
            selected_rank = (
                int((conditional > conditional[selected_index]).sum().item()) + 1
            )
        else:
            selected_probability = None
            selected_conditional_probability = None
            selected_rank = None
        per_dimension.append(
            {
                "action_dimension": dimension,
                "selected_token_id": int(selected_id),
                "selected_token_is_action_token": bool(in_action_vocab),
                "selected_token_probability": selected_probability,
                "selected_token_conditional_probability": selected_conditional_probability,
                "selected_token_conditional_rank": selected_rank,
                "full_next_token": {
                    "vocabulary_size": int(score.shape[-1]),
                    "entropy_nats": float(full_entropy.item()),
                    "normalized_entropy": float(full_entropy.item())
                    / float(np.log(score.shape[-1])),
                    "top1_probability": float(full_top_values[0].item()),
                    "top1_token_id": int(full_top_indices[0].item()),
                    "top1_top2_margin": float(
                        (full_top_values[0] - full_top_values[1]).item()
                    ),
                },
                "conditional_action_token": {
                    "probability_mass_in_full_vocabulary": float(
                        action_probability_mass.item()
                    ),
                    "entropy_nats": float(action_entropy.item()),
                    "normalized_entropy": float(action_entropy.item())
                    / float(np.log(action_vocab_size)),
                    "top1_probability": float(action_top_values[0].item()),
                    "top1_token_id": int(start + action_top_indices[0].item()),
                    "top1_top2_margin": float(
                        (action_top_values[0] - action_top_values[1]).item()
                    ),
                },
            }
        )

    def nested_mean(group: str, key: str) -> float:
        values = [float(row[group][key]) for row in per_dimension]
        return float(np.mean(values))

    return {
        "distribution": (
            "full next-token summaries plus summaries conditional on the "
            "OpenVLA action-token subset"
        ),
        "action_vocab_start": start,
        "action_vocab_end_exclusive": action_vocab_end,
        "action_vocab_size": action_vocab_size,
        "per_action_dimension": per_dimension,
        "summary": {
            "mean_full_entropy_nats": nested_mean("full_next_token", "entropy_nats"),
            "mean_full_top1_probability": nested_mean(
                "full_next_token", "top1_probability"
            ),
            "mean_full_top1_top2_margin": nested_mean(
                "full_next_token", "top1_top2_margin"
            ),
            "mean_action_token_probability_mass": nested_mean(
                "conditional_action_token", "probability_mass_in_full_vocabulary"
            ),
            "mean_conditional_action_entropy_nats": nested_mean(
                "conditional_action_token", "entropy_nats"
            ),
            "mean_conditional_action_top1_probability": nested_mean(
                "conditional_action_token", "top1_probability"
            ),
            "mean_conditional_action_top1_top2_margin": nested_mean(
                "conditional_action_token", "top1_top2_margin"
            ),
        },
    }


def infer_action_with_uncertainty(
    model,
    processor,
    cfg: Any,
    source_rgb: np.ndarray,
    task_label: str,
    unnorm_key: str,
) -> PolicyOutput:
    """Generate once, decode exactly like OpenVLA, and retain only summaries."""

    import torch
    from PIL import Image

    model_rgb, preprocessing = _prepare_model_input_rgb(
        source_rgb, center_crop=bool(cfg.model.center_crop)
    )
    prompt = _build_prompt(task_label, cfg.model.checkpoint)
    device = _model_device(model)
    inputs = processor(prompt, Image.fromarray(model_rgb).convert("RGB"))
    inputs = inputs.to(device, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        empty_token = torch.tensor(
            [[29871]], dtype=input_ids.dtype, device=input_ids.device
        )
        input_ids = torch.cat((input_ids, empty_token), dim=1)

    action_dim = int(model.get_action_dim(unnorm_key))
    if action_dim != 7:
        raise ValueError(f"Expected OpenVLA 7D action, got {action_dim}")
    generate_kwargs = dict(inputs)
    generate_kwargs["input_ids"] = input_ids
    with torch.inference_mode():
        generated = model.generate(
            **generate_kwargs,
            max_new_tokens=action_dim,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    if len(generated.scores) != action_dim:
        raise RuntimeError(
            f"Expected {action_dim} generation-score tensors, got {len(generated.scores)}"
        )
    token_tensor = generated.sequences[0, -action_dim:]
    token_ids = [int(value) for value in token_tensor.detach().cpu().tolist()]

    # This is the pinned OpenVLA predict_action de-tokenization, kept verbatim
    # in mathematical form so uncertainty collection does not change actions.
    discretized = int(model.vocab_size) - np.asarray(token_ids, dtype=np.int64)
    discretized = np.clip(
        discretized - 1,
        a_min=0,
        a_max=int(np.asarray(model.bin_centers).shape[0]) - 1,
    )
    normalized = np.asarray(model.bin_centers)[discretized]
    stats = model.get_action_stats(unnorm_key)
    mask = np.asarray(
        stats.get("mask", np.ones_like(stats["q01"], dtype=bool)), dtype=bool
    )
    action_high = np.asarray(stats["q99"])
    action_low = np.asarray(stats["q01"])
    action = np.where(
        mask,
        0.5 * (normalized + 1.0) * (action_high - action_low) + action_low,
        normalized,
    )
    if action.shape != (7,):
        raise RuntimeError(f"Decoded action has unexpected shape {action.shape}")

    uncertainty = _summarize_action_scores(
        generated.scores,
        token_ids,
        action_vocab_end=int(model.vocab_size),
        action_vocab_size=int(np.asarray(model.bin_centers).shape[0]),
    )
    preprocessing["prompt_template"] = (
        "openvla-v01" if "openvla-v01" in cfg.model.checkpoint else "openvla"
    )
    return PolicyOutput(
        raw_action=action,
        action_token_ids=token_ids,
        uncertainty=uncertainty,
        model_input_rgb=model_rgb,
        preprocessing=preprocessing,
    )
