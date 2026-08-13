"""Runtime primitives owned by the independent extended collector."""

from __future__ import annotations

import json
import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _revision_kwargs(cfg: Any) -> dict[str, str]:
    result: dict[str, str] = {}
    if cfg.model.revision:
        result["revision"] = cfg.model.revision
    if cfg.model.code_revision:
        result["code_revision"] = cfg.model.code_revision
    return result


def load_openvla(cfg: Any):
    """Load the pinned OpenVLA policy without Event-SAE evaluator imports."""

    import tensorflow as tf
    import torch
    from transformers import AutoModelForVision2Seq

    if not torch.cuda.is_available():
        raise RuntimeError("Extended OpenVLA collection requires an NVIDIA GPU")
    common = {
        "torch_dtype": torch.bfloat16,
        "load_in_8bit": cfg.model.load_in_8bit,
        "load_in_4bit": cfg.model.load_in_4bit,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        **_revision_kwargs(cfg),
    }
    try:
        import flash_attn  # noqa: F401

        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model.checkpoint,
            attn_implementation="flash_attention_2",
            **common,
        )
        print("OpenVLA attention: flash_attention_2")
    except (ImportError, RuntimeError, ValueError) as exc:
        print(f"OpenVLA attention: sdpa (flash-attn unavailable: {exc})")
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model.checkpoint,
            attn_implementation="sdpa",
            **common,
        )
    if not cfg.model.load_in_8bit and not cfg.model.load_in_4bit:
        model = model.to(torch.device("cuda:0"))

    statistics_path = f"{cfg.model.checkpoint}/dataset_statistics.json"
    if tf.io.gfile.exists(statistics_path):
        with tf.io.gfile.GFile(statistics_path, "r") as stream:
            model.norm_stats = json.load(stream)
    if not hasattr(model, "norm_stats"):
        raise RuntimeError(
            "OpenVLA action normalization statistics are unavailable for the pinned checkpoint"
        )
    return model


def load_processor(cfg: Any):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        cfg.model.checkpoint,
        trust_remote_code=True,
        **_revision_kwargs(cfg),
    )


def get_source_rgb(obs: dict[str, Any], size: int = 224) -> np.ndarray:
    """Reproduce the OpenVLA-LIBERO source-image conversion exactly."""

    import tensorflow as tf

    image = np.asarray(obs["agentview_image"])[::-1, ::-1]
    image = tf.image.encode_jpeg(image)
    image = tf.io.decode_image(image, expand_animations=False, dtype=tf.uint8)
    image = tf.image.resize(image, (size, size), method="lanczos3", antialias=True)
    image = tf.cast(tf.clip_by_value(tf.round(image), 0, 255), tf.uint8)
    return np.asarray(image.numpy(), dtype=np.uint8)


def dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def to_executed_libero_action(raw_action: np.ndarray) -> np.ndarray:
    """Apply the OpenVLA gripper convention used by LIBERO evaluation."""

    action = np.asarray(raw_action).copy()
    action[..., -1] = 2.0 * action[..., -1] - 1.0
    action[..., -1] = np.sign(action[..., -1])
    action[..., -1] *= -1.0
    return action
