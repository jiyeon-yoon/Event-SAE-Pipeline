"""openVLA model loading and per-step action inference."""

import json
from typing import Any

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

from event_sae.openvla.eval.utils import DEVICE, OPENVLA_V01_SYSTEM_PROMPT


ACTION_DIM = 7


def load_model(cfg: Any):
    revision_kwargs = {}
    if cfg.model.revision:
        revision_kwargs["revision"] = cfg.model.revision
    if cfg.model.code_revision:
        revision_kwargs["code_revision"] = cfg.model.code_revision
    # Try flash_attention_2 first; fall back to sdpa if unavailable or incompatible.
    try:
        import flash_attn  # noqa: F401
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                cfg.model.checkpoint,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                load_in_8bit=cfg.model.load_in_8bit,
                load_in_4bit=cfg.model.load_in_4bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **revision_kwargs,
            )
            print("Successfully loaded model with flash_attention_2")
        except (ValueError, RuntimeError, ImportError) as e:
            print(f"Warning: flash_attention_2 failed ({e}), falling back to sdpa")
            model = AutoModelForVision2Seq.from_pretrained(
                cfg.model.checkpoint,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                load_in_8bit=cfg.model.load_in_8bit,
                load_in_4bit=cfg.model.load_in_4bit,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                **revision_kwargs,
            )
    except ImportError:
        print("Warning: flash-attn not available, using sdpa attention")
        model = AutoModelForVision2Seq.from_pretrained(
            cfg.model.checkpoint,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            load_in_8bit=cfg.model.load_in_8bit,
            load_in_4bit=cfg.model.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **revision_kwargs,
        )
    if not cfg.model.load_in_8bit and not cfg.model.load_in_4bit:
        model = model.to(DEVICE)

    dataset_statistics_path = f"{cfg.model.checkpoint}/dataset_statistics.json"
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if tf.io.gfile.exists(dataset_statistics_path):
        with tf.io.gfile.GFile(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        model.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA checkpoint. "
            "Otherwise, you may run into errors when calling predict_action due to a missing unnorm key."
        )

    print(f"Loaded model: {type(model)}")
    return model


def get_processor(cfg: Any):
    kwargs = {}
    if cfg.model.revision:
        kwargs["revision"] = cfg.model.revision
    if cfg.model.code_revision:
        kwargs["code_revision"] = cfg.model.code_revision
    return AutoProcessor.from_pretrained(
        cfg.model.checkpoint, trust_remote_code=True, **kwargs
    )


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [height_offsets, width_offsets, height_offsets + new_heights, width_offsets + new_widths], axis=1
    )

    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    if expanded_dims:
        image = image[0]

    return image


def _build_prompt(task_label: str, checkpoint: str) -> str:
    if "openvla-v01" in checkpoint:
        return f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
    return f"In: What action should the robot take to {task_label.lower()}?\nOut:"


def get_action(model, processor, cfg: Any, obs: dict, task_label: str, unnorm_key: str) -> np.ndarray:
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if cfg.model.center_crop:
        batch_size = 1
        crop_scale = 0.9
        image_tf = tf.convert_to_tensor(np.array(image))
        orig_dtype = image_tf.dtype
        image_tf = tf.image.convert_image_dtype(image_tf, tf.float32)
        image_tf = crop_and_resize(image_tf, crop_scale, batch_size)
        image_tf = tf.clip_by_value(image_tf, 0, 1)
        image_tf = tf.image.convert_image_dtype(image_tf, orig_dtype, saturate=True)
        image = Image.fromarray(image_tf.numpy()).convert("RGB")

    prompt = _build_prompt(task_label, cfg.model.checkpoint)
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)
    action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    assert action.shape == (ACTION_DIM,)
    return action
