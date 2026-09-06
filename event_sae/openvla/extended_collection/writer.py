"""Durable writers for aligned episode, simulator, vision, and JSON data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np


def jsonable(value: Any) -> Any:
    """Convert numpy/scalar/container values without silently stringifying them."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    raise TypeError(f"Cannot serialize value of type {type(value)!r}")


def array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _append_jsonl(stream, value: Dict[str, Any], flush: bool) -> None:
    stream.write(
        json.dumps(jsonable(value), ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    if flush:
        stream.flush()


@dataclass
class EpisodeBuffer:
    episode_num: int
    sim_pre: Dict[str, list[np.ndarray]] = field(default_factory=dict)
    sim_post: Dict[str, list[np.ndarray]] = field(default_factory=dict)
    step_ids: list[int] = field(default_factory=list)
    model_input_rgb: list[np.ndarray] = field(default_factory=list)
    observed_source_rgb: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        step_in_episode: int,
        pre_vectors: Dict[str, np.ndarray],
        post_vectors: Dict[str, np.ndarray],
        model_input_rgb: np.ndarray | None,
        observed_source_rgb: np.ndarray | None,
    ) -> tuple[int, int | None, int | None]:
        sim_row = len(self.step_ids)
        self.step_ids.append(int(step_in_episode))
        for key, value in pre_vectors.items():
            self.sim_pre.setdefault(key, []).append(np.asarray(value).copy())
        for key, value in post_vectors.items():
            self.sim_post.setdefault(key, []).append(np.asarray(value).copy())
        vision_row = None
        if model_input_rgb is not None:
            vision_row = len(self.model_input_rgb)
            self.model_input_rgb.append(
                np.asarray(model_input_rgb, dtype=np.uint8).copy()
            )
        observed_vision_row = None
        if observed_source_rgb is not None:
            observed_vision_row = len(self.observed_source_rgb)
            self.observed_source_rgb.append(
                np.asarray(observed_source_rgb, dtype=np.uint8).copy()
            )
        return sim_row, vision_row, observed_vision_row


class ExtendedRunWriter:
    """Own all output streams so the runner can close them in one finally block."""

    def __init__(
        self,
        run_dir: str | Path,
        *,
        flush_every_step: bool = True,
        enable_pair_results: bool = False,
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.flush_every_step = bool(flush_every_step)
        for directory in (
            "initial_states",
            "sim_state",
            "vision",
            "videos",
            "schemas",
        ):
            (self.run_dir / directory).mkdir(parents=True, exist_ok=True)
        self.prompt_stream = (self.run_dir / "prompt_records.jsonl").open(
            "w", encoding="utf-8"
        )
        self.trajectory_stream = (self.run_dir / "trajectory_records.jsonl").open(
            "w", encoding="utf-8"
        )
        self.episode_stream = (self.run_dir / "episode_results.jsonl").open(
            "w", encoding="utf-8"
        )
        self.action_stream = (self.run_dir / "action_records.jsonl").open(
            "w", encoding="utf-8"
        )
        self.uncertainty_stream = (self.run_dir / "policy_uncertainty.jsonl").open(
            "w", encoding="utf-8"
        )
        self.pair_stream = (
            (self.run_dir / "pair_results.jsonl").open("w", encoding="utf-8")
            if enable_pair_results
            else None
        )
        self._buffer: EpisodeBuffer | None = None

    def write_manifest(self, value: Dict[str, Any]) -> None:
        (self.run_dir / "manifest.json").write_text(
            json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_schema(self, task_id: int, value: Dict[str, Any]) -> str:
        path = self.run_dir / "schemas" / f"task_{task_id:02d}.json"
        path.write_text(
            json.dumps(jsonable(value), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return str(path.relative_to(self.run_dir))

    def begin_episode(
        self,
        *,
        prompt_record: Dict[str, Any],
        initial_state: Any,
    ) -> Dict[str, Any]:
        if self._buffer is not None:
            raise RuntimeError("Previous episode was not finished")
        episode_num = int(prompt_record["episode_num"])
        initial_state_array = np.asarray(initial_state)
        initial_path = (
            self.run_dir / "initial_states" / f"episode_{episode_num:06d}.npz"
        )
        np.savez_compressed(initial_path, initial_state=initial_state_array)
        initial_hash = array_sha256(initial_state_array)
        record = dict(prompt_record)
        record.update(
            {
                "initial_state_sha256": initial_hash,
                "initial_state_path": str(initial_path.relative_to(self.run_dir)),
                "initial_state_shape": list(initial_state_array.shape),
                "initial_state_dtype": str(initial_state_array.dtype),
            }
        )
        _append_jsonl(self.prompt_stream, record, True)
        self._buffer = EpisodeBuffer(episode_num=episode_num)
        return record

    def write_step(
        self,
        *,
        common: Dict[str, Any],
        pre_json: Dict[str, Any],
        post_json: Dict[str, Any],
        pre_vectors: Dict[str, np.ndarray],
        post_vectors: Dict[str, np.ndarray],
        raw_action: np.ndarray,
        executed_action: np.ndarray,
        policy: Dict[str, Any],
        policy_action: np.ndarray | None = None,
        intervention: Dict[str, Any] | None = None,
        model_input_rgb: np.ndarray | None,
        reward: float,
        done: bool,
        info: Dict[str, Any],
        observed_source_rgb: np.ndarray | None = None,
    ) -> None:
        if self._buffer is None:
            raise RuntimeError("begin_episode must be called before write_step")
        step_id = int(common["step_in_episode"])
        sim_row, vision_row, observed_vision_row = self._buffer.add(
            step_id,
            pre_vectors,
            post_vectors,
            model_input_rgb,
            observed_source_rgb,
        )
        policy_action = (
            np.asarray(executed_action)
            if policy_action is None
            else np.asarray(policy_action)
        )
        intervention = dict(intervention or {"forced_open_applied": False})
        record = {
            **common,
            "alignment": "pre_state + action -> post_state",
            "sim_state_row": sim_row,
            "model_input_rgb_row": vision_row,
            "observed_source_rgb_row": observed_vision_row,
            "pre": pre_json,
            "post": post_json,
            "raw_openvla_action": np.asarray(raw_action),
            "policy_libero_action": policy_action,
            "executed_libero_action": np.asarray(executed_action),
            "intervention": intervention,
            "reward": float(reward),
            "done": bool(done),
            "info": info,
        }
        _append_jsonl(self.trajectory_stream, record, self.flush_every_step)
        _append_jsonl(
            self.action_stream,
            {
                **common,
                "raw_openvla_action": np.asarray(raw_action),
                "policy_libero_action": policy_action,
                "executed_libero_action": np.asarray(executed_action),
                "intervention": intervention,
            },
            self.flush_every_step,
        )
        _append_jsonl(
            self.uncertainty_stream,
            {**common, **policy},
            self.flush_every_step,
        )

    def write_pair_result(self, value: Dict[str, Any]) -> None:
        if self.pair_stream is None:
            raise RuntimeError("Pair-result output was not enabled")
        _append_jsonl(self.pair_stream, value, True)

    def finish_episode(
        self,
        *,
        result: Dict[str, Any],
        compress_npz: bool,
    ) -> Dict[str, Any]:
        if self._buffer is None:
            raise RuntimeError("No active episode")
        buffer = self._buffer
        save = np.savez_compressed if compress_npz else np.savez
        sim_path = self.run_dir / "sim_state" / f"episode_{buffer.episode_num:06d}.npz"
        sim_payload: Dict[str, Any] = {
            "step_in_episode": np.asarray(buffer.step_ids, dtype=np.int32)
        }
        for phase, values in (("pre", buffer.sim_pre), ("post", buffer.sim_post)):
            for key, rows in values.items():
                sim_payload[f"{phase}_{key}"] = (
                    np.stack(rows) if rows else np.empty((0,))
                )
        save(sim_path, **sim_payload)

        vision_path: Path | None = None
        if buffer.model_input_rgb or buffer.observed_source_rgb:
            vision_path = (
                self.run_dir / "vision" / f"episode_{buffer.episode_num:06d}.npz"
            )
            vision_payload: Dict[str, Any] = {
                "step_in_episode": np.asarray(buffer.step_ids, dtype=np.int32)
            }
            if buffer.model_input_rgb:
                if len(buffer.model_input_rgb) != len(buffer.step_ids):
                    raise RuntimeError("model_input_rgb is not step-aligned")
                vision_payload["model_input_rgb"] = np.stack(
                    buffer.model_input_rgb
                ).astype(np.uint8)
            if buffer.observed_source_rgb:
                if len(buffer.observed_source_rgb) != len(buffer.step_ids):
                    raise RuntimeError("observed_source_rgb is not step-aligned")
                vision_payload["observed_source_rgb"] = np.stack(
                    buffer.observed_source_rgb
                ).astype(np.uint8)
            save(vision_path, **vision_payload)
        final = dict(result)
        final.update(
            {
                "sim_state_path": str(sim_path.relative_to(self.run_dir)),
                "vision_path": (
                    None
                    if vision_path is None
                    else str(vision_path.relative_to(self.run_dir))
                ),
                "recorded_steps": len(buffer.step_ids),
            }
        )
        _append_jsonl(self.episode_stream, final, True)
        self._buffer = None
        return final

    def close(self) -> None:
        streams = [
            self.prompt_stream,
            self.trajectory_stream,
            self.episode_stream,
            self.action_stream,
            self.uncertainty_stream,
        ]
        if self.pair_stream is not None:
            streams.append(self.pair_stream)
        for stream in streams:
            if not stream.closed:
                stream.flush()
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
