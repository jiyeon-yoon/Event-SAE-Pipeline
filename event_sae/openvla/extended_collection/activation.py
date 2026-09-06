"""Independent dense Layer-31 activation shard collector."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class Layer31ActivationCollector:
    """Capture every decoder forward while retaining policy-step metadata."""

    layer_idx = 31

    def __init__(self, model, output_dir: str | Path, flush_every: int):
        if flush_every <= 0:
            raise ValueError("flush_every must be positive")
        layers = model.language_model.model.layers
        if self.layer_idx >= len(layers):
            raise IndexError(
                f"OpenVLA has {len(layers)} decoder layers; layer 31 is unavailable"
            )
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._index = (self.output_dir / "activation_index.jsonl").open(
            "w", encoding="utf-8"
        )
        self._flush_every = int(flush_every)
        self._buffer: list[torch.Tensor] = []
        self._records: list[dict] = []
        self._rows = 0
        self._shard_id = 0
        self._forward_id = 0
        self._closed = False
        self._transaction: tuple[int, int, int, int] | None = None
        self._hook = layers[self.layer_idx].register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[-1] != 4096:
            raise ValueError(
                f"Unexpected Layer-31 output {tuple(hidden.shape)}; expected (batch, tokens, 4096)"
            )
        sample = hidden.reshape(-1, 4096).detach().to(torch.float32).cpu()
        row_count = int(sample.shape[0])
        context = getattr(self.model, "_sae_hook_context", {}) or {}
        required = ("episode_num", "task_id", "task_episode_idx", "step_in_episode")
        missing = [name for name in required if context.get(name) is None]
        if missing:
            raise RuntimeError(
                "Layer-31 forward occurred without collection context: "
                + ", ".join(missing)
            )
        self._forward_id += 1
        self._records.append(
            {
                "layer_idx": self.layer_idx,
                "row_start": self._rows,
                "row_end": self._rows + row_count,
                "episode_num": int(context["episode_num"]),
                "task_id": int(context["task_id"]),
                "task_episode_idx": int(context["task_episode_idx"]),
                "task_description": context.get("task_description"),
                "step_in_episode": int(context["step_in_episode"]),
                "pair_id": context.get("pair_id"),
                "condition": context.get("condition"),
                "pair_seed": context.get("pair_seed"),
                "global_forward_idx": self._forward_id,
                "tokens_in_forward": row_count,
            }
        )
        self._buffer.append(sample)
        self._rows += row_count
        if self._transaction is None and self._rows >= self._flush_every:
            self._flush()

    def begin_step(self, context: dict) -> None:
        """Stage hook rows until the matching policy step is durably written."""

        if self._transaction is not None:
            raise RuntimeError("An activation step transaction is already active")
        self.model._sae_hook_context = dict(context)
        self._transaction = (
            len(self._buffer),
            len(self._records),
            self._rows,
            self._forward_id,
        )

    def commit_step(self) -> None:
        if self._transaction is None:
            raise RuntimeError("No activation step transaction is active")
        self._transaction = None
        if self._rows >= self._flush_every:
            self._flush()

    def abort_step(self) -> None:
        if self._transaction is None:
            return
        buffer_len, record_len, rows, forward_id = self._transaction
        del self._buffer[buffer_len:]
        del self._records[record_len:]
        self._rows = rows
        self._forward_id = forward_id
        self._transaction = None

    def flush_episode(self) -> None:
        """Persist committed rows at an episode boundary."""

        if self._transaction is not None:
            raise RuntimeError("Cannot flush during an activation transaction")
        self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        shard_name = f"layer_31_shard_{self._shard_id:06d}.pt"
        shard_path = self.output_dir / shard_name
        temporary_path = shard_path.with_suffix(shard_path.suffix + ".tmp")
        torch.save(torch.cat(self._buffer, dim=0), temporary_path)
        temporary_path.replace(shard_path)
        for record in self._records:
            record["shard_path"] = shard_name
            self._index.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._index.flush()
        self._buffer.clear()
        self._records.clear()
        self._rows = 0
        self._shard_id += 1

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.abort_step()
            self._flush()
        finally:
            self._hook.remove()
            self._index.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
