"""Evaluate raw OpenVLA or the reconstruction-hooked SAE policy on LIBERO."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from event_sae.openvla.eval.config import load_config, parse_overrides, resolve_task_ids
from event_sae.openvla.eval.runner import eval_libero
from event_sae.openvla.intervene import (
    ReconstructionHookHandle,
    apply_resid_post_reconstruction_hook,
)


def _count_jsonl(path: str | None) -> int:
    if path is None:
        return 0
    with Path(path).open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _git_provenance() -> dict:
    root = Path(__file__).resolve().parents[2]
    return {
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(
                ["git", "-C", str(root), "status", "--porcelain"], text=True
            ).strip()
        ),
        "remote": subprocess.check_output(
            ["git", "-C", str(root), "remote", "get-url", "origin"], text=True
        ).strip(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_config(cfg) -> dict:
    """Stable protocol fields; output paths are deliberately excluded."""

    payload = asdict(cfg)
    payload["logging"].pop("root_dir", None)
    payload["env"]["resolved_task_ids"] = resolve_task_ids(cfg.env.task_ids, 10)
    return payload


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run raw-policy or reconstruction-hooked OpenVLA LIBERO evaluation."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--mode", required=True, choices=("raw", "reconstruction"))
    parser.add_argument("--sae-checkpoint", default="")
    parser.add_argument("--layer-idx", type=int, default=31)
    parser.add_argument(
        "--result-output",
        default=None,
        help="Optional stable JSON path in addition to the copy inside the timestamped run dir.",
    )
    parser.add_argument(
        "--expected-rollouts",
        type=int,
        required=True,
        help="Fail unless exactly this many rollouts complete.",
    )
    parser.add_argument(
        "--expected-code-revision",
        required=True,
        help="Exact clean Git commit that is allowed to start this costly evaluation.",
    )
    args = parser.parse_args()

    if args.mode == "reconstruction" and not args.sae_checkpoint:
        parser.error("--mode reconstruction requires --sae-checkpoint")

    code = _git_provenance()
    if code["dirty"]:
        raise RuntimeError("Refusing expensive rollouts from a dirty Git worktree")
    if code["commit"] != args.expected_code_revision:
        raise RuntimeError(
            f"Code revision {code['commit']} does not match --expected-code-revision "
            f"{args.expected_code_revision}"
        )

    cfg = load_config(args.config, overrides=parse_overrides(args.override))
    cfg.sae_collect.enabled = False
    reconstruction_handles: list[ReconstructionHookHandle] = []

    def applier(*, model, cfg, run_dir, log_file):
        del cfg
        handle = apply_resid_post_reconstruction_hook(
            model=model,
            layer_idx=args.layer_idx,
            sae_checkpoint_path=args.sae_checkpoint,
            run_dir=run_dir,
            log_file=log_file,
        )
        reconstruction_handles.append(handle)
        return [handle]

    try:
        result = eval_libero(
            cfg,
            extra_hook_applier=applier if args.mode == "reconstruction" else None,
        )
    finally:
        # eval_libero removes hooks after a normal run. This idempotent cleanup
        # also covers a fatal hook/model error during a rollout.
        for handle in reconstruction_handles:
            handle.remove()
    run_dir = Path(result.run_dir)
    stdout_path = run_dir / "stdout.log"
    stdout_text = stdout_path.read_text(encoding="utf-8")
    if "Caught exception:" in stdout_text:
        raise RuntimeError(
            f"One or more rollouts caught an exception; inspect {stdout_path}"
        )

    completed_rollouts = _count_jsonl(result.prompt_records_path)
    if completed_rollouts != args.expected_rollouts:
        raise RuntimeError(
            f"Expected {args.expected_rollouts} rollouts, found {completed_rollouts}"
        )

    payload = {
        "schema_version": "event_sae_policy_evaluation_v1",
        "mode": args.mode,
        "run_dir": str(run_dir),
        "completed_rollouts": completed_rollouts,
        "success_rate": result.success_rate,
        "model_checkpoint": cfg.model.checkpoint,
        "model_revision": cfg.model.revision,
        "model_code_revision": cfg.model.code_revision,
        "task_suite": cfg.env.task_suite_name,
        "seed": cfg.env.seed,
        "requested_task_ids": cfg.env.task_ids,
        "num_trials_per_task": cfg.env.num_trials_per_task,
        "code": code,
        "run_config": _protocol_config(cfg),
    }
    if args.mode == "reconstruction":
        checkpoint = Path(args.sae_checkpoint).expanduser().resolve()
        payload.update(
            {
                "sae_checkpoint": str(checkpoint),
                "sae_sha256": _sha256(checkpoint),
                "layer_idx": args.layer_idx,
                "hook_metrics": reconstruction_handles[0].summary(),
            }
        )
    fingerprint_input = {
        key: payload[key]
        for key in (
            "mode",
            "completed_rollouts",
            "run_config",
            "code",
        )
    }
    if args.mode == "reconstruction":
        fingerprint_input.update(
            {
                "sae_sha256": payload["sae_sha256"],
                "layer_idx": payload["layer_idx"],
            }
        )
    payload["protocol_fingerprint"] = _fingerprint(fingerprint_input)

    output_path = run_dir / "policy_evaluation.json"
    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(output_path)
    if args.result_output is not None:
        stable_output = Path(args.result_output).expanduser().resolve()
        stable_output.parent.mkdir(parents=True, exist_ok=True)
        stable_temporary = stable_output.with_suffix(stable_output.suffix + ".tmp")
        stable_temporary.write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        stable_temporary.replace(stable_output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
