"""CLI: run an openVLA LIBERO eval with a single-feature SAE intervention hook.

For each ``(feature_id, alpha)`` pair, runs ``eval_libero`` with the
residual-preserving latent-edit hook applied at the configured SAE layer
and reports closed-loop success rate. Compare against a baseline run
(same config, no ``--intervene-feature-id``) to get the SR delta.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from event_sae.openvla.eval.config import load_config, parse_overrides, resolve_task_ids
from event_sae.openvla.eval.runner import eval_libero
from event_sae.openvla.intervene import (
    InterventionHookHandle,
    apply_resid_post_feature_perturb_hook,
)


def _count_jsonl(path: str | None) -> int:
    if path is None:
        return 0
    with Path(path).open("r", encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    }


def _protocol_config(cfg) -> dict:
    payload = asdict(cfg)
    payload["logging"].pop("root_dir", None)
    payload["env"]["resolved_task_ids"] = resolve_task_ids(cfg.env.task_ids, 10)
    return payload


def _fingerprint(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run LIBERO eval with a single-feature SAE intervention.")
    ap.add_argument("--config", required=True, help="YAML eval config (same schema as collect_activations).")
    ap.add_argument(
        "--override", action="append", default=[], help="Dotted key=value overrides for the YAML (repeatable)."
    )
    ap.add_argument("--sae-checkpoint", required=True, help="Path to a post_mlp_residual BatchTopKSAE ae.pt.")
    ap.add_argument("--layer-idx", type=int, required=True, help="Decoder layer index to hook.")
    ap.add_argument("--feature-id", type=int, required=True, help="SAE feature column index to perturb.")
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="Scaling applied to z[feature_id]: 0 zeros out, (0,1) suppresses, 1 no-op, >1 amplifies.",
    )
    ap.add_argument(
        "--hook-start-step",
        type=int,
        default=0,
        help="Step in episode at which the hook becomes active (set >0 to skip a warm-up).",
    )
    ap.add_argument("--expected-rollouts", type=int, required=True)
    ap.add_argument("--expected-code-revision", required=True)
    ap.add_argument("--result-output", required=True)
    args = ap.parse_args()

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

    handles: list[InterventionHookHandle] = []

    def applier(*, model, cfg, run_dir, log_file):
        del cfg
        handle = apply_resid_post_feature_perturb_hook(
            model=model,
            layer_idx=args.layer_idx,
            sae_checkpoint_path=args.sae_checkpoint,
            feature_idx=args.feature_id,
            alpha=args.alpha,
            hook_start_step=args.hook_start_step,
            run_dir=run_dir,
            log_file=log_file,
        )
        handles.append(handle)
        return [handle]

    try:
        result = eval_libero(cfg, extra_hook_applier=applier)
    finally:
        for handle in handles:
            handle.remove()

    run_dir = Path(result.run_dir)
    stdout_path = run_dir / "stdout.log"
    if "Caught exception:" in stdout_path.read_text(encoding="utf-8"):
        raise RuntimeError(f"A rollout caught an exception; inspect {stdout_path}")
    completed = _count_jsonl(result.prompt_records_path)
    if completed != args.expected_rollouts:
        raise RuntimeError(
            f"Expected {args.expected_rollouts} rollouts, found {completed}"
        )
    hook_metrics = handles[0].summary()
    if int(hook_metrics["num_forwards"]) <= 0:
        raise RuntimeError("Intervention hook did not run")

    checkpoint = Path(args.sae_checkpoint).expanduser().resolve()
    payload = {
        "schema_version": "event_sae_feature_intervention_v1",
        "mode": "intervention",
        "run_dir": str(run_dir),
        "completed_rollouts": completed,
        "success_rate": result.success_rate,
        "prompt_records_path": result.prompt_records_path,
        "episode_results_path": result.episode_results_path,
        "actions_path": str(run_dir / "actions.json") if cfg.logging.save_actions else None,
        "feature_id": args.feature_id,
        "alpha": args.alpha,
        "layer_idx": args.layer_idx,
        "hook_start_step": args.hook_start_step,
        "hook_metrics": hook_metrics,
        "sae_checkpoint": str(checkpoint),
        "sae_sha256": _sha256(checkpoint),
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
    payload["protocol_fingerprint"] = _fingerprint(
        {
            key: payload[key]
            for key in (
                "mode",
                "completed_rollouts",
                "feature_id",
                "alpha",
                "layer_idx",
                "hook_start_step",
                "sae_sha256",
                "run_config",
                "code",
            )
        }
    )
    output = Path(args.result_output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2))
    print(f"INTERVENTION_OK: {output}")


if __name__ == "__main__":
    main()
