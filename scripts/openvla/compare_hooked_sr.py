"""Compare existing raw SR with a validated reconstruction-hook evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compare_results(
    raw: dict, reconstruction: dict, *, expected_code_revision: str | None = None
) -> dict:
    if raw.get("mode") != "raw":
        raise RuntimeError("Expected a raw-policy evaluation")
    if reconstruction.get("mode") != "reconstruction":
        raise RuntimeError("Expected a reconstruction-hook policy evaluation")
    for label, result in (("raw", raw), ("reconstruction", reconstruction)):
        code = result.get("code")
        if not isinstance(code, dict) or not isinstance(code.get("commit"), str) or not code["commit"]:
            raise RuntimeError(f"{label} result is missing required code provenance")
        if code.get("dirty") is not False:
            raise RuntimeError(f"{label} result does not prove a clean Git worktree")
        run_config = result.get("run_config")
        if not isinstance(run_config, dict) or not run_config:
            raise RuntimeError(f"{label} result is missing required run_config")
        if not isinstance(run_config.get("model"), dict) or not isinstance(
            run_config.get("env"), dict
        ):
            raise RuntimeError(f"{label} result has an incomplete run_config")
    protocol_keys = (
        "model_checkpoint",
        "model_revision",
        "model_code_revision",
        "task_suite",
        "seed",
        "requested_task_ids",
        "num_trials_per_task",
        "code",
        "run_config",
    )
    mismatches = {
        key: (raw.get(key), reconstruction.get(key))
        for key in protocol_keys
        if raw.get(key) != reconstruction.get(key)
    }
    if mismatches:
        raise RuntimeError(f"Raw/reconstruction protocols differ: {mismatches}")
    if expected_code_revision is not None:
        for label, result in (("raw", raw), ("reconstruction", reconstruction)):
            if result.get("code", {}).get("commit") != expected_code_revision:
                raise RuntimeError(
                    f"{label} result commit does not match --expected-code-revision"
                )
    if int(raw["completed_rollouts"]) != 100:
        raise RuntimeError("Raw comparison must contain 100 matching rollouts")
    if int(reconstruction["completed_rollouts"]) != 100:
        raise RuntimeError("Reconstruction evaluation must contain 100 rollouts")
    hook = reconstruction.get("hook_metrics", {})
    if int(hook.get("num_forwards", 0)) <= 0 or int(hook.get("num_tokens", 0)) <= 0:
        raise RuntimeError("Reconstruction hook did not fire")

    raw_sr = float(raw["success_rate"])
    reconstruction_sr = float(reconstruction["success_rate"])
    return {
        "schema_version": "event_sae_hooked_sr_comparison_v1",
        "raw_success_rate": raw_sr,
        "reconstruction_success_rate": reconstruction_sr,
        "absolute_delta": reconstruction_sr - raw_sr,
        "absolute_drop": raw_sr - reconstruction_sr,
        "rollouts_per_condition": 100,
        "protocol": {key: raw.get(key) for key in protocol_keys},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-result", required=True)
    parser.add_argument("--reconstruction-result", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--expected-code-revision", required=True)
    args = parser.parse_args()

    raw = json.loads(Path(args.raw_result).read_text(encoding="utf-8"))
    reconstruction = json.loads(
        Path(args.reconstruction_result).read_text(encoding="utf-8")
    )
    result = {
        **compare_results(
            raw,
            reconstruction,
            expected_code_revision=args.expected_code_revision,
        ),
        "raw_result": str(Path(args.raw_result).resolve()),
        "reconstruction_result": str(Path(args.reconstruction_result).resolve()),
    }
    output = Path(args.output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2))
    print(f"HOOKED_SR_OK: {output}")


if __name__ == "__main__":
    main()
