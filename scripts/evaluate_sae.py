"""CLI: evaluate offline SAE reconstruction fidelity on dense shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.evaluate import OfflineFidelityConfig, evaluate_offline_fidelity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute global FVE, reconstruction MSE, alive fraction, and average L0."
    )
    parser.add_argument(
        "--data-dir",
        action="append",
        required=True,
        help=(
            "Activation directory to search recursively. Repeat for multiple roots; "
            "a common parent directory is also accepted."
        ),
    )
    parser.add_argument(
        "--sae-checkpoint", required=True, help="ae.pt or its trainer directory"
    )
    parser.add_argument("--output", required=True, help="Path to the output JSON file")
    parser.add_argument(
        "--layer-idx", type=int, default=None, help="Default: checkpoint layer"
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--device", default="", help="Default: cuda:0 when available, else cpu"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional smoke-test limit. Omit for the full evaluation.",
    )
    args = parser.parse_args()

    result = evaluate_offline_fidelity(
        OfflineFidelityConfig(
            data_dirs=tuple(args.data_dir),
            sae_checkpoint_path=args.sae_checkpoint,
            output_path=args.output,
            layer_idx=args.layer_idx,
            batch_size=args.batch_size,
            device=args.device,
            max_rows=args.max_rows,
        )
    )
    print(json.dumps(result["metrics"], indent=2))
    print(f"FIDELITY_OK: {Path(args.output).expanduser().resolve()}")


if __name__ == "__main__":
    main()
