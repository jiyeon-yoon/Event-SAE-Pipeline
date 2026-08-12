"""CLI: compute vision embeddings + state vectors for each keyframe sample.

Reads `samples.jsonl` produced by `scripts/extract_keyframe_media.py`, writes
`event_features.jsonl` ready for `scripts/cluster_events.py`.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from event_sae.events.build_features import build_event_features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vision embeddings + state vectors per sample."
    )
    parser.add_argument("--samples-path", required=True, help="Path to samples.jsonl")
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output JSONL (default: event_features.jsonl next to samples.jsonl)",
    )
    parser.add_argument(
        "--vision-model-name-or-path",
        default="google/siglip-base-patch16-224",
        help="Frozen vision encoder",
    )
    parser.add_argument(
        "--vision-model-revision",
        default=None,
        help="Optional immutable Hugging Face commit for the vision encoder.",
    )
    parser.add_argument("--device", default=None, help="Torch device (default: cuda if available)")
    parser.add_argument(
        "--frame-positions",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Frame indices (into sample.frame_paths) used for the vision embedding",
    )
    args = parser.parse_args()

    samples_path = Path(args.samples_path).resolve()
    output_path = (
        Path(args.output_path).resolve()
        if args.output_path is not None
        else samples_path.with_name("event_features.jsonl").resolve()
    )

    build_event_features(
        samples_path=samples_path,
        output_path=output_path,
        vision_model_name_or_path=args.vision_model_name_or_path,
        vision_model_revision=args.vision_model_revision,
        device=args.device,
        frame_positions=list(args.frame_positions),
    )


if __name__ == "__main__":
    main()
