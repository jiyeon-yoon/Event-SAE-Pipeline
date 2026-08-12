"""Upload only derived Spatial reproduction artifacts and verify them remotely.

The merged run contains symlinks back to roughly 281 GiB of source data and is
deliberately excluded.  Discovery outputs are resumably uploaded from
``<work-dir>/pipeline``; smaller closed-loop result folders can be added later.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


DISCOVERY_REQUIRED = (
    "offline_fidelity.json",
    "keyframes/waypoint_summary.json",
    "clusters/summary.json",
    "scores/event_feature_scores.pt",
    "rankings/candidates.jsonl",
    "topk/manifest.json",
)


def _remote_metadata(api: HfApi, repo_id: str) -> dict[str, int | None]:
    info = api.dataset_info(repo_id, files_metadata=True)
    return {item.rfilename: item.size for item in info.siblings}


def _verify_remote_sizes(
    *, remote: dict[str, int | None], local_root: Path, relative_paths: tuple[str, ...]
) -> None:
    for relative in relative_paths:
        local = local_root / relative
        if not local.is_file():
            raise FileNotFoundError(local)
        remote_size = remote.get(relative)
        if remote_size != local.stat().st_size:
            raise RuntimeError(
                f"Remote size mismatch for {relative}: remote={remote_size}, "
                f"local={local.stat().st_size}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument(
        "--section",
        choices=("discovery", "hooked-sr", "intervention"),
        default="discovery",
    )
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    work = Path(args.work_dir).expanduser().resolve()
    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    if args.section == "discovery":
        local_root = work / "pipeline"
        summary = work / "pipeline_summary.json"
        if not summary.is_file():
            raise FileNotFoundError(summary)
        # upload_large_folder resumes completed files after a disconnect.
        api.upload_large_folder(
            repo_id=args.repo_id,
            repo_type="dataset",
            folder_path=local_root,
        )
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="dataset",
            path_or_fileobj=summary,
            path_in_repo="pipeline_summary.json",
        )
        remote = _remote_metadata(api, args.repo_id)
        _verify_remote_sizes(
            remote=remote,
            local_root=local_root,
            relative_paths=DISCOVERY_REQUIRED,
        )
        if remote.get("pipeline_summary.json") != summary.stat().st_size:
            raise RuntimeError("Remote pipeline_summary.json size mismatch")
        print(f"REMOTE_DISCOVERY_OK: https://huggingface.co/datasets/{args.repo_id}")
        return

    section_root = work / args.section
    if not section_root.is_dir():
        raise FileNotFoundError(section_root)
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="dataset",
        folder_path=section_root,
        path_in_repo=args.section,
    )
    remote = _remote_metadata(api, args.repo_id)
    local_files = tuple(
        path.relative_to(section_root).as_posix()
        for path in section_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    prefixed_remote = {
        path.removeprefix(f"{args.section}/"): size
        for path, size in remote.items()
        if path.startswith(f"{args.section}/")
    }
    _verify_remote_sizes(
        remote=prefixed_remote,
        local_root=section_root,
        relative_paths=local_files,
    )
    print(
        f"REMOTE_{args.section.upper().replace('-', '_')}_OK: "
        f"https://huggingface.co/datasets/{args.repo_id}/tree/main/{args.section}"
    )


if __name__ == "__main__":
    main()
