from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.openvla.verify_extended_dataset_upload import (  # noqa: E402
    compare_inventories,
    local_inventory,
)


def test_local_inventory_ignores_upload_cache(tmp_path: Path):
    (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    cache = tmp_path / ".cache" / "huggingface"
    cache.mkdir(parents=True)
    (cache / "state").write_text("ignored", encoding="utf-8")
    assert local_inventory(tmp_path) == {"manifest.json": 2}


def test_compare_inventories_detects_missing_and_wrong_size():
    missing, wrong = compare_inventories(
        {"a": 3, "b": 4, "c": 5}, {"a": 3, "b": 8}
    )
    assert missing == ["c"]
    assert wrong == [("b", 4, 8)]
