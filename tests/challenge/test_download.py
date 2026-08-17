from __future__ import annotations

import json
from pathlib import Path

import pytest

from imas_ambix.challenge.download import REVISION, InventoryItem, read_inventory


def test_inventory_rejects_a_different_revision(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        json.dumps({"kind": "inventory", "revision": "moving-target"}) + "\n"
    )
    with pytest.raises(ValueError, match="does not match"):
        read_inventory(inventory)


def test_inventory_reads_pinned_objects(tmp_path: Path) -> None:
    inventory = tmp_path / "inventory.jsonl"
    item = InventoryItem(path="data/example.parquet", size=12, sha256="abc")
    inventory.write_text(
        json.dumps({"kind": "inventory", "revision": REVISION})
        + "\n"
        + json.dumps(item.__dict__)
        + "\n"
    )
    assert read_inventory(inventory) == [item]
