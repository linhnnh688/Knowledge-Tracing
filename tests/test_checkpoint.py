"""Quy ước tên checkpoint (luôn có seed), tương thích tên cũ, manifest, fingerprint, metric."""

import json

import numpy as np
import pytest
import torch

from kt_common.checkpoint import (
    atomic_save,
    check_fingerprint,
    checkpoint_path,
    find_checkpoint,
    manifest_seed,
    record_checkpoint,
)
from kt_common.metrics import auc_acc


def test_checkpoint_path_always_has_seed(tmp_path):
    assert checkpoint_path(tmp_path, "DKT", 42).name == "DKT-seed42.pt"
    assert checkpoint_path(tmp_path, "GRKT-train", 7, prefix="ckpt_").name == "ckpt_GRKT-train-seed7.pt"


def test_find_checkpoint_prefers_new_then_legacy(tmp_path):
    assert find_checkpoint(tmp_path, "DKT", 42) is None
    legacy = tmp_path / "DKT.pt"
    legacy.write_bytes(b"x")
    assert find_checkpoint(tmp_path, "DKT", 42) == legacy
    # Tên cũ không hậu tố chỉ có nghĩa là seed 42.
    assert find_checkpoint(tmp_path, "DKT", 123) is None
    new = checkpoint_path(tmp_path, "DKT", 42)
    new.write_bytes(b"y")
    assert find_checkpoint(tmp_path, "DKT", 42) == new
    assert find_checkpoint(None, "DKT", 42) is None


def test_record_checkpoint_manifest(tmp_path):
    p = checkpoint_path(tmp_path, "GKT-Zero", 456)
    atomic_save({"a": torch.zeros(1)}, p)
    record_checkpoint(p, name="GKT-Zero", config={"lr": 1e-3})
    data = json.loads((tmp_path / "checkpoints_manifest.json").read_text(encoding="utf-8"))
    assert data[p.name]["seed"] == 456 and data[p.name]["config"]["lr"] == 1e-3
    assert manifest_seed(p) == 456
    assert not list(tmp_path.glob("*.tmp"))


def test_checkpoint_seed_sources(tmp_path):
    from level2.baselines import checkpoint_seed

    assert checkpoint_seed(tmp_path / "DKT-seed123.pt") == 123
    legacy = tmp_path / "DKT.pt"
    legacy.write_bytes(b"x")
    assert checkpoint_seed(legacy) is None  # không đoán 42
    (tmp_path / "checkpoint_seeds.json").write_text(json.dumps({"DKT.pt": 42}))
    assert checkpoint_seed(legacy) == 42


def test_check_fingerprint():
    check_fingerprint(None, {"a": 1}, "hint")
    check_fingerprint({"a": 1}, {}, "hint")
    check_fingerprint({"a": 1, "b": 2}, {"a": 1}, "hint")
    with pytest.raises(RuntimeError, match="a: 1 -> 2.*hint"):
        check_fingerprint({"a": 1}, {"a": 2}, "hint")


def test_auc_acc():
    assert auc_acc([], []) == (-1.0, -1.0)
    assert auc_acc([1, 1], [0.2, 0.9]) == (-1.0, -1.0)
    auc, acc = auc_acc(np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.6, 0.4]))
    assert auc == pytest.approx(0.75) and acc == pytest.approx(0.5)
