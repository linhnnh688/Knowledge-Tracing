"""Checkpoint dùng chung: ghi nguyên tử, state/optimizer/RNG, fingerprint, quy ước tên file.

Trước đây ``level2/baselines.py`` và ``level2/grkt_api.py`` mỗi file có một bản riêng của
cùng các hàm này. Định dạng payload (khoá, chuỗi ``format``) KHÔNG đổi, nên checkpoint
resume đã sinh trên Kaggle vẫn nạp được.

Quy ước tên file (mục 1.4 của ``docs/tasks/improve_checklist.md``)
------------------------------------------------------------------
Mọi checkpoint mới mang seed trong tên: ``<prefix><name>-seed<seed>.pt``. Bản cũ đặt tên
seed 42 là ``<prefix><name>.pt`` (không hậu tố) — chính quy ước "magic number" đó từng để
vòng đa seed ghi đè checkpoint seed 42. :func:`find_checkpoint` vẫn tìm tên cũ để đọc được
sản phẩm đã có, nhưng không bao giờ GHI tên cũ nữa. Mỗi lần ghi, :func:`record_checkpoint`
cập nhật ``checkpoints_manifest.json`` trong cùng thư mục (tên file -> seed, cấu hình...).
"""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .log import get_logger

log = get_logger("checkpoint")

#: Seed mà quy ước tên cũ ngầm định cho file không có hậu tố ``-seed<s>``.
LEGACY_UNSUFFIXED_SEED = 42
MANIFEST_NAME = "checkpoints_manifest.json"
_SEED_RE = re.compile(r"-seed(\d+)\.pt$")


# ---------------------------------------------------------------- tên file
def checkpoint_path(directory, name: str, seed: int, prefix: str = "") -> Path:
    """Đường dẫn GHI checkpoint: luôn có hậu tố seed."""
    return Path(directory) / f"{prefix}{name}-seed{int(seed)}.pt"


def legacy_checkpoint_path(directory, name: str, seed: int, prefix: str = "") -> Optional[Path]:
    """Tên theo quy ước cũ, chỉ khác tên mới khi ``seed == 42``."""
    if int(seed) != LEGACY_UNSUFFIXED_SEED:
        return None
    return Path(directory) / f"{prefix}{name}.pt"


def find_checkpoint(directory, name: str, seed: int, prefix: str = "") -> Optional[Path]:
    """Checkpoint đã có của ``(name, seed)``: ưu tiên tên mới, rồi tới tên cũ; ``None`` nếu không có."""
    if directory is None:
        return None
    new = checkpoint_path(directory, name, seed, prefix)
    if new.exists():
        return new
    old = legacy_checkpoint_path(directory, name, seed, prefix)
    if old is not None and old.exists():
        return old
    return None


def seed_from_filename(path) -> Optional[int]:
    m = _SEED_RE.search(Path(path).name)
    return int(m.group(1)) if m else None


def _git_sha() -> Optional[str]:
    try:
        root = Path(__file__).resolve().parents[1]
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - không có git (vd zip) thì bỏ qua
        return None


def record_checkpoint(path, **meta) -> None:
    """Ghi/ghi đè một dòng trong ``checkpoints_manifest.json`` cạnh ``path``."""
    path = Path(path)
    manifest = path.parent / MANIFEST_NAME
    try:
        data = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    except (OSError, ValueError):
        data = {}
    entry = {k: v for k, v in meta.items() if v is not None}
    entry.setdefault("seed", seed_from_filename(path))
    entry["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    entry["git_sha"] = _git_sha()
    data[path.name] = entry
    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, manifest)


def manifest_entry(path) -> Optional[dict]:
    """Dòng manifest của ``path`` (``None`` nếu không có manifest/dòng)."""
    path = Path(path)
    manifest = path.parent / MANIFEST_NAME
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    entry = data.get(path.name) if isinstance(data, dict) else None
    return entry if isinstance(entry, dict) else None


def manifest_seed(path) -> Optional[int]:
    """Seed ghi trong manifest cho ``path`` (``None`` nếu không có)."""
    path = Path(path)
    manifest = path.parent / MANIFEST_NAME
    if not manifest.exists():
        return None
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    seed = (data.get(path.name) or {}).get("seed") if isinstance(data, dict) else None
    return int(seed) if seed is not None else None


# ---------------------------------------------------------------- ghi / state
def atomic_save(payload, path) -> None:
    """File tạm + ``os.replace``: session bị kill giữa lúc ghi không làm hỏng checkpoint cũ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def cpu_state_dict(model_or_state) -> Dict[str, torch.Tensor]:
    """Bản sao state_dict trên CPU (nhận model hoặc state_dict)."""
    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, "state_dict")
        else model_or_state
    )
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def optimizer_state_cpu(optimizer) -> dict:
    """Trạng thái optimizer về CPU: checkpoint nạp lại ở phiên khác không phụ thuộc GPU."""
    state = optimizer.state_dict()
    return {
        "state": {
            idx: {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in entry.items()
            }
            for idx, entry in state["state"].items()
        },
        "param_groups": state["param_groups"],
    }


def optimizer_state_to_device(optimizer, device) -> None:
    for entry in optimizer.state.values():
        for k, v in entry.items():
            if torch.is_tensor(v):
                entry[k] = v.to(device)


def rng_state() -> dict:
    """RNG của Python/NumPy/PyTorch(+CUDA) để resume tiếp đúng dòng ngẫu nhiên."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state: Optional[dict]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as exc:  # noqa: BLE001 - RNG khác thiết bị không đáng làm chết lần resume
        log.info(
            f"  [resume] không khôi phục được RNG ({type(exc).__name__}: {exc}) — train tiếp "
            "với dòng ngẫu nhiên mới; kết quả không trùng bit-for-bit với lần chạy liền mạch."
        )


# ---------------------------------------------------------------- fingerprint
def check_fingerprint(previous: Optional[dict], current: Optional[dict], hint: str) -> None:
    """Raise nếu checkpoint được tạo với cấu hình khác (``load_state_dict`` không bắt được).

    Bỏ qua khi một trong hai bên trống (checkpoint đời cũ không có fingerprint). ``hint`` là
    câu hướng dẫn xoá file nào, ghép vào cuối thông báo lỗi.
    """
    if not previous or not current:
        return
    diff = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
    if diff:
        detail = ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in diff.items())
        raise RuntimeError(f"Checkpoint được tạo với cấu hình khác ({detail}). {hint}")
