"""Đọc header file và dò dữ liệu đã attach trong ``/kaggle/input`` (không phụ thuộc module khác)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def _probe_header(path) -> list:
    """Đọc dòng header của file GỐC (BOM đã được gỡ) — không nạp dữ liệu."""
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)

def find_kaggle_input(
    patterns=("junyi",), preferred_names=("Log_Problem.csv",)
) -> Optional[Path]:
    """dò file Junyi trong /kaggle/input, ưu tiên tên quen thuộc; None nếu chưa attach."""
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    hits = [
        p for p in root.rglob("*.csv") if any(k in str(p).lower() for k in patterns)
    ]
    if not hits:
        return None
    for name in preferred_names:
        for p in hits:
            if p.name.lower() == name.lower():
                return p
    return max(hits, key=lambda p: p.stat().st_size)

def list_kaggle_inputs(max_files: int = 40) -> str:
    """Mô tả ngắn những gì THỰC SỰ có trong /kaggle/input, dùng trong thông báo lỗi.

    Phân biệt được ba tình huống mà trước đây đều ra cùng một FileNotFoundError: chưa attach
    gì, attach rồi nhưng Kaggle không giải nén (dataset phân mảnh/thiếu quyền), và attach đúng
    nhưng tên file/thư mục khác mẫu dò.
    """
    root = Path("/kaggle/input")
    if not root.exists():
        return "không có /kaggle/input (không phải môi trường Kaggle)."
    try:
        mounts = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        return f"không đọc được /kaggle/input: {exc}"
    if not mounts:
        return "/kaggle/input RỖNG — chưa attach dataset nào (Add input > Datasets)."
    lines = [f"/kaggle/input có {len(mounts)} dataset đã attach:"]
    for m in mounts:
        try:
            files = sorted(
                m.rglob("*"),
                key=lambda p: p.stat().st_size if p.is_file() else -1,
                reverse=True,
            )
            csvs = [p for p in files if p.is_file() and p.suffix.lower() == ".csv"]
            size = sum(p.stat().st_size for p in csvs) / 1024 / 1024
            shown = ", ".join(
                f"{p.relative_to(m)} ({p.stat().st_size / 1024 / 1024:.0f} MB)"
                for p in (csvs or files)[:5]
            )
            lines.append(
                f"  {m.name}: {len(csvs)} file CSV, {size:,.0f} MB | {shown or '(trống — chưa giải nén?)'}"
            )
            if len(csvs) > 5:
                lines.append(f"      ... còn lại {len(csvs) - 5} file CSV khác")
        except OSError as exc:
            lines.append(f"  {m.name}: không liệt kê được ({exc})")
    return "\n".join(lines[:max_files])
