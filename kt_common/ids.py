"""Chuẩn hoá định danh người học và tên khái niệm."""

from __future__ import annotations


def as_id(x):
    """``int`` nếu ép được (id số trong CSV thường bị đọc thành float/str), giữ nguyên nếu không."""
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def skill_label(lookup, skill_id) -> str:
    """``"<id>:<tên>"`` khi có bảng tra, ngược lại chỉ ``"<id>"``."""
    if lookup and skill_id in lookup:
        return f"{skill_id}:{lookup[skill_id]}"
    return str(skill_id)
