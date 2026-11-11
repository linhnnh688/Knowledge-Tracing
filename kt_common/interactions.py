"""Một nguồn tiền xử lý duy nhất cho CSV tương tác (mục 2.1 của improve_checklist).

``GKT/processing.load_dataset`` (Cấp độ 1) và ``level2/grkt_data.build_grkt_data`` (Cấp độ 2)
cùng gọi :func:`load_interactions`, nên mọi thay đổi tiền xử lý chỉ sửa ở MỘT chỗ.

Các bước:

1. bỏ dòng thiếu ``skill_id``;
2. **sắp chuỗi từng người học theo ``order_id``** (ổn định), giữ thứ tự người học theo lần xuất
   hiện đầu tiên — file ASSISTments 2009 gốc KHÔNG theo thời gian (~27 nghìn lần ``order_id``
   đi lùi), trong khi mô hình chỉ được dùng tương tác TRƯỚC thời điểm dự đoán (checklist 0.2);
3. **gộp dòng trùng ``(user_id, order_id)``**, giữ dòng có ``skill_id`` NHỎ NHẤT (tất định, không
   phụ thuộc thứ tự file; trùng quy ước với ``prepare_interactions`` của nhánh
   ``fix/assist2009-leak-batchnorm``): file gốc nhân bản MỘT lượt trả lời
   thành nhiều dòng khi câu hỏi gắn nhiều kỹ năng (~34% số dòng); hai dòng liền nhau mang CÙNG
   nhãn nên mô hình "đoán" bước t+1 sau khi đã thấy chính nhãn đó — rò nhãn (Xiong et al., EDM
   2016; pyKT, NeurIPS 2022; checklist 0.1). Mô hình ở đây nhận một khái niệm mỗi bước nên
   khái niệm không bao giờ là ``skill_id`` nhỏ nhất của lượt nào thì bị mất. Q-matrix đa khái
   niệm cần sửa mô hình;
4. bỏ người học có ít hơn ``min_seq_len`` lượt (đếm SAU khi gộp);
5. ``skill`` = ``pd.factorize(skill_id, sort=True)`` (0-based).

Bước 2–3 chỉ chạy khi file có cột ``order_id``. CSV chuẩn hoá của Junyi đã sắp theo thời gian
và mỗi lượt một dòng nên không đổi gì.

Số liệu trong ``results/`` và ``archive/`` được tính TRƯỚC khi có bước 2–3 (dữ liệu còn trùng
và sai thứ tự) nên không so trực tiếp được với lần chạy mới.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("skill_id", "correct", "user_id")

#: Phiên bản quy trình tiền xử lý. Ghi vào fingerprint/manifest của mọi checkpoint mới: checkpoint
#: không mang đúng giá trị này (vd train trên dữ liệu trùng/sai thứ tự trước bản sửa) bị từ chối
#: thay vì nạp âm thầm — cùng shape nên ``load_state_dict`` không bắt được.
DATA_VERSION = "order-dedup-v1"


def _sort_and_dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if "order_id" not in df.columns:
        return df
    # Thứ hạng người học theo lần xuất hiện đầu tiên -> giữ thứ tự người học như file.
    user_rank = pd.Series(pd.factorize(df["user_id"])[0], index=df.index)
    df = df.assign(_u=user_rank).sort_values(["_u", "order_id", "skill_id"], kind="mergesort")
    df = df.drop_duplicates(subset=["user_id", "order_id"], keep="first")
    return df.drop(columns="_u")


def load_interactions(path, min_seq_len: int = 3) -> pd.DataFrame:
    """Đọc CSV, kiểm cột bắt buộc, sắp + gộp, lọc và đánh chỉ số khái niệm (cột ``skill``)."""
    df = pd.read_csv(path)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"The column '{col}' was not found on {path}")
    df = df.dropna(subset=["skill_id"])
    df = _sort_and_dedupe(df)
    sizes = df.groupby("user_id")["user_id"].transform("size")
    df = df[sizes >= min_seq_len].copy()
    df["skill"], _ = pd.factorize(df["skill_id"], sort=True)
    return df


def iter_users(df: pd.DataFrame) -> Iterator[Tuple[object, pd.DataFrame]]:
    """``(user_id, khung dòng của người đó)`` theo thứ tự xuất hiện đầu tiên trong file."""
    return iter(df.groupby("user_id", sort=False))


def truncate_len(n: int, max_seq_len) -> int:
    """Số bước giữ lại: ``max_seq_len`` = 0/None nghĩa là KHÔNG cắt."""
    return min(int(max_seq_len), n) if max_seq_len else n


def skill_lookup(df: pd.DataFrame) -> dict:
    """``{skill (0-based): skill_name}`` nếu file có cột ``skill_name``."""
    if "skill_name" not in df.columns:
        return {}
    first = df[["skill", "skill_name"]].drop_duplicates("skill")
    return {int(k): str(v) for k, v in zip(first["skill"], first["skill_name"])}


def sequence_order_check(order_ids) -> dict:
    """Đếm triệu chứng rò rỉ trên ĐÚNG chuỗi đưa vào mô hình (danh sách ``order_id`` mỗi người học).

    ``non_increasing``: bước có ``order_id`` không lớn hơn bước trước (tương lai nằm trong lịch
    sử); ``duplicate_attempts``: bước trùng ``order_id`` với bước trước (cùng một lượt). Cả hai
    phải bằng 0. ``checked=False`` khi file không có ``order_id`` (không kiểm được).
    """
    out = {"users": len(order_ids), "targets": 0, "non_increasing": 0, "duplicate_attempts": 0}
    for seq in order_ids:
        for prev, cur in zip(seq[:-1], seq[1:]):
            out["targets"] += 1
            out["non_increasing"] += int(cur <= prev)
            out["duplicate_attempts"] += int(cur == prev)
    out["checked"] = any(len(s) for s in order_ids)
    return out


def order_report_lines(dataset) -> list:
    """Dòng leak report về thứ tự thời gian + lượt trùng (dùng chung Level 1 và Level 2)."""
    st = dict(getattr(dataset, "data_stats", {}) or {})
    chk = sequence_order_check(getattr(dataset, "order_ids", []) or [])
    lines = [
        f"Tiền xử lý ({st.get('data_version', DATA_VERSION)}): {st.get('users')} người học, "
        f"{st.get('interactions')} lượt, {st.get('concepts')} khái niệm"
    ]
    if chk["checked"]:
        lines.append(
            f"Thứ tự trong chuỗi: {chk['targets']} bước dự đoán, order_id không tăng="
            f"{chk['non_increasing']}, lượt trùng={chk['duplicate_attempts']} (phải = 0)"
        )
    else:
        lines.append("Thứ tự trong chuỗi: file không có order_id -> giữ thứ tự file (không kiểm được)")
    if st.get("max_seq_len"):
        lines.append(
            f"Cắt chuỗi max_seq_len={st['max_seq_len']}: bỏ {st.get('truncated_rows')} lượt "
            f"({100 * st.get('truncated_frac', 0):.1f}%) của {st.get('users_truncated')} người học"
        )
    return lines


def interaction_stats(df: pd.DataFrame, max_seq_len=None) -> dict:
    """Số đo chất lượng dữ liệu SAU tiền xử lý — ghi vào artifact để báo cáo nói rõ.

    * ``dup_adjacent``: số cặp bước liền nhau là CÙNG một lượt (phải bằng 0);
    * ``order_backsteps``: số lần ``order_id`` đi lùi trong chuỗi một người học (phải bằng 0);
    * ``truncated_rows``/``truncated_frac`` (checklist 0.3): tương tác bị ``max_seq_len`` bỏ đi.
    """
    lengths = df.groupby("user_id", sort=False).size().to_numpy()
    stats = {
        "data_version": DATA_VERSION,
        "users": int(len(lengths)),
        "interactions": int(lengths.sum()),
        "concepts": int(df["skill"].nunique()) if "skill" in df.columns else None,
    }
    if "order_id" in df.columns:
        diff = df.groupby("user_id", sort=False)["order_id"].diff()
        stats["dup_adjacent"] = int((diff == 0).sum())
        stats["order_backsteps"] = int((diff < 0).sum())
    if max_seq_len:
        cut = np.clip(lengths - int(max_seq_len), 0, None)
        stats["max_seq_len"] = int(max_seq_len)
        stats["users_truncated"] = int((cut > 0).sum())
        stats["truncated_rows"] = int(cut.sum())
        stats["truncated_frac"] = float(cut.sum() / max(1, lengths.sum()))
    return stats
