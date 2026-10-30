"""Phân tích nâng cao Cấp độ 2: hiệu chỉnh dự đoán (ECE + Brier) cho mọi mô hình.

Cùng một định nghĩa được dùng cho DKT, GKT và GRKT để so sánh được:

* ``Brier = mean((p - y)^2)``          : sai số bình phương trung bình trên nhãn 0/1.
* ``ECE  = Σ_b (n_b / N) * |avg(p_b) - avg(y_b)|`` trên 10 ngăn đều nhau theo ``p``.
  ECE bỏ qua "ngăn" nào rỗng; nếu một mô hình không dự đoán trong một khoảng nào đó thì
  khoảng đó không đóng góp, nên ECE luôn nằm trong [0, 1].
* ``ACE  = mean_b |avg(p_b) - avg(y_b)|`` (không trọng số theo số lượng) — báo kèm vì nhạy
  hơn với các ngăn thưa.

Dự đoán được lấy trên TẤT CẢ vị trí có nhãn của tập test, cùng mặt nạ với AUC/ACC
(định nghĩa ``filt``/``labels`` của GRKT và ``_epoch_loop`` của Cấp độ 1 trùng nhau).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np


def brier_score(probs: Sequence[float], labels: Sequence[float]) -> float:
    p = np.asarray(probs, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    if p.size == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probs: Sequence[float],
    labels: Sequence[float],
    n_bins: int = 10,
) -> dict:
    """ECE trọng số theo số lượng + ACE không trọng số + bảng từng ngăn."""
    p = np.asarray(probs, dtype="float64")
    y = np.asarray(labels, dtype="float64")
    if p.size == 0 or p.shape != y.shape:
        raise ValueError(f"probs/labels phải cùng độ dài và khác rỗng: {p.shape} vs {y.shape}")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] = edges[-1] + 1e-9
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    n = p.size
    ece = 0.0
    ace_terms = []
    bins = []
    for b in range(n_bins):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            continue
        avg_p = float(p[m].mean())
        avg_y = float(y[m].mean())
        gap = abs(avg_p - avg_y)
        ece += (cnt / n) * gap
        ace_terms.append(gap)
        bins.append({
            "bin": b,
            "lo": float(edges[b]),
            "hi": float(edges[b + 1]),
            "n": cnt,
            "avg_pred": avg_p,
            "avg_label": avg_y,
            "gap": gap,
        })
    return {
        "ece": float(ece),
        "ace": float(np.mean(ace_terms)) if ace_terms else float("nan"),
        "n_bins_used": len(ace_terms),
        "mean_pred": float(p.mean()),
        "observed_rate": float(y.mean()),
        "reliability": bins,
    }


def calibration_table(
    entries: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    n_bins: int = 10,
    extra: Dict[str, dict] = None,
):
    """Bảng hiệu chỉnh cho ``{tên mô hình: (probs, labels)}``.

    ``extra`` cho phép ghi thêm cột mô tả (ví dụ số cạnh đồ thị của từng biến thể).
    Trả về ``pandas.DataFrame`` đã sắp xếp theo ECE tăng dần.
    """
    import pandas as pd

    extra = extra or {}
    rows = []
    for name, (probs, labels) in entries.items():
        p = np.asarray(probs, dtype="float64")
        y = np.asarray(labels, dtype="float64")
        cal = expected_calibration_error(p, y, n_bins=n_bins)
        row = {
            "model": name,
            "n": int(p.size),
            "brier": brier_score(p, y),
            "ece": cal["ece"],
            "ace": cal["ace"],
            "mean_pred": cal["mean_pred"],
            "observed_rate": cal["observed_rate"],
            "over_prediction": cal["mean_pred"] - cal["observed_rate"],
        }
        row.update(extra.get(name, {}))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("ece").reset_index(drop=True)


def reliability_plot(
    entries: Dict[str, Tuple[Sequence[float], Sequence[float]]],
    out_path,
    n_bins: int = 10,
    title: str = "Đồ thị độ tin cậy (reliability diagram)",
) -> Path:
    """Vẽ đường cong reliability của nhiều mô hình trên cùng trục + khoảng lệch tuyệt đối."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="hoàn hảo")
    for name, (probs, labels) in entries.items():
        cal = expected_calibration_error(probs, labels, n_bins=n_bins)
        xs = [b["avg_pred"] for b in cal["reliability"]]
        ys = [b["avg_label"] for b in cal["reliability"]]
        ns = [b["n"] for b in cal["reliability"]]
        # ``plot`` chỉ nhận ``markersize`` VÔ HƯỚNG (truyền mảng làm Line2D.set_markersize
        # raise "only length-1 arrays can be converted"), nên điểm được vẽ riêng bằng
        # ``scatter`` (``s`` là diện tích, đơn vị point²) với kích thước tỉ lệ số mẫu mỗi ngăn.
        sizes = np.clip(np.sqrt(np.asarray(ns, dtype="float64")) / 12.0, 3.0, 12.0)
        ax.plot(xs, ys, linewidth=1.3,
                label=f"{name} (ECE={cal['ece']:.3f}, n={int(np.sum(ns))})")
        ax.scatter(xs, ys, s=sizes ** 2, zorder=3)
    ax.set_xlabel("xác suất dự đoán trung bình theo ngăn")
    ax.set_ylabel("tỉ lệ làm đúng quan sát được")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def save_reliability_rows(entries: Dict[str, Tuple[Sequence[float], Sequence[float]]], out_path, n_bins: int = 10) -> Path:
    """Xuất ``reliability.csv`` (mỗi dòng = một ngăn của một mô hình) để vẽ lại bằng Excel/Bàn cờ."""
    import pandas as pd

    rows = []
    for name, (probs, labels) in entries.items():
        cal = expected_calibration_error(probs, labels, n_bins=n_bins)
        for b in cal["reliability"]:
            rows.append({"model": name, **b})
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path
