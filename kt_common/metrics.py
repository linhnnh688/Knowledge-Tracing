"""Chỉ số nhị phân TOÀN CỤC dùng chung cho mọi mô hình (DKT, GKT, GRKT).

Một nơi duy nhất để bảng Cấp độ 1 và Cấp độ 2 chắc chắn so sánh được: AUC/ACC tính trên toàn
bộ vị trí có nhãn của loader (không lấy trung bình AUC từng batch), ngưỡng ACC = 0.5, xác suất
kẹp ``[PROB_EPS, 1 - PROB_EPS]`` khi tính log-loss (giống ``GKT/metrics.KTLoss``).
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

#: Kẹp xác suất trước khi lấy log — trùng với ``GKT.metrics.KTLoss``.
PROB_EPS = 1e-6


def auc_acc(y_true, y_prob) -> Tuple[float, float]:
    """``(auc, acc)``; ``(-1, -1)`` khi rỗng hoặc chỉ có một lớp (AUC không xác định)."""
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return -1.0, -1.0
    auc = float(roc_auc_score(y_true, y_prob))
    acc = float(((y_prob >= 0.5) == (y_true >= 0.5)).mean())
    return auc, acc
