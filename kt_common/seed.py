"""Đặt seed cho mọi nguồn ngẫu nhiên và (tuỳ chọn) bật thuật toán tất định của PyTorch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python/NumPy/PyTorch (+CUDA).

    ``deterministic=True`` còn:

    * đặt ``PYTHONHASHSEED`` (chỉ có tác dụng với tiến trình con, ví dụ DataLoader worker —
      tiến trình hiện tại đã băm chuỗi với seed lúc khởi động);
    * đặt ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` nếu chưa có (cuBLAS cần biến này để tất định;
      chỉ có tác dụng nếu đặt TRƯỚC lần gọi CUDA đầu tiên);
    * ``torch.use_deterministic_algorithms(True, warn_only=True)``: op không có bản tất định
      (vd ``index_put`` cộng dồn trên CUDA) chỉ phát cảnh báo thay vì làm chết lần chạy.

    Trên GPU, số liệu có thể lệch nhẹ (cỡ 1e-4..1e-3 AUC) so với các lần chạy cũ không bật
    chế độ này, và train có thể chậm hơn một chút.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
