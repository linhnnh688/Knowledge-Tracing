import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> Path:
    """Đưa thư mục gốc repo vào ``sys.path`` để ``import GKT``/``GRKT``/``kt_common`` chạy được."""
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return repo_root()


def ensure_gkt_on_path() -> Path:
    """Giữ lại cho code/notebook cũ; KHÔNG còn chèn ``GKT/`` vào ``sys.path``.

    ``GKT/`` giờ là package dùng import tương đối, nên import đúng là
    ``from GKT.metrics import KTLoss`` (không phải ``from metrics import KTLoss``). Chèn
    ``GKT/`` vào ``sys.path`` như trước sẽ biến ``models``/``utils``/``metrics`` thành tên
    module toàn cục, dễ va chạm với package khác.
    """
    ensure_repo_on_path()
    return repo_root() / "GKT"
