"""Logger cho package: mặc định in INFO ra ``sys.stdout`` với định dạng trần (giống ``print``).

Vì sao không dùng ``logging.basicConfig``: mức mặc định của logging là WARNING nên mọi dòng
INFO sẽ biến mất khỏi notebook nếu người dùng không cấu hình gì. Handler ở đây tra
``sys.stdout`` tại THỜI ĐIỂM GHI (không giữ tham chiếu lúc import), nên Jupyter/ipykernel
đổi ``sys.stdout`` theo cell vẫn hiện đúng chỗ.

Tắt bớt log: ``logging.getLogger("kt").setLevel(logging.WARNING)``.
"""

from __future__ import annotations

import logging
import sys

ROOT = "kt"


class _StdoutHandler(logging.StreamHandler):
    def emit(self, record: logging.LogRecord) -> None:
        self.stream = sys.stdout
        super().emit(record)


def _root() -> logging.Logger:
    root = logging.getLogger(ROOT)
    if not any(isinstance(h, _StdoutHandler) for h in root.handlers):
        handler = _StdoutHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
    return root


def get_logger(name: str) -> logging.Logger:
    """Logger con của ``kt`` (vd ``get_logger("level1.pipeline")`` -> ``kt.level1.pipeline``)."""
    _root()
    return logging.getLogger(f"{ROOT}.{name}")
