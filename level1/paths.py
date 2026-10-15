import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_gkt_on_path() -> Path:
    gkt = repo_root() / "GKT"
    gkt_str = str(gkt)
    if gkt_str not in sys.path:
        sys.path.insert(0, gkt_str)
    return gkt
