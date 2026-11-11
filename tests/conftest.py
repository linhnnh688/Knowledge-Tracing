import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SMOKE_CSV = ROOT / "GKT" / "data" / "assistment_test15.csv"
FULL_CSV = ROOT / "GKT" / "data" / "skill_builder_data.csv"


def is_lfs_pointer(path: Path) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(40).startswith(b"version https://git-lfs")
    except OSError:
        return True


@pytest.fixture(scope="session")
def cpu():
    import torch

    torch.set_num_threads(2)
    return torch.device("cpu")


@pytest.fixture(scope="session")
def smoke_bundle():
    from level1.pipeline import load_kt_data

    return load_kt_data(
        str(SMOKE_CSV), batch_size=4, graph_type="Transition", seed=42, max_seq_len=80, use_cuda=False
    )


@pytest.fixture(scope="session")
def full_csv():
    if not FULL_CSV.exists() or is_lfs_pointer(FULL_CSV):
        pytest.skip("skill_builder_data.csv chưa có (git lfs pull)")
    return FULL_CSV
