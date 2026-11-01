"""Điểm chạy thử nghiệm thu của bản port GRKT (SỬA so với bản gốc).

Bản gốc ``from models import *`` không chạy được trong repo này vì mô hình nằm ở
``GRKT/model/GRKT.py``; nếu ``GKT/`` (Cấp độ 1) cũng nằm trên ``sys.path`` thì câu lệnh
đó nạp nhầm ``GKT/models.py`` (DKT/GKT, không có GRKT).

CẢNH BÁO giao thức: đường dẫn CLI này dùng ``Runner`` k-fold của upstream, tức cắt
chuỗi mỗi người học thành các cửa sổ 100 tương tác — cùng một người học xuất hiện ở
cả train lẫn test và ``rel_map``/``pre_map`` trong ``data/*.bin`` được dựng từ toàn bộ
dữ liệu. Vì vậy số liệu từ đây KHÔNG đủ điều kiện nộp cho Cấp độ 2. Giao thức chính
thức (split theo người học dùng lại của Cấp độ 1, đồ thị chỉ từ tập train) là
``level2/`` + ``kaggle_level2.ipynb``.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GRKT.model.GRKT import GRKT  # noqa: E402
from GRKT.Runner import Runner  # noqa: E402
import argparse  # noqa: E402
import pickle  # noqa: E402

# Đăng ký tường minh thay cho eval(tên) — tránh thực thi code từ chuỗi đầu vào CLI
MODEL_REGISTRY = {"GRKT": GRKT}
ALLOWED_DATASETS = {"assist09"}


def training(argv=None):

    print("Training start...")
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model", default="GRKT", type=str, help="Model used.")
    parser.add_argument("--dataset", default="assist09", type=str, help="Data used.")

    if argv:
        sys.argv = argv

    args, _ = parser.parse_known_args()
    if args.model not in MODEL_REGISTRY:
        raise SystemExit(
            f"Mô hình không được đăng ký: {args.model}. Cho phép: {sorted(MODEL_REGISTRY)}"
        )
    if args.dataset not in ALLOWED_DATASETS:
        raise SystemExit(
            f"Tập dữ liệu không được cho phép: {args.dataset}. Cho phép: {sorted(ALLOWED_DATASETS)}"
        )
    model_class = MODEL_REGISTRY[args.model]
    bin_path = Path("data") / args.dataset / f"{args.dataset}.bin"
    with bin_path.open("rb") as f:
        dataset = pickle.load(f)

    Runner.parse_args(parser)
    model_class.parse_args(parser)

    args, _ = parser.parse_known_args()

    for key in dataset:
        if key != "data":
            args.__setattr__(key, dataset[key])

    runner = Runner(args)
    result = runner.run(model_class, dataset["data"])
    return result


if __name__ == "__main__":
    training()
