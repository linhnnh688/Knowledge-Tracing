"""CLI Cấp độ 1: ``python -m level1 [--quick] [--epochs N] [--seeds 42 123 ...]``.

Chạy từ thư mục gốc repo. Notebook Kaggle vẫn là đường chạy chính cho báo cáo (có thêm
ShuffledRow, dashboard, case study); CLI này để tái lập nhanh bảng DKT/GKT ngoài notebook.
"""

from __future__ import annotations

import argparse

from .pipeline import DEFAULT_GRAPH_TYPES, ExperimentConfig, run_level1


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m level1", description=__doc__)
    p.add_argument("--data-file", default="skill_builder_data.csv")
    p.add_argument("--quick", action="store_true", help="smoke test: 15 người học, 3 epoch")
    p.add_argument("--graph-types", nargs="+", default=list(DEFAULT_GRAPH_TYPES))
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--save-dir", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-seq-len", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    args = p.parse_args(argv)

    cfg = ExperimentConfig()
    data_file = args.data_file
    if args.quick:
        data_file = "assistment_test15.csv"
        cfg = cfg.replace(epochs=3, batch_size=16, max_seq_len=80)
    overrides = {
        k: v
        for k, v in {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_seq_len": args.max_seq_len,
            "lr": args.lr,
        }.items()
        if v is not None
    }
    run_level1(
        data_file,
        graph_types=args.graph_types,
        seeds=args.seeds,
        save_dir=args.save_dir,
        config=cfg.replace(**overrides),
    )


if __name__ == "__main__":
    main()
