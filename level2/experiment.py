"""Điều phối toàn bộ thực nghiệm Cấp độ 2 cho MỘT tập dữ liệu.

Notebook ``kaggle_level2.ipynb`` chỉ gọi :func:`run_dataset` (và lặp qua danh sách tập dữ
liệu); mọi logic nằm ở đây để tái lập được ngoài Kaggle.

Thứ tự các bước — trùng với các mục in ra màn hình:

1. Chuẩn hoá CSV + nạp bằng pipeline Cấp độ 1 (``level1.pipeline.load_kt_data``) → split
   theo người học, seed 42; lưu ``split.json``.
2. Dựng chuỗi GRKT từ CÙNG file chuẩn hoá và ĐỐI CHIẾU từng người học với bundle Cấp độ 1
   (``level2.grkt_data.build_grkt_data``).
3. Đếm ``rel_map``/``pre_map`` CHỈ từ tập train, sinh các biến thể đồ thị
   (train/dense/zero/shuffled) và in ``leak_report``.
4. Huấn luyện GRKT cho từng biến thể; early stop theo val AUC toàn cục.
5. Đường nền DKT/GKT trên cùng split (nạp checkpoint Cấp độ 1 nếu có).
6. Ghi quỹ đạo mastery m(u,c,t) + hai tiêu chí hợp lý (hướng lan truyền, tính cục bộ) và
   kiểm tra suy giảm theo thời gian.
7. Hiệu chỉnh dự đoán (ECE/Brier) cho mọi mô hình.
8. Đa seed cho GRKT (split cố định, chỉ đổi khởi tạo + thứ tự batch).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .baselines import level1_baseline
from .calibration import calibration_table, reliability_plot, save_reliability_rows
from .dataset_spec import ensure_canonical, get_spec, has_canonical, probe_columns
from .grkt_api import predict_grkt, run_grkt_experiment
from .grkt_data import (
    adjacency_from_graphs,
    build_grkt_data,
    build_relevance_maps,
    graph_variant,
    leak_report,
    split_sequences,
)
from .trajectory import (
    export_mastery_csv,
    plot_mastery_curves,
    plot_mastery_heatmap,
    record_trajectories,
    save_trajectories,
    trajectory_metrics_table,
)

GRAPH_VARIANTS: Sequence[str] = ("train", "dense", "zero", "shuffled")

#: Đường nền so sánh với GRKT — đúng bộ mô hình Cấp độ 1 đã có checkpoint trên ASSISTments.
BASELINE_SPECS: Sequence[tuple] = (("DKT", "N/A"), ("GKT", "Transition"), ("GKT", "Zero"))

#: Tập dữ liệu mà Cấp độ 1 đã train và lưu checkpoint (``outputs/level1/*.pt``). Tập khác
#: phải huấn luyện lại đường nền, nên ``baseline_epochs`` được dùng riêng cho lần train đó.
LEVEL1_CKPT_DATASETS: frozenset = frozenset({"assist2009"})

#: Thứ tự cột của bảng so sánh; khai báo tường minh để bảng rỗng vẫn có cột (notebook đọc
#: ``combined[["model", "auc", ...]]`` mà không cần biết lần chạy có bao nhiêu mô hình).
TABLE_COLUMNS: list = [
    "model", "model_type", "graph", "auc", "acc", "loss",
    "val_auc", "n_test", "edges_rel", "edges_pre", "seed", "seconds",
]


def _device_of(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if hasattr(device, "type") else torch.device(str(device))


def run_dataset(
    dataset: str = "assist2009",
    *,
    raw_path=None,
    save_root=None,
    epochs: int = 8,
    batch_size: int = 32,
    max_seq_len: int = 100,
    lr: float = 1e-3,
    patience: int = 3,
    d_hidden: int = 128,
    k_hidden: int = 16,
    k_hop: int = 1,
    alpha: float = 0.01,
    tau: float = 0.2,
    graph_thresh: float = 0.6,
    bin_thresh: float = 0.5,
    cold_thresh: int = 5,
    gap_cap_min: int = 240,
    max_users: Optional[int] = None,
    min_interactions: Optional[int] = None,
    split_seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    variants: Sequence[str] = GRAPH_VARIANTS,
    run_baselines: bool = True,
    baseline_ckpt_dir=None,
    baseline_epochs: Optional[int] = None,
    allow_retrain_baselines: bool = True,
    record_trajectory: bool = True,
    trajectory_users: int = 150,
    trajectory_batch: int = 16,
    run_calibration: bool = True,
    run_multi_seed: bool = False,
    seeds: Sequence[int] = (42, 123, 456),
    device=None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Chạy toàn bộ pipeline Cấp độ 2 trên một tập dữ liệu; trả về dict chứa bảng + đường dẫn.

    ``raw_path`` chỉ nên truyền khi chạy ĐÚNG MỘT tập dữ liệu (nó áp cho mọi tập trong vòng
    lặp của :func:`run_level2`). Với Junyi trên Kaggle thì không cần: ``ensure_canonical`` tự
    dò trong ``/kaggle/input``. ``max_users``/``min_interactions`` chỉ có tác dụng với tập phải
    chuẩn hoá (Junyi); ASSISTments 2009 dùng thẳng CSV đã commit nên giữ toàn bộ người học —
    muốn chạy nhanh trên tập này thì dùng ``dataset="assist2009_smoke"``.

    ``baseline_epochs`` là số epoch khi phải HUẤN LUYỆN LẠI đường nền DKT/GKT (tập chưa có
    checkpoint Cấp độ 1). Mặc định ``None`` = dùng chung ``epochs`` với GRKT; nên đặt 20 theo
    đúng công thức Cấp độ 1 để đường nền không yếu một cách nhân tạo.
    """
    spec = get_spec(dataset)
    device = _device_of(device)
    save_dir = Path(save_root if save_root is not None else Path(spec.canonical_file).parent) / spec.name
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {"dataset": spec.name, "save_dir": str(save_dir), "artifacts": {}}

    # ------------------------------------------------------------ 1. chuẩn hoá + split
    print("=" * 72)
    print(f">>> [C2-B1] Chuẩn hoá dữ liệu và chia split theo người học: {spec.name}")
    print("=" * 72)
    # Chỉ cảnh báo khi CSV chuẩn hoá ĐÃ có sẵn (repo hoặc lần chạy trước): lúc đó file được
    # dùng lại nguyên trạng nên max_users/min_interactions không làm thay đổi tập người học.
    if has_canonical(spec, save_dir) and (max_users is not None or min_interactions is not None):
        print(f"  [cảnh báo] {spec.name}: dùng lại CSV chuẩn hoá có sẵn nên max_users/"
              f"min_interactions KHÔNG được áp dụng (chỉ có tác dụng khi chuẩn hoá lại từ file gốc).")
    canonical = ensure_canonical(spec, out_dir=save_dir, raw_path=raw_path,
                                 max_users=max_users, min_interactions=min_interactions, verbose=verbose)
    min_seq = int(min_interactions if min_interactions is not None else spec.min_interactions)
    print(f"  CSV chuẩn hoá: {canonical}")
    print(f"  Cột: {probe_columns(canonical)}")

    from level1.pipeline import load_kt_data, save_split

    shared = load_kt_data(
        str(canonical),
        batch_size=batch_size,
        graph_type="Transition",
        model_type="GKT",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq,
        use_cuda=device.type == "cuda",
    )
    split = save_split(shared, save_dir / "split.json")
    n_tr, n_va, n_te = (len(split[f"{k}_users"]) for k in ("train", "val", "test"))
    print(f"  concept_num={shared['concept_num']} | split user train/val/test = {n_tr}/{n_va}/{n_te} (seed={split_seed})")
    out["split"] = split
    out["concept_num"] = int(shared["concept_num"])
    out["artifacts"]["split"] = str(save_dir / "split.json")

    # ------------------------------------------------------------ 2. chuỗi GRKT
    print(">>> [C2-B2] Dựng chuỗi GRKT từ cùng file chuẩn hoá và đối chiếu với Cấp độ 1 ...")
    data = build_grkt_data(
        shared,
        canonical,
        time_mode=spec.time_mode,
        gap_cap_min=gap_cap_min,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq,
        max_concepts=spec.max_concepts,
        dataset=spec.name,
        verbose=verbose,
    )
    splits = split_sequences(data, split)
    out["grkt_data"] = data
    out["splits"] = splits

    # ------------------------------------------------------------ 3. đồ thị từ train
    print(">>> [C2-B3] Đếm rel_map/pre_map CHỈ từ tập train + sinh biến thể đồ thị ...")
    maps_train = build_relevance_maps(splits["train"]["sequences"], data.concept_num, cold_thresh=cold_thresh)
    maps_val = build_relevance_maps(splits["val"]["sequences"], data.concept_num, cold_thresh=cold_thresh)
    graphs = {kind: graph_variant(maps_train, kind, data.concept_num, thresh=graph_thresh, seed=split_seed)
              for kind in variants}
    # adjacency phục vụ tiêu chí quỹ đạo LU LU là đồ thị train thật (biến thể "train"),
    # kể cả khi lần chạy này không huấn luyện biến thể đó (vd VARIANTS=("zero",) smoke test).
    train_graph = graphs.get("train") or graph_variant(
        maps_train, "train", data.concept_num, thresh=graph_thresh, seed=split_seed)
    adj = adjacency_from_graphs(train_graph["rel_map"], train_graph["pre_map"])
    report = leak_report(data, splits, shared, maps_train, maps_val, graphs,
                         train_ratio=train_ratio, val_ratio=val_ratio, seed=split_seed)
    print(report)
    (save_dir / "leak_report.txt").write_text(report + "\n", encoding="utf-8")
    out["maps"] = maps_train
    out["graphs"] = graphs
    out["adj"] = adj
    out["artifacts"]["leak_report"] = str(save_dir / "leak_report.txt")
    np.savez_compressed(
        save_dir / "graphs_train_only.npz",
        rel=maps_train["rel"], pre=maps_train["pre"],
        rel_bin=graphs["train"]["rel_map"], pre_bin=graphs["train"]["pre_map"],
    )
    out["artifacts"]["graphs"] = str(save_dir / "graphs_train_only.npz")

    # ------------------------------------------------------------ 4. GRKT
    print(">>> [C2-B4] Huấn luyện GRKT cho từng biến thể đồ thị ...")
    grkt_results: List[dict] = []
    for kind in variants:
        print(f"\n--- GRKT với đồ thị '{kind}' ---")
        res = run_grkt_experiment(
            data,
            splits,
            graphs[kind],
            graph_kind=kind,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            d_hidden=d_hidden,
            k_hidden=k_hidden,
            k_hop=k_hop,
            alpha=alpha,
            tau=tau,
            thresh=bin_thresh,
            seed=split_seed,
            dataset=spec.name,
            device=device,
            save_dir=save_dir,
            verbose=verbose,
        )
        grkt_results.append(res)
    out["grkt_results"] = grkt_results

    # ------------------------------------------------------------ 5. đường nền Cấp độ 1
    base_results: List[dict] = []
    base_probs: Dict[str, tuple] = {}
    if run_baselines:
        print("\n>>> [C2-B5] Đánh giá đường nền DKT / GKT trên CÙNG split ...")
        # Checkpoint của Cấp độ 1 chỉ tồn tại cho ASSISTments 2009. Với tập khác (Junyi) phải
        # huấn luyện lại, và số epoch lấy theo công thức Cấp độ 1 (baseline_epochs) để đường
        # nền không bị yếu một cách nhân tạo khi đặt cạnh GRKT.
        ckpt_dir = baseline_ckpt_dir if spec.name in LEVEL1_CKPT_DATASETS else None
        if baseline_ckpt_dir and ckpt_dir is None and verbose:
            print(f"  [baseline] {spec.name} không có checkpoint Cấp độ 1 -> huấn luyện lại "
                  f"{baseline_epochs or epochs} epoch (trọng số ghi vào {save_dir}/baselines)")
        got = level1_baseline(
            shared,
            specs=BASELINE_SPECS,
            ckpt_dir=ckpt_dir,
            epochs=int(baseline_epochs or epochs),
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            lr=lr,
            seed=split_seed,
            device=device,
            allow_retrain=allow_retrain_baselines,
            collect_probs=True,
            shared_graph_type="Transition",
            save_dir=save_dir,
            verbose=verbose,
        )
        base_results, base_probs = got["results"], got["probs"]
    out["baseline_results"] = base_results

    # ------------------------------------------------------------ bảng so sánh
    table = pd.DataFrame([_table_row(r) for r in base_results + grkt_results], columns=TABLE_COLUMNS)
    table = table.sort_values("auc", ascending=False)
    table.to_csv(save_dir / "level2_metrics.csv", index=False)
    out["table"] = table
    out["artifacts"]["metrics"] = str(save_dir / "level2_metrics.csv")
    print("\n=== BẢNG SO SÁNH DKT / GKT / GRKT (cùng split, cùng chỉ số) ===")
    print(table.to_string(index=False))

    # ------------------------------------------------------------ 6. quỹ đạo mastery
    traj_runs: Dict[str, dict] = {}
    traj_table = None
    if record_trajectory:
        print("\n>>> [C2-B6] Ghi quỹ đạo mastery m(u,c,t) và đo tiêu chí hợp lý ...")
        plot_dir = save_dir / "trajectories"
        plot_dir.mkdir(parents=True, exist_ok=True)
        for res in grkt_results:
            kind = res["graph_type"]
            model = res["model"].to(device)
            traj = record_trajectories(
                model,
                splits["test"]["sequences"][:trajectory_users],
                device=device,
                batch_size=trajectory_batch,
                concept_num=data.concept_num,
                graph_kind=kind,
                seed=split_seed,
                verbose=False,
            )
            traj_runs[kind] = traj
            save_trajectories(traj, plot_dir / f"traj_{kind}.npz")
            export_mastery_csv(traj, plot_dir / f"mastery_{kind}.csv", max_users=25,
                               skill_lookup=data.skill_lookup)
            plot_mastery_heatmap(traj, plot_dir / f"heatmap_{kind}.png", user_index=0,
                                 skill_lookup=data.skill_lookup)
            plot_mastery_curves(traj, plot_dir / f"curves_{kind}.png", user_indices=(0, 1),
                                skill_lookup=data.skill_lookup)
            model.cpu()
        traj_table = trajectory_metrics_table(traj_runs, adj=adj)
        traj_table.to_csv(save_dir / "trajectory_metrics.csv", index=False)
        out["trajectories"] = traj_runs
        out["trajectory_table"] = traj_table
        out["artifacts"]["trajectory_metrics"] = str(save_dir / "trajectory_metrics.csv")
        out["artifacts"]["trajectory_plots"] = str(plot_dir)
        print("\n=== TIÊU CHÍ HỢP LÝ CỦA QUỸ ĐẠO (in trên cùng tập 'láng giềng' của đồ thị train) ===")
        print(traj_table.to_string(index=False))

    # ------------------------------------------------------------ 7. calibration
    cal = pd.DataFrame()
    entries: Dict[str, tuple] = dict(base_probs)
    if run_calibration:
        print("\n>>> [C2-B7] Hiệu chỉnh dự đoán (ECE / Brier) ...")
        for res in grkt_results:
            model = res["model"].to(device)
            p, y = predict_grkt(model, splits["test"]["items"], device=device)
            entries[res["name"]] = (p, y)
            model.cpu()
        cal = calibration_table(entries, n_bins=10,
                                extra={r["name"]: {"graph": r["graph_type"]} for r in grkt_results})
        cal.to_csv(save_dir / "calibration.csv", index=False)
        reliability_plot(entries, save_dir / "reliability.png")
        save_reliability_rows(entries, save_dir / "reliability_rows.csv")
        out["artifacts"]["calibration"] = str(save_dir / "calibration.csv")
        out["artifacts"]["reliability"] = str(save_dir / "reliability.png")
        print(cal.to_string(index=False))
    out["calibration"] = cal
    out["calibration_probs"] = entries

    # ------------------------------------------------------------ 8. đa seed
    multi = None
    if run_multi_seed:
        ms_kind = "train" if "train" in graphs else variants[0]
        print("\n>>> [C2-B8] Đa seed cho GRKT-%s (split cố định seed=%d) ..." % (ms_kind, split_seed))
        rows = []
        for s in seeds:
            res = run_grkt_experiment(
                data, splits, graphs[ms_kind],
                graph_kind=ms_kind, epochs=epochs, batch_size=batch_size, lr=lr, patience=patience,
                d_hidden=d_hidden, k_hidden=k_hidden, k_hop=k_hop, alpha=alpha, tau=tau,
                thresh=bin_thresh, seed=int(s), dataset=spec.name, device=device,
                save_dir=None, verbose=verbose,
            )
            rows.append({"model": res["name"], "seed": int(s), "test_auc": res["test_auc"],
                         "test_acc": res["test_acc"], "test_loss": res["test_loss"]})
            del res
        ms = pd.DataFrame(rows)
        agg = ms.groupby("model").agg(
            auc_mean=("test_auc", "mean"), auc_std=("test_auc", "std"),
            acc_mean=("test_acc", "mean"), acc_std=("test_acc", "std"),
            loss_mean=("test_loss", "mean"), loss_std=("test_loss", "std"), n_seed=("seed", "count"),
        ).reset_index()
        # ddof=1 với n=3 cho std mẫu; ghi cả số seed để người đọc không nhầm với tổng thể
        agg.to_csv(save_dir / "multiseed_metrics.csv", index=False)
        out["multiseed_raw"] = ms
        out["multiseed"] = agg
        out["artifacts"]["multiseed"] = str(save_dir / "multiseed_metrics.csv")
        print(agg.to_string(index=False))

    summary = {
        "dataset": spec.name,
        "canonical": str(canonical),
        "time_mode": data.time_mode,
        "data_stats": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                       for k, v in data.stats.items()},
        "graph_edges": {k: {"rel": int(v["edges_rel"]), "pre": int(v["edges_pre"])} for k, v in graphs.items()},
        "counts": {"train_edges_from_train": int((maps_train["rel"] > 0).sum() + (maps_train["pre"] > 0).sum()),
                   "train_edges_from_val": int((maps_val["rel"] > 0).sum() + (maps_val["pre"] > 0).sum())},
        "models": [_table_row(r) for r in base_results + grkt_results],
        "trajectory": None if traj_table is None else traj_table.to_dict(orient="records"),
        "calibration": cal.to_dict(orient="records"),
    }
    (save_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str),
                                           encoding="utf-8")
    out["artifacts"]["summary"] = str(save_dir / "summary.json")
    out["spec"] = spec
    return out


def _table_row(r: dict) -> dict:
    return {
        "model": r["name"],
        "model_type": r["model_type"],
        "graph": r["graph_type"],
        "auc": round(float(r["test_auc"]), 4),
        "acc": round(float(r["test_acc"]), 4),
        "loss": round(float(r["test_loss"]), 4),
        "val_auc": (None if r.get("best_val_auc") is None or np.isnan(r.get("best_val_auc", float("nan")))
                    else round(float(r["best_val_auc"]), 4)),
        "n_test": int(r.get("test_n", 0)),
        "edges_rel": int(r.get("edges_rel", 0)),
        "edges_pre": int(r.get("edges_pre", 0)),
        "seed": int(r.get("seed", 42)),
        "seconds": round(float(r.get("train_seconds", 0.0)), 1),
    }


def run_level2(
    datasets: Sequence[str] = ("assist2009", "junyi"),
    *,
    save_root=None,
    **kwargs,
) -> Dict[str, object]:
    """Lặp :func:`run_dataset` qua nhiều tập dữ liệu và nối bảng (yêu cầu ≥ 2 dataset)."""
    runs: Dict[str, dict] = {}
    frames = []
    for name in datasets:
        print("\n" + "#" * 72)
        print(f"# TẬP DỮ LIỆU: {name}")
        print("#" * 72)
        run = run_dataset(name, save_root=save_root, **kwargs)
        runs[name] = run
        t = run["table"].copy()
        t.insert(0, "dataset", name)
        frames.append(t)
    combined = pd.concat(frames, ignore_index=True)
    if save_root is not None:
        p = Path(save_root) / "level2_all_datasets.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(p, index=False)
    return {"runs": runs, "combined": combined}
