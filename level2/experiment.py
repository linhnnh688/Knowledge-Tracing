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
6. Ghi quỹ đạo mastery m(u,c,t) + tiêu chí THẬT (``precedence_gap``, ``stability_*``,
   ``forget_slope``) + kiểm chứng cài đặt (``consistency_*``/``locality`` bị cấu trúc mô hình
   ép ≡ 1, không dùng làm bằng chứng).
7. Hiệu chỉnh dự đoán (ECE/Brier) cho mọi mô hình.
8. Đa seed cho GRKT (split cố định, chỉ đổi khởi tạo + thứ tự batch).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .baselines import level1_baseline
from .calibration import calibration_table, reliability_plot, save_reliability_rows
from .dataset_spec import ensure_canonical, get_spec, has_canonical, probe_columns
from .grkt_api import TrainingPaused, predict_grkt, run_grkt_experiment
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
    stability_under_perturbation,
    trajectory_metrics_table,
)

#: Bốn biến thể đồ thị GRKT hỗ trợ. Ablation cấu trúc đồ thị là yêu cầu Cấp độ 1 (đã có số
#: liệu trong outputs/level1), nên mặc định của run_dataset chỉ train ``("train",)`` — mỗi biến
#: thể là một lần train GRKT riêng, bật đủ bốn biến thể tốn gấp ~4 thời lượng GPU.
GRAPH_VARIANTS: Sequence[str] = ("train", "dense", "zero", "shuffled")

#: Đường nền so sánh với GRKT — đúng bộ mô hình Cấp độ 1 đã có checkpoint trên ASSISTments.
BASELINE_SPECS: Sequence[tuple] = (
    ("DKT", "N/A"),
    ("GKT", "Transition"),
    ("GKT", "Zero"),
)


def grkt_activation_gib(
    concept_num: int,
    batch_size: int,
    max_seq_len: int,
    k_hidden: int,
    n_step_tensors: int = 40,
) -> float:
    """Gibibyte tensor kích hoạt GRKT phải GIỮ cho một batch (ước lượng bậc độ lớn).

    Mỗi bước thời gian sinh khoảng ``n_step_tensors`` tensor có dạng ``[B, K+1, k_hidden]``
    (retrieval, gain, loss, learn — mỗi giai đoạn ba loại cạnh rel/pre/sub), và autograd giữ
    chúng cho tới khi backprop qua toàn bộ ``max_seq_len`` bước. Con số này tỉ lệ thuận với số
    khái niệm, nên nó là lý do ``top_concepts`` của Junyi tồn tại: log có 1.326 bài tập
    (``ucid``; nếu lấy nhầm ``upid`` làm khái niệm thì K = 25.785 câu hỏi), còn đồ thị dense
    (K+1)² và trạng thái ``[B, K+1, k_hidden]`` chỉ vừa VRAM 16 GB của T4 khi K ≈ 200.
    """
    nbytes = (
        n_step_tensors
        * max(1, int(max_seq_len))
        * int(batch_size)
        * (int(concept_num) + 1)
        * int(k_hidden)
        * 4
    )
    return nbytes / 1024**3


#: Tập dữ liệu mà Cấp độ 1 đã train và lưu checkpoint (``outputs/level1/*.pt``). Tập khác
#: phải huấn luyện lại đường nền, nên ``baseline_epochs`` được dùng riêng cho lần train đó.
LEVEL1_CKPT_DATASETS: frozenset = frozenset({"assist2009"})

#: Thứ tự cột của bảng so sánh; khai báo tường minh để bảng rỗng vẫn có cột (notebook đọc
#: ``combined[["model", "auc", ...]]`` mà không cần biết lần chạy có bao nhiêu mô hình).
TABLE_COLUMNS: list = [
    "model",
    "model_type",
    "graph",
    "auc",
    "acc",
    "loss",
    "val_auc",
    "n_test",
    "edges_rel",
    "edges_pre",
    "seed",
    "checkpoint_seed",
    "seconds",
]


def _device_of(device):
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device if hasattr(device, "type") else torch.device(str(device))


def _safe_plot(fn, *args, **kwargs):
    """Vẽ hình là bước TRANG TRÍ; số liệu (CSV/npz) đã ghi xong trước nó.

    Một lần ValueError/TypeError trong matplotlib ở đây từng làm chết cả cell thực nghiệm sau
    hàng giờ train GRKT trên Kaggle (kết quả không còn trong biến ``result``). Vì vậy lỗi vẽ
    chỉ in cảnh báo chứ không ném lên :func:`run_dataset`.
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:
        print(
            f"[cảnh báo] {getattr(fn, '__name__', fn)} thất bại: {type(exc).__name__}: {exc} "
            "— số liệu vẫn còn, chỉ thiếu hình vẽ."
        )
        return None


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
    top_concepts: Optional[int] = None,
    content_path=None,
    split_seed: int = 42,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    variants: Sequence[str] = ("train",),
    run_baselines: bool = True,
    baseline_ckpt_dir=None,
    baseline_epochs: Optional[int] = None,
    allow_retrain_baselines: bool = True,
    record_trajectory: bool = True,
    trajectory_users: int = 150,
    trajectory_batch: int = 16,
    run_calibration: bool = False,
    run_multi_seed: bool = False,
    seeds: Sequence[int] = (42, 123, 456),
    resume_grkt_ckpt: bool = False,
    time_budget_min: Optional[float] = None,
    deadline: Optional[float] = None,
    ckpt_every_min: float = 10.0,
    device=None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Chạy toàn bộ pipeline Cấp độ 2 trên một tập dữ liệu; trả về dict chứa bảng + đường dẫn.

    Mặc định chỉ làm đúng những gì yêu cầu Cấp độ 2 nêu: train GRKT trên đồ thị ``train``,
    đánh giá trên cùng split với đường nền Cấp độ 1, và ghi quỹ đạo m(u,c,t) kèm tiêu chí hợp
    lý. Hai thứ ngoài phạm vi đó phải bật thủ công — ``variants=GRAPH_VARIANTS`` cho ablation
    cấu trúc đồ thị (mỗi biến thể thêm một lần train) và ``run_calibration`` cho ECE/Brier (đề
    bài chỉ cần MỘT phân tích nâng cao). ``run_multi_seed`` bật đa seed cho GRKT; notebook báo
    cáo chỉ bật nó trên tập rẻ (assist2009) và ghi rõ tập còn lại chạy 1 seed.

    ``raw_path`` chỉ nên truyền khi chạy ĐÚNG MỘT tập dữ liệu (nó áp cho mọi tập trong vòng
    lặp của :func:`run_level2`). Với Junyi trên Kaggle thì không cần: ``ensure_canonical`` tự
    dò trong ``/kaggle/input``. ``max_users``/``min_interactions`` chỉ có tác dụng với tập phải
    chuẩn hoá (Junyi); ASSISTments 2009 dùng thẳng CSV đã commit nên giữ toàn bộ người học —
    muốn chạy nhanh trên tập này thì dùng ``dataset="assist2009_smoke"``.

    ``baseline_epochs`` là số epoch khi phải HUẤN LUYỆN LẠI đường nền DKT/GKT (tập chưa có
    checkpoint Cấp độ 1). Mặc định ``None`` = dùng chung ``epochs`` với GRKT; nên đặt 20 theo
    đúng công thức Cấp độ 1 để đường nền không yếu một cách nhân tạo.

    Resume qua nhiều phiên Kaggle: ``resume_grkt_ckpt=True`` nạp lại ``GRKT-<graph>.pt`` (đã
    train xong -> bỏ qua train) hoặc ``ckpt_GRKT-<graph>.pt`` (dừng giữa epoch -> train tiếp
    đúng batch còn lại). ``time_budget_min``/``deadline`` là ngân sách thời gian: khi hết,
    ``train_grkt`` ghi checkpoint rồi raise :class:`~level2.grkt_api.TrainingPaused` để phiên
    Kaggle không bị kill giữa lúc ghi. ``ckpt_every_min`` là nhịp ghi checkpoint giữa epoch.
    """
    spec = get_spec(dataset)
    device = _device_of(device)
    save_dir = (
        Path(save_root if save_root is not None else Path(spec.canonical_file).parent)
        / spec.name
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    out = {"dataset": spec.name, "save_dir": str(save_dir), "artifacts": {}}
    # ``run_level2`` truyền xuống một ``deadline`` tuyệt đối chung cho mọi tập; gọi trực tiếp
    # ``run_dataset`` thì ``time_budget_min`` được quy ra mốc tuyệt đối ngay tại đây.
    if deadline is None and time_budget_min:
        deadline = time.time() + float(time_budget_min) * 60.0

    # ------------------------------------------------------------ 1. chuẩn hoá + split
    print("=" * 72)
    print(f">>> [C2-B1] Chuẩn hoá dữ liệu và chia split theo người học: {spec.name}")
    print("=" * 72)
    # Chỉ cảnh báo khi CSV chuẩn hoá ĐÃ có sẵn (repo hoặc lần chạy trước): lúc đó file được
    # dùng lại nguyên trạng nên max_users/min_interactions/top_concepts không làm thay đổi
    # tập người học/khái niệm. Trước lần chuẩn hoá đầu tiên chúng VẪN có tác dụng.
    if has_canonical(spec, save_dir) and (
        max_users is not None
        or min_interactions is not None
        or top_concepts is not None
    ):
        print(
            f"  [cảnh báo] {spec.name}: dùng lại CSV chuẩn hoá có sẵn nên max_users/"
            f"min_interactions/top_concepts KHÔNG được áp dụng (chỉ có tác dụng khi chuẩn hoá "
            f"lại từ file gốc; muốn chạy nhanh: dataset='assist2009_smoke')."
        )
    canonical = ensure_canonical(
        spec,
        out_dir=save_dir,
        raw_path=raw_path,
        max_users=max_users,
        min_interactions=min_interactions,
        top_concepts=top_concepts,
        content_path=content_path,
        verbose=verbose,
    )
    min_seq = int(
        min_interactions if min_interactions is not None else spec.min_interactions
    )
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
    print(
        f"  concept_num={shared['concept_num']} | split user train/val/test = {n_tr}/{n_va}/{n_te} (seed={split_seed})"
    )
    out["split"] = split
    out["concept_num"] = int(shared["concept_num"])
    out["artifacts"]["split"] = str(save_dir / "split.json")

    # ------------------------------------------------------------ 2. chuỗi GRKT
    print(
        ">>> [C2-B2] Dựng chuỗi GRKT từ cùng file chuẩn hoá và đối chiếu với Cấp độ 1 ..."
    )
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
    maps_train = build_relevance_maps(
        splits["train"]["sequences"], data.concept_num, cold_thresh=cold_thresh
    )
    maps_val = build_relevance_maps(
        splits["val"]["sequences"], data.concept_num, cold_thresh=cold_thresh
    )
    graphs = {
        kind: graph_variant(
            maps_train, kind, data.concept_num, thresh=graph_thresh, seed=split_seed
        )
        for kind in variants
    }
    # adjacency phục vụ tiêu chí quỹ đạo LU LU là đồ thị train thật (biến thể "train"),
    # kể cả khi lần chạy này không huấn luyện biến thể đó (vd VARIANTS=("zero",) smoke test).
    train_graph = graphs.get("train") or graph_variant(
        maps_train, "train", data.concept_num, thresh=graph_thresh, seed=split_seed
    )
    adj = adjacency_from_graphs(train_graph["rel_map"], train_graph["pre_map"])
    if int(train_graph["edges_rel"]) + int(train_graph["edges_pre"]) == 0:
        # Xảy ra với tập quá nhỏ (smoke) hoặc cold_thresh/graph_thresh quá cao: GRKT-train khi
        # đó chạy y hệt GRKT-zero, mọi kết luận về "tác động của đồ thị" đều vô nghĩa.
        print(
            f"  [cảnh báo] đồ thị train RỖNG (rel={train_graph['edges_rel']} pre={train_graph['edges_pre']}) "
            f"với cold_thresh={cold_thresh}, graph_thresh={graph_thresh} trên "
            f"{len(splits['train']['sequences'])} người học. Ablation sẽ không phân biệt được; "
            "hạ cold_thresh/graph_thresh hoặc chạy ở chế độ đầy đủ."
        )
    report = leak_report(
        data,
        splits,
        shared,
        maps_train,
        maps_val,
        graphs,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=split_seed,
    )
    print(report)
    (save_dir / "leak_report.txt").write_text(report + "\n", encoding="utf-8")
    out["maps"] = maps_train
    out["graphs"] = graphs
    out["adj"] = adj
    out["artifacts"]["leak_report"] = str(save_dir / "leak_report.txt")
    np.savez_compressed(
        save_dir / "graphs_train_only.npz",
        rel=maps_train["rel"],
        pre=maps_train["pre"],
        rel_bin=train_graph["rel_map"],
        pre_bin=train_graph["pre_map"],
    )
    out["artifacts"]["graphs"] = str(save_dir / "graphs_train_only.npz")

    # ------------------------------------------------------------ 4. GRKT
    print(">>> [C2-B4] Huấn luyện GRKT cho từng biến thể đồ thị ...")
    gib = grkt_activation_gib(data.concept_num, batch_size, max_seq_len, k_hidden)
    print(
        f"  [bộ nhớ] GRKT giữ ước lượng {gib:.1f} GiB tensor kích hoạt cho MỘT batch "
        f"(K={data.concept_num}, B={batch_size}, S={max_seq_len}, k_hidden={k_hidden})"
    )
    if gib > 8.0:
        print(
            f"  [cảnh báo] {gib:.1f} GiB là sát/near vượt VRAM T4 (15 GiB) — giảm top_concepts "
            "(nhiều khái niệm hiếm chỉ làm đồ thị to lên chứ không thêm tín hiệu), rồi mới tới "
            "batch_size hoặc max_seq_len. Xem mục 'Sự cố thường gặp' của notebook."
        )
    if resume_grkt_ckpt:
        print(
            "  [resume] BẬT: ưu tiên GRKT-<graph>.pt (đã train xong -> chỉ đánh giá); nếu chưa "
            "có thì dùng ckpt_GRKT-<graph>.pt (dừng giữa epoch -> train tiếp). Chỉ dùng khi "
            "lần chạy trước có CÙNG split/đồ thị/siêu tham số — fingerprint trong checkpoint "
            "sẽ chặn nếu khác."
        )
    if deadline is not None:
        print(
            f"  [ngân sách] còn {max(0.0, deadline - time.time()) / 60.0:.0f} phút; checkpoint "
            f"giữa epoch mỗi {ckpt_every_min:g} phút -> {save_dir}/ckpt_GRKT-<graph>.pt"
        )
    if deadline is not None and time.time() >= deadline:
        # Ngân sách đã cạn trước cả batch đầu (thường do tập trước ăn hết thời gian): dừng
        # luôn thay vì sinh checkpoint khởi tạo ngẫu nhiên rồi mới raise.
        raise TrainingPaused(
            f"[{spec.name}] hết ngân sách thời gian TRƯỚC khi bắt đầu GRKT. Không có checkpoint "
            f"mới; tăng TIME_BUDGET_MIN (hoặc chạy riêng tập này ở phiên sau)."
        )
    grkt_results: List[dict] = []
    for kind in variants:
        print(f"\n--- GRKT với đồ thị '{kind}' ---")
        final_ckpt = save_dir / f"GRKT-{kind}.pt"
        rolling_ckpt = save_dir / f"ckpt_GRKT-{kind}.pt"
        resume = None
        if resume_grkt_ckpt:
            resume = (
                final_ckpt
                if final_ckpt.exists()
                else (rolling_ckpt if rolling_ckpt.exists() else None)
            )
            if resume is None and verbose:
                print(
                    f"  [resume] không có {final_ckpt.name} lẫn {rolling_ckpt.name} -> train từ đầu"
                )
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
            resume_ckpt=(str(resume) if resume is not None else None),
            deadline=deadline,
            ckpt_every_min=ckpt_every_min,
            verbose=verbose,
        )
        grkt_results.append(res)
    out["grkt_results"] = grkt_results

    # Hết ngân sách ngay sau GRKT: ghi bảng của những gì đã có rồi dừng, thay vì chạy đường
    # nền/quỹ đạo và bị Kaggle kill giữa chừng (checkpoint GRKT vẫn còn để phiên sau train tiếp).
    if deadline is not None and time.time() >= deadline:
        table = pd.DataFrame(
            [_table_row(r) for r in grkt_results], columns=TABLE_COLUMNS
        )
        table = table.sort_values("auc", ascending=False)
        table.to_csv(save_dir / "level2_metrics.csv", index=False)
        raise TrainingPaused(
            f"[{spec.name}] hết ngân sách thời gian sau GRKT. Đã ghi {save_dir/'level2_metrics.csv'} "
            f"(chỉ các dòng GRKT) và checkpoint trong {save_dir}. Phiên sau attach lại zip, đặt "
            "RESUME_GRKT_CKPT=True và tăng TIME_BUDGET_MIN để chạy tiếp đường nền/quỹ đạo."
        )

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
            print(
                f"  [baseline] {spec.name} không có checkpoint Cấp độ 1 -> huấn luyện lại "
                f"{baseline_epochs or epochs} epoch (trọng số ghi vào {save_dir}/baselines)"
            )
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
            deadline=deadline,
            verbose=verbose,
        )
        base_results, base_probs = got["results"], got["probs"]
    out["baseline_results"] = base_results

    # ------------------------------------------------------------ bảng so sánh
    table = pd.DataFrame(
        [_table_row(r) for r in base_results + grkt_results], columns=TABLE_COLUMNS
    )
    table = table.sort_values("auc", ascending=False)
    table.to_csv(save_dir / "level2_metrics.csv", index=False)
    out["table"] = table
    out["artifacts"]["metrics"] = str(save_dir / "level2_metrics.csv")
    print("\n=== BẢNG SO SÁNH DKT / GKT / GRKT (cùng split, cùng chỉ số) ===")
    print(table.to_string(index=False))

    # ------------------------------------------------------------ 6. quỹ đạo mastery
    traj_runs: Dict[str, dict] = {}
    stability_runs: Dict[str, dict] = {}
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
            stability_runs[kind] = stability_under_perturbation(
                model,
                splits["test"]["sequences"][:trajectory_users],
                device=device,
                batch_size=trajectory_batch,
                concept_num=data.concept_num,
                graph_kind=kind,
                seed=split_seed,
                verbose=verbose,
            )
            save_trajectories(traj, plot_dir / f"traj_{kind}.npz")
            export_mastery_csv(
                traj,
                plot_dir / f"mastery_{kind}.csv",
                max_users=25,
                skill_lookup=data.skill_lookup,
            )
            _safe_plot(
                plot_mastery_heatmap,
                traj,
                plot_dir / f"heatmap_{kind}.png",
                user_index=0,
                skill_lookup=data.skill_lookup,
            )
            _safe_plot(
                plot_mastery_curves,
                traj,
                plot_dir / f"curves_{kind}.png",
                user_indices=(0, 1),
                skill_lookup=data.skill_lookup,
            )
            model.cpu()
        traj_table = trajectory_metrics_table(
            traj_runs,
            adj=adj,
            pre_map=train_graph["pre_map"],
            stability=stability_runs,
            seed=split_seed,
        )
        traj_table.to_csv(save_dir / "trajectory_metrics.csv", index=False)
        out["trajectories"] = traj_runs
        out["stability"] = stability_runs
        out["trajectory_table"] = traj_table
        out["artifacts"]["trajectory_metrics"] = str(
            save_dir / "trajectory_metrics.csv"
        )
        out["artifacts"]["trajectory_plots"] = str(plot_dir)
        print(
            "\n=== TIÊU CHÍ QUỸ ĐẠO (láng giềng/cạnh tiên quyết cố định theo đồ thị train) ==="
        )
        print(traj_table.to_string(index=False))
        print(
            "  * Tiêu chí THẬT (có thể sai): precedence, precedence_gap, stability_spearman,\n"
            "    stability_changed_frac, forget_slope, pearson_dt_dm.\n"
            "  * consistency_strengthen / consistency_spread / locality bị cấu trúc mô hình\n"
            "    (Positive_Linear dương, k_hop=1) ép về 1 — chỉ là kiểm chứng cài đặt, KHÔNG\n"
            "    dùng làm bằng chứng vai trò đồ thị."
        )

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
        cal = calibration_table(
            entries,
            n_bins=10,
            extra={r["name"]: {"graph": r["graph_type"]} for r in grkt_results},
        )
        cal.to_csv(save_dir / "calibration.csv", index=False)
        _safe_plot(reliability_plot, entries, save_dir / "reliability.png")
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
        print(
            "\n>>> [C2-B8] Đa seed cho GRKT-%s (split cố định seed=%d) ..."
            % (ms_kind, split_seed)
        )
        rows = []
        paused = None
        try:
            for s in seeds:
                s = int(s)
                # Cùng quy ước tên với grkt_api: seed 42 dùng lại checkpoint của lần chạy
                # chính (khỏi train lại), seed khác có file riêng để resume đúng seed.
                suffix = "" if s == split_seed else f"-seed{s}"
                seed_name = f"GRKT-{ms_kind}{suffix}"
                final_ckpt = save_dir / f"{seed_name}.pt"
                rolling_ckpt = save_dir / f"ckpt_{seed_name}.pt"
                resume = None
                if resume_grkt_ckpt:
                    resume = (
                        str(final_ckpt)
                        if final_ckpt.exists()
                        else (str(rolling_ckpt) if rolling_ckpt.exists() else None)
                    )
                res = run_grkt_experiment(
                    data,
                    splits,
                    graphs[ms_kind],
                    graph_kind=ms_kind,
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
                    seed=s,
                    dataset=spec.name,
                    device=device,
                    save_dir=save_dir,
                    resume_ckpt=resume,
                    deadline=deadline,
                    ckpt_every_min=ckpt_every_min,
                    verbose=verbose,
                )
                # Bỏ hậu tố seed khỏi tên mô hình để groupby gộp đúng 3 seed thành MỘT dòng
                # mean ± std; tên file checkpoint vẫn tách theo seed.
                rows.append(
                    {
                        "model": f"GRKT-{ms_kind}",
                        "seed": s,
                        "test_auc": res["test_auc"],
                        "test_acc": res["test_acc"],
                        "test_loss": res["test_loss"],
                    }
                )
                del res
        except TrainingPaused as exc:
            # Giữ kết quả các seed ĐÃ xong thay vì mất trắng vì seed đang dở; checkpoint của
            # seed dở đã được train_grkt ghi trước khi raise.
            paused = exc
        if rows:
            ms = pd.DataFrame(rows)
            agg = (
                ms.groupby("model")
                .agg(
                    auc_mean=("test_auc", "mean"),
                    auc_std=("test_auc", "std"),
                    acc_mean=("test_acc", "mean"),
                    acc_std=("test_acc", "std"),
                    loss_mean=("test_loss", "mean"),
                    loss_std=("test_loss", "std"),
                    n_seed=("seed", "count"),
                )
                .reset_index()
            )
            # ddof=1 với n=3 cho std mẫu; ghi cả số seed để người đọc không nhầm với tổng thể
            agg.to_csv(save_dir / "multiseed_metrics.csv", index=False)
            out["multiseed_raw"] = ms
            out["multiseed"] = agg
            out["artifacts"]["multiseed"] = str(save_dir / "multiseed_metrics.csv")
            print(agg.to_string(index=False))
        if paused is not None:
            raise paused

    summary = {
        "dataset": spec.name,
        "canonical": str(canonical),
        "time_mode": data.time_mode,
        "data_stats": {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in data.stats.items()
        },
        "graph_edges": {
            k: {"rel": int(v["edges_rel"]), "pre": int(v["edges_pre"])}
            for k, v in graphs.items()
        },
        "counts": {
            "train_edges_from_train": int(
                (maps_train["rel"] > 0).sum() + (maps_train["pre"] > 0).sum()
            ),
            "train_edges_from_val": int(
                (maps_val["rel"] > 0).sum() + (maps_val["pre"] > 0).sum()
            ),
        },
        "models": [_table_row(r) for r in base_results + grkt_results],
        "trajectory": (
            None if traj_table is None else traj_table.to_dict(orient="records")
        ),
        "calibration": cal.to_dict(orient="records"),
    }
    (save_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
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
        "val_auc": (
            None
            if r.get("best_val_auc") is None
            or np.isnan(r.get("best_val_auc", float("nan")))
            else round(float(r["best_val_auc"]), 4)
        ),
        "n_test": int(r.get("test_n", 0)),
        "edges_rel": int(r.get("edges_rel", 0)),
        "edges_pre": int(r.get("edges_pre", 0)),
        # seed đã TRAIN ra trọng số; None = checkpoint không ghi seed (không được đoán 42).
        "seed": (None if r.get("seed") is None else int(r["seed"])),
        "checkpoint_seed": (
            None if r.get("ckpt_seed") is None else int(r["ckpt_seed"])
        ),
        "seconds": round(float(r.get("train_seconds", 0.0)), 1),
    }


def run_level2(
    datasets: Sequence[str] = ("assist2009", "junyi"),
    *,
    save_root=None,
    time_budget_min: Optional[float] = None,
    variants_by_dataset: Optional[Dict[str, Sequence[str]]] = None,
    multi_seed_datasets: Optional[Sequence[str]] = None,
    **kwargs,
) -> Dict[str, object]:
    """Lặp :func:`run_dataset` qua nhiều tập dữ liệu và nối bảng (yêu cầu ≥ 2 dataset).

    ``time_budget_min`` là ngân sách của CẢ LẦN CHẠY: mốc dừng được tính MỘT lần ở đây rồi
    truyền xuống từng tập, nên tập thứ hai không được cấp lại đủ ngân sách và phiên Kaggle
    không bị kill giữa lúc ghi checkpoint.

    ``variants_by_dataset`` / ``multi_seed_datasets`` cho phép giới hạn phần đắt tiền vào
    tập rẻ (thường là assist2009): ablation đồ thị và đa seed chỉ chạy nơi còn ngân sách
    GPU, tập còn lại giữ cấu hình mặc định (``variants=("train",)``, 1 seed) và điều này
    phải được ghi rõ trong báo cáo.

    Có mặt ``variants_by_dataset`` thì bảng khai báo là nguồn quyết định: tập **không** có
    trong bảng nhận đúng ``("train",)``, không thừa hưởng ``variants`` truyền qua ``kwargs``.
    Nhờ vậy bật ablation cho tập rẻ không âm thầm train thêm biến thể nặng ở tập còn lại.
    """
    deadline = (
        (time.time() + float(time_budget_min) * 60.0) if time_budget_min else None
    )
    runs: Dict[str, dict] = {}
    frames = []
    ms_set = set(multi_seed_datasets) if multi_seed_datasets is not None else None
    for name in datasets:
        print("\n" + "#" * 72)
        print(f"# TẬP DỮ LIỆU: {name}")
        print("#" * 72)
        ds_kwargs = dict(kwargs)
        if variants_by_dataset is not None:
            # Tập ngoài bảng PHẢI nhận đúng một biến thể; nếu thừa hưởng `variants` toàn cục
            # thì khai báo ablation cho assist2009 sẽ kéo theo GRKT-zero ở junyi (hàng giờ GPU).
            ds_kwargs["variants"] = tuple(variants_by_dataset.get(name, ("train",)))
        if ms_set is not None:
            ds_kwargs["run_multi_seed"] = name in ms_set
        run = run_dataset(name, save_root=save_root, deadline=deadline, **ds_kwargs)
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
