"""API huấn luyện/đánh giá GRKT cho Cấp độ 2, thay thế ``GRKT/Runner.py``.

Lý do không dùng ``Runner.py``:

* Runner chia k-fold trên CỬA SỔ 100 tương tác cắt từ chuỗi của từng người học, nên cùng
  một người học xuất hiện ở cả train lẫn test (phạm ``YEU_CAU.md`` §6.4). Ở đây split là
  danh sách người học của Cấp độ 1 (``level1.pipeline.save_split``) — mỗi người học chỉ
  thuộc MỘT tập.
* Runner nhân bản model bằng ``deepcopy(self)`` để giữ model tốt nhất; bản đó giữ cả
  tensor trên GPU và làm hết VRAM sau vài epoch. Ở đây chỉ snapshot ``state_dict()`` về
  CPU, giống ``level1.pipeline.train_one``.

Giữ nguyên tối đa đường dẫn tính toán của upstream: ``get_feed_dict``/``feature2tensor``
(Batch tự đệm theo chuỗi dài nhất trong batch), ``train_an_epoch``/``eval_an_epoch``,
``AUC``/``ACC`` của ``GRKT/metric.py``. Chỉ vòng lặp epoch/checkpoint là của Cấp độ 2.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from level1.paths import repo_root


# ---------------------------------------------------------------- import GRKT
def ensure_grkt_importable() -> Path:
    """Đảm bảo ``import GRKT`` hoạt động dù notebook chạy từ /kaggle/working."""
    root = str(repo_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    return repo_root()


def import_grkt():
    ensure_grkt_importable()
    from GRKT.model.GRKT import GRKT

    return GRKT


# ---------------------------------------------------------------- args
def make_args(
    n_knows: int,
    n_probs: int,
    rel_map: np.ndarray,
    pre_map: np.ndarray,
    *,
    dataset: str = "assist2009",
    device: str = "cpu",
    lr: float = 1e-3,
    l2: float = 0.0,
    epoch: int = 20,
    early_stop: int = 5,
    batch_size: int = 64,
    d_hidden: int = 128,
    k_hidden: int = 16,
    k_hop: int = 1,
    pos_mode: str = "sigmoid",
    tau: float = 0.2,
    alpha: float = 0.01,
    thresh: float = 0.5,
    seed: int = 42,
    bar: int = 0,
    save_model: int = 0,
    save: int = 0,
) -> argparse.Namespace:
    """Namespace chứa ĐỦ các khoá mà ``BaseModel``/``GRKT`` đọc tới bằng ``args.<key>``.

    Hai điểm khác mặc định của upstream, có chủ đích và được giải thích ở ``KE_HOACH.md``:

    * ``thresh=0.5`` (upstream 0.6): ma trận ``rel_map``/``pre_map`` Cấp độ 2 đưa vào đây
      đã là 0/1 (xem ``level2.grkt_data.graph_variant``), nên ``(map > 0.5)`` giữ nguyên
      đồ thị. Tuyệt đối không đặt 0: nhánh ``if args.thresh == 0`` của GRKT gán toàn bộ
      ma trận = 1, kể cả hàng/cột padding số 0.
    * ``alpha=0.01`` và trục thời gian tính bằng PHÚT đã chặn trên
      (``level2.grkt_data._cumulative_times``). Với mốc thời gian thô, ``exp(-(n+1)*Δt*k)``
      underflow về 0 và mô hình "quên sạch" sau đúng một bước.
    """
    rel_map = np.asarray(rel_map, dtype=np.float32)
    pre_map = np.asarray(pre_map, dtype=np.float32)
    if rel_map.shape != (n_knows + 1, n_knows + 1) or pre_map.shape != (n_knows + 1, n_knows + 1):
        raise ValueError(
            f"rel_map/pre_map phải có dạng ({n_knows + 1}, {n_knows + 1}) "
            f"(n_knows + 1 cho padding), nhận được {rel_map.shape} và {pre_map.shape}"
        )
    return argparse.Namespace(
        # dữ liệu / đồ thị
        dataset=dataset,
        n_knows=int(n_knows),
        n_probs=int(n_probs),
        rel_map=rel_map,
        pre_map=pre_map,
        thresh=float(thresh),
        # quy mô mô hình
        d_hidden=int(d_hidden),
        k_hidden=int(k_hidden),
        k_hop=int(k_hop),
        pos_mode=pos_mode,
        tau=float(tau),
        alpha=float(alpha),
        # huấn luyện
        lr=float(lr),
        l2=float(l2),
        epoch=int(epoch),
        early_stop=int(early_stop),
        batch_size=int(batch_size),
        device=device,
        seed=int(seed),
        bar=int(bar),
        save_model=int(save_model),
        save=int(save),
        # các khoá Runner còn tham chiếu (không dùng tới vòng lặp của Runner)
        model="GRKT",           # GRKT/Training.py đọc args.model để tra MODEL_REGISTRY
        k_exps=1,
        n_exps=1,
        repeat=1,
        exp=0,
        split_ratio=1.0,
    )


def build_grkt(args: argparse.Namespace, device=None):
    """Khởi tạo GRKT và chuyển lên thiết bị. ``args.k_hop`` phải >= 1."""
    if args.k_hop < 1:
        raise ValueError("GRKT yêu cầu k_hop >= 1 (các ModuleList học theo số hop)")
    GRKT = import_grkt()
    model = GRKT(args)
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------- batching
def batch_items(items: Sequence[list], batch_size: int, shuffle: bool = True) -> List[list]:
    """Tách danh sách chuỗi tương tác thành batch, giống ``BaseModel.train_an_epoch``.

    Đệm (padding) do ``feature2tensor`` quyết định theo chuỗi dài nhất của TỪNG batch,
    nên không được để lẫn chuỗi rất ngắn với chuỗi rất dài khi debug bộ nhớ.
    """
    data = list(items)
    if shuffle:
        random.shuffle(data)
    return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]


def _to_device_feed(model, batch, mode: str, device) -> dict:
    feed = model.get_feed_dict(batch, mode)
    # get_feed_dict đã chuyển theo args.device; đảm bảo khớp với nơi model đang nằm
    if str(device) != str(feed["probs"].device):
        for key in ("probs", "knows", "corrs", "times", "lags", "filt", "labels"):
            feed[key] = feed[key].to(device)
    return feed


# ---------------------------------------------------------------- train
def train_grkt(
    model,
    train_items: Sequence[list],
    val_items: Sequence[list],
    epochs: int = 20,
    lr: Optional[float] = None,
    patience: int = 5,
    device=None,
    tag: str = "GRKT",
    verbose: bool = True,
) -> Dict[str, object]:
    """Vòng lặp epoch riêng của Cấp độ 2: early stop theo val AUC TOÀN CỤC.

    Dùng lại ``train_an_epoch``/``eval_an_epoch`` của upstream cho việc đệm batch và tính
    AUC/ACC, nhưng tự quản lý optimizer và checkpoint (snapshot ``state_dict`` về CPU thay
    vì ``deepcopy(model)``).
    """
    if device is None:
        device = next(model.parameters()).device
    if lr is None:
        lr = float(getattr(model.args, "lr", 1e-3))
    model.args.device = device
    model.optim = model.get_optimizer()

    history = {k: [] for k in ["train_loss", "val_loss", "val_auc", "val_acc"]}
    best_auc, best_state, wait = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, int(epochs) + 1):
        train_loss = model.train_an_epoch(list(train_items))
        va = _eval_items(model, val_items, device, tag=tag)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(va["loss"])
        history["val_auc"].append(va["auc"])
        history["val_acc"].append(va["acc"])
        improved = va["auc"] > best_auc
        if improved:
            best_auc = va["auc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if verbose:
            print(
                f"  [{tag}] epoch {epoch:02d}/{epochs} train loss={float(train_loss):.4f} "
                f"val loss={va['loss']:.4f} val auc={va['auc']:.4f} val acc={va['acc']:.4f}"
                + (" *" if improved else "")
            )
        if wait >= patience:
            if verbose:
                print(f"  [{tag}] early stopping tại epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    elapsed = time.time() - t0
    if verbose:
        print(f"  [{tag}] xong sau {elapsed:.1f}s | val AUC tốt nhất={best_auc:.4f}")
    return {"history": history, "best_val_auc": float(best_auc), "seconds": elapsed}


@torch.no_grad()
def _eval_items(model, items: Sequence[list], device, tag: str = "") -> Dict[str, float]:
    """Đánh giá TOÀN CỤC (nối mọi batch rồi tính AUC/BCE một lần), giống ``_epoch_loop`` Cấp độ 1."""
    from sklearn.metrics import roc_auc_score

    model.eval()
    scores, labels, losses, counts = [], [], [], []
    for batch in batch_items(items, model.args.batch_size, shuffle=False):
        feed = _to_device_feed(model, batch, "valid", device)
        model.forward(feed)
        s = feed["scores"].detach().float().cpu()
        y = feed["labels"].detach().float().cpu()
        if s.numel() == 0:
            continue
        scores.append(s)
        labels.append(y)
        p = s.clamp(1e-6, 1.0 - 1e-6)
        bce = -(y * torch.log(p) + (1.0 - y) * torch.log(1.0 - p))
        losses.append(bce.sum())
        counts.append(int(y.numel()))
    if not scores:
        return {"auc": -1.0, "acc": -1.0, "loss": 0.0, "n": 0}
    y_true = torch.cat(labels).numpy()
    y_pred = torch.cat(scores).numpy()
    n = int(sum(counts))
    loss = float(sum(losses) / max(n, 1))
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_pred))
        acc = float(((y_pred >= 0.5) == (y_true >= 0.5)).mean())
    else:
        auc, acc = -1.0, -1.0
    return {"auc": auc, "acc": acc, "loss": loss, "n": n}


def evaluate_grkt(model, items: Sequence[list], device=None) -> Dict[str, float]:
    """Điểm test: AUC/ACC toàn cục + BCE đã kẹp [1e-6, 1-1e-6] như ``GKT/metrics.KTLoss``."""
    if device is None:
        device = next(model.parameters()).device
    model.args.device = device
    model.to(device)
    return _eval_items(model, items, device)


def predict_grkt(model, items: Sequence[list], device=None):
    """Trả về (scores, labels) đã nối toàn cục — dùng cho bảng calibration ECE/Brier."""
    if device is None:
        device = next(model.parameters()).device
    model.args.device = device
    model.to(device)
    model.eval()
    scores, labels = [], []
    for batch in batch_items(items, model.args.batch_size, shuffle=False):
        feed = _to_device_feed(model, batch, "test", device)
        model.forward(feed)
        if feed["scores"].numel() == 0:
            continue
        scores.append(feed["scores"].detach().float().cpu())
        labels.append(feed["labels"].detach().float().cpu())
    if not scores:
        return np.array([]), np.array([])
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


# ---------------------------------------------------------------- experiment
def run_grkt_experiment(
    grkt_data,
    split_sets: Dict[str, dict],
    graph_info: Dict[str, object],
    *,
    graph_kind: str = "train",
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 5,
    d_hidden: int = 128,
    k_hidden: int = 16,
    k_hop: int = 1,
    alpha: float = 0.01,
    tau: float = 0.2,
    thresh: float = 0.5,
    seed: int = 42,
    dataset: str = "assist2009",
    device=None,
    save_dir=None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Huấn luyện một biến thể đồ thị của GRKT và trả về dict ĐỦ KHOÁ như ``level1.pipeline.run_experiment``.

    Nhờ trùng bộ khoá (``name/model_type/graph_type/concept_num/best_val_auc/test_loss/
    test_auc/test_acc/history/seed/model``) nên bảng so sánh DKT/GKT/GRKT chỉ cần nối các
    kết quả của hai cấp.
    """
    from level1.pipeline import set_seed

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    rel = graph_info["rel_map"] if isinstance(graph_info, dict) else graph_info[0]
    pre = graph_info["pre_map"] if isinstance(graph_info, dict) else graph_info[1]
    args = make_args(
        n_knows=grkt_data.concept_num,
        n_probs=grkt_data.n_probs,
        rel_map=rel,
        pre_map=pre,
        dataset=dataset,
        device=str(device),
        lr=lr,
        epoch=epochs,
        early_stop=patience,
        batch_size=batch_size,
        d_hidden=d_hidden,
        k_hidden=k_hidden,
        k_hop=k_hop,
        alpha=alpha,
        tau=tau,
        thresh=thresh,
        seed=seed,
    )
    model = build_grkt(args, device=device)
    if verbose:
        n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"  GRKT[{graph_kind}] n_knows={args.n_knows} n_probs={args.n_probs} "
            f"edges rel/pre={int(np.asarray(rel).sum())}/{int(np.asarray(pre).sum())} "
            f"| tham số={n_param / 1e6:.2f}M"
        )

    train_items = split_sets["train"]["items"]
    val_items = split_sets["val"]["items"]
    test_items = split_sets["test"]["items"]
    out = train_grkt(model, train_items, val_items, epochs=epochs, lr=lr, patience=patience,
                     device=device, tag=f"GRKT-{graph_kind}", verbose=verbose)
    test_metrics = evaluate_grkt(model, test_items, device=device)

    result = {
        "name": f"GRKT-{graph_kind}",
        "model_type": "GRKT",
        "graph_type": graph_kind,
        "concept_num": grkt_data.concept_num,
        "best_val_auc": out["best_val_auc"],
        "test_loss": test_metrics["loss"],
        "test_auc": test_metrics["auc"],
        "test_acc": test_metrics["acc"],
        "test_n": test_metrics["n"],
        "history": out["history"],
        "train_seconds": out["seconds"],
        "seed": seed,
        "edges_rel": int(rel.sum()) if hasattr(rel, "sum") else 0,
        "edges_pre": int(pre.sum()) if hasattr(pre, "sum") else 0,
        "args": args,
    }
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = save_dir / f"{result['name']}.pt"
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, ckpt)
        result["ckpt"] = str(ckpt)
    result["model"] = model.cpu()
    print(
        f">>> {result['name']} test AUC={result['test_auc']:.4f} "
        f"Acc={result['test_acc']:.4f} Loss={result['test_loss']:.4f}"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
