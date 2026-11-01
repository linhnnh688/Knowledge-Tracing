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
import os
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
    if rel_map.shape != (n_knows + 1, n_knows + 1) or pre_map.shape != (
        n_knows + 1,
        n_knows + 1,
    ):
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
        model="GRKT",  # GRKT/Training.py đọc args.model để tra MODEL_REGISTRY
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
def batch_items(
    items: Sequence[list], batch_size: int, shuffle: bool = True
) -> List[list]:
    """Tách danh sách chuỗi tương tác thành batch, giống ``BaseModel.train_an_epoch``.

    Đệm (padding) do ``feature2tensor`` quyết định theo chuỗi dài nhất của TỪNG batch,
    nên không được để lẫn chuỗi rất ngắn với chuỗi rất dài khi debug bộ nhớ.
    """
    data = list(items)
    if shuffle:
        random.shuffle(data)
    return [data[i : i + batch_size] for i in range(0, len(data), batch_size)]


def _to_device_feed(model, batch, mode: str, device) -> dict:
    feed = model.get_feed_dict(batch, mode)
    # get_feed_dict đã chuyển theo args.device; đảm bảo khớp với nơi model đang nằm
    if str(device) != str(feed["probs"].device):
        for key in ("probs", "knows", "corrs", "times", "lags", "filt", "labels"):
            feed[key] = feed[key].to(device)
    return feed


# ---------------------------------------------------------------- checkpoint/resume
#: Định danh cho phép phân biệt checkpoint "đang train dở" (có optimizer/epoch/batch) với
#: checkpoint cuối cùng ``GRKT-<graph>.pt`` (trọng số tốt nhất + fingerprint, nạp là bỏ qua train).
RESUME_FORMAT = "grkt-resume-v1"
FINAL_FORMAT = "grkt-final-v1"


class TrainingPaused(RuntimeError):
    """Hết ngân sách thời gian. Checkpoint đã được ghi NGAY TRƯỚC khi raise."""


def _atomic_torch_save(payload, path: Path) -> None:
    """Ghi ra file tạm rồi ``os.replace`` — session bị kill giữa lúc ghi không làm hỏng file cũ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _cpu_state_dict(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state.items()}


def _optimizer_state_cpu(optim) -> dict:
    """Đưa trạng thái Adam về CPU; checkpoint nhỏ, nạp lại ở phiên khác không phụ thuộc GPU."""
    state = optim.state_dict()
    return {
        "state": {
            idx: {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in entry.items()
            }
            for idx, entry in state["state"].items()
        },
        "param_groups": state["param_groups"],
    }


def _optimizer_state_to_device(optim, device) -> None:
    for entry in optim.state.values():
        for k, v in entry.items():
            if torch.is_tensor(v):
                entry[k] = v.to(device)


def _rng_state() -> dict:
    """RNG của cả ba thư viện: GRKT dùng ``gumbel_softmax`` ngay cả lúc eval.

    Không lưu thì lần resume sẽ bắt đầu một dòng ngẫu nhiên khác, nên val/test AUC sau khi
    resume lệch so với lần chạy liền mạch dù trọng số đúng — và tiêu chí dừng sớm bị nhiễu.
    """
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Optional[dict]) -> None:
    if not state:
        return
    try:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state.get("cuda") and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except (
        Exception
    ) as exc:  # noqa: BLE001 - RNG khác thiết bị không đáng làm chết lần resume
        print(
            f"  [resume] không khôi phục được RNG ({type(exc).__name__}: {exc}) — train tiếp "
            "với dòng ngẫu nhiên mới; kết quả không trùng bit-for-bit với lần chạy liền mạch."
        )


def _check_fingerprint(previous: Optional[dict], current: dict) -> None:
    """Từ chối resume nếu split/đồ thị/siêu tham số đã đổi (load_state_dict không bắt được)."""
    if not previous:
        return
    diff = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
    if diff:
        detail = ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in diff.items())
        raise RuntimeError(
            "Checkpoint được tạo với cấu hình khác (" + detail + "). Xoá checkpoint cũ "
            "(cả ckpt_GRKT-*.pt lẫn GRKT-*.pt) rồi train lại — tiếp tục với cấu hình khác "
            "sẽ cho kết quả không tái lập được."
        )


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
    *,
    seed: int = 42,
    resume_state: Optional[dict] = None,
    ckpt_path=None,
    fingerprint: Optional[dict] = None,
    deadline: Optional[float] = None,
    ckpt_every_s: float = 600.0,
) -> Dict[str, object]:
    """Vòng lặp epoch riêng của Cấp độ 2: early stop theo val AUC TOÀN CỤC.

    Khác upstream ``Runner.fit`` ở hai điểm phục vụ resume qua nhiều phiên Kaggle:

    * Shuffle bằng ``np.random.RandomState`` và lưu ``order`` + ``batch_pos`` vào checkpoint,
      nên dừng giữa epoch vẫn tiếp tục đúng thứ tự batch còn lại.
    * Cứ ``ckpt_every_s`` giây (và cuối mỗi epoch) ghi ``ckpt_path`` gồm weights, optimizer,
      epoch, batch, best_state. Session bị kill mất tối đa ``ckpt_every_s`` phút.
    * ``deadline`` (epoch giây) cho phép dừng chủ động trước khi Kaggle cắt phiên; checkpoint
      được ghi rồi raise :class:`TrainingPaused`.
    """
    if device is None:
        device = next(model.parameters()).device
    if lr is None:
        lr = float(getattr(model.args, "lr", 1e-3))
    model.args.device = device
    model.optim = model.get_optimizer()

    history = {k: [] for k in ["train_loss", "val_loss", "val_auc", "val_acc"]}
    best_auc, best_state, wait = -1.0, None, 0
    epoch, batch_pos = 0, 0
    order: Optional[List[int]] = None
    loss_sum, loss_count = 0.0, 0
    if resume_state:
        _check_fingerprint(resume_state.get("fingerprint"), fingerprint or {})
        model.load_state_dict(resume_state["state_dict"])
        model.optim.load_state_dict(resume_state["optim"])
        _optimizer_state_to_device(model.optim, device)
        model.to(device)
        _restore_rng(resume_state.get("rng"))
        epoch = int(resume_state.get("epoch", 0))
        batch_pos = int(resume_state.get("batch_pos", 0))
        order = resume_state.get("order")
        loss_sum = float(resume_state.get("loss_sum", 0.0))
        loss_count = int(resume_state.get("loss_count", 0))
        best_auc = float(resume_state.get("best_auc", -1.0))
        best_state = resume_state.get("best_state")
        wait = int(resume_state.get("wait", 0))
        history = resume_state.get("history", history)
        if verbose:
            print(
                f"  [{tag}] RESUME: đã xong epoch {epoch}, đang ở batch {batch_pos} "
                f"| best val AUC={best_auc:.4f} | chờ {wait}/{patience}"
            )

    def _save(
        ep: int, pos: int, ord_: Optional[List[int]], lsum: float, lcnt: int
    ) -> None:
        if ckpt_path is None:
            return
        _atomic_torch_save(
            {
                "format": RESUME_FORMAT,
                "state_dict": _cpu_state_dict(model.state_dict()),
                "optim": _optimizer_state_cpu(model.optim),
                "epoch": int(ep),
                "batch_pos": int(pos),
                "order": ord_,
                "loss_sum": float(lsum),
                "loss_count": int(lcnt),
                "best_auc": float(best_auc),
                "best_state": best_state,
                "wait": int(wait),
                "history": history,
                "rng": _rng_state(),
                "fingerprint": fingerprint or {},
                "tag": tag,
            },
            ckpt_path,
        )

    t0 = time.time()
    last_ckpt = t0
    n_items = len(train_items)
    bs = max(1, int(model.args.batch_size))
    while epoch < int(epochs):
        current = epoch + 1  # epoch đang chạy (1-based); ``epoch`` = số epoch đã xong
        if order is None:
            # Epoch mới (hoặc vừa nạp checkpoint cuối epoch): sinh thứ tự batch và reset loss.
            order = (
                np.random.RandomState(int(seed) * 1000 + current)
                .permutation(n_items)
                .tolist()
            )
            loss_sum, loss_count = 0.0, 0
            batch_pos = 0
        n_batches = (len(order) + bs - 1) // bs
        model.train()
        for bidx in range(batch_pos, n_batches):
            if deadline is not None and time.time() >= deadline:
                # Lưu ``current - 1`` (số epoch đã xong) + batch đã xong + order, để resume
                # không bỏ qua epoch khi dừng đúng ở batch 0.
                _save(current - 1, bidx, order, loss_sum, loss_count)
                raise TrainingPaused(
                    f"[{tag}] hết ngân sách thời gian tại epoch {current}, batch {bidx}/"
                    f"{n_batches}. Checkpoint đã lưu: {ckpt_path}. Phiên sau attach lại zip "
                    f"và đặt RESUME_GRKT_CKPT=True."
                )
            chunk = [train_items[i] for i in order[bidx * bs : (bidx + 1) * bs]]
            model.optim.zero_grad()
            feed = model.get_feed_dict(chunk, "train")
            feed["mark"] = bidx * bs
            model.forward(feed)
            loss = model.loss(feed)
            loss.backward()
            model.optim.step()
            loss_sum += float(loss.item())
            loss_count += 1
            now = time.time()
            if (
                ckpt_path is not None
                and ckpt_every_s
                and now - last_ckpt >= ckpt_every_s
            ):
                _save(current - 1, bidx + 1, order, loss_sum, loss_count)
                last_ckpt = now
                if verbose:
                    print(
                        f"  [{tag}] checkpoint giữa epoch: epoch {current}, batch {bidx + 1}/"
                        f"{n_batches} -> {Path(ckpt_path).name}"
                    )

        epoch, batch_pos, order = current, 0, None
        train_loss = loss_sum / max(loss_count, 1)
        va = _eval_items(model, val_items, device, tag=tag)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(va["loss"])
        history["val_auc"].append(va["auc"])
        history["val_acc"].append(va["acc"])
        improved = va["auc"] > best_auc
        if improved:
            best_auc = va["auc"]
            best_state = _cpu_state_dict(model.state_dict())
            wait = 0
        else:
            wait += 1
        if verbose:
            print(
                f"  [{tag}] epoch {epoch:02d}/{epochs} train loss={float(train_loss):.4f} "
                f"val loss={va['loss']:.4f} val auc={va['auc']:.4f} val acc={va['acc']:.4f}"
                + (" *" if improved else "")
            )
        _save(epoch, 0, None, 0.0, 0)
        last_ckpt = time.time()
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
def _eval_items(
    model, items: Sequence[list], device, tag: str = ""
) -> Dict[str, float]:
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
    resume_ckpt: Optional[str] = None,
    deadline: Optional[float] = None,
    ckpt_every_min: float = 10.0,
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
    # Seed 42 giữ tên cũ (GRKT-<graph>.pt) để không phá sản phẩm đã có; seed khác mang hậu tố
    # riêng, nếu không vòng đa seed sẽ ghi đè checkpoint của lần chạy chính và resume sai seed.
    name = (
        f"GRKT-{graph_kind}"
        if int(seed) == 42
        else f"GRKT-{graph_kind}-seed{int(seed)}"
    )

    # Vân tay dữ liệu/cấu hình: load_state_dict chỉ bắt khác kiến trúc, KHÔNG bắt khác split,
    # đồ thị hay batch_size — trong khi resume sai split sẽ cho kết quả không tái lập được.
    fingerprint = {
        "dataset": dataset,
        "graph_kind": graph_kind,
        "n_knows": int(grkt_data.concept_num),
        "n_probs": int(grkt_data.n_probs),
        "n_train": len(train_items),
        "n_val": len(val_items),
        "n_test": len(test_items),
        "n_steps_train": int(sum(len(items) for items in train_items)),
        "batch_size": int(batch_size),
        "d_hidden": int(d_hidden),
        "k_hidden": int(k_hidden),
        "k_hop": int(k_hop),
        "alpha": float(alpha),
        "tau": float(tau),
        "thresh": float(thresh),
        "seed": int(seed),
        "lr": float(lr),
        "edges_rel": int(np.asarray(rel).sum()),
        "edges_pre": int(np.asarray(pre).sum()),
    }
    rolling_ckpt = (
        (Path(save_dir) / f"ckpt_{name}.pt") if save_dir is not None else None
    )

    loaded_from = None
    resume_state = None
    skipped_training = False
    if resume_ckpt is not None and Path(resume_ckpt).exists():
        # Checkpoint giữa epoch có list/dict/float (order, history, fingerprint) nên phải dùng
        # weights_only=False; file do chính repo sinh ra và người dùng tự attach.
        loaded_from = str(resume_ckpt)
        try:
            payload = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 - checkpoint ghi dở lúc bị kill
            # ``GRKT-<graph>.pt`` được ghi nguyên tử nên trường hợp này chỉ xảy ra với file do
            # bản cũ sinh ra; còn ``ckpt_GRKT-<graph>.pt`` thì vẫn resume tiếp được.
            fallback = (
                (Path(save_dir) / f"ckpt_{name}.pt") if save_dir is not None else None
            )
            if fallback is not None and fallback.exists():
                print(
                    f"  [resume] {loaded_from} hỏng ({type(exc).__name__}: {exc}) -> "
                    f"dùng {fallback.name}"
                )
                resume_ckpt = fallback
                loaded_from = str(fallback)
                payload = torch.load(fallback, map_location="cpu", weights_only=False)
            else:
                raise RuntimeError(
                    f"Checkpoint {loaded_from} không đọc được ({type(exc).__name__}: {exc}). "
                    "Xoá file này rồi chạy lại (không còn checkpoint giữa epoch để nạp thay)."
                ) from exc

        fmt = payload.get("format") if isinstance(payload, dict) else None
        if fmt == RESUME_FORMAT:
            resume_state = payload
            if verbose:
                print(
                    f"  [resume] {loaded_from}: tiếp tục từ epoch {payload.get('epoch')}, "
                    f"batch {payload.get('batch_pos')} | best val AUC="
                    f"{float(payload.get('best_auc', -1.0)):.4f}"
                )
        elif fmt == FINAL_FORMAT:
            # Không có bước này thì đổi split/đồ thị giữa hai phiên vẫn nạp được trọng số
            # (cùng shape) và bảng so sánh trộn hai lần chạy mà không báo lỗi.
            _check_fingerprint(payload.get("fingerprint"), fingerprint)
            model.load_state_dict(payload["state_dict"])
            skipped_training = True
        elif isinstance(payload, dict) and any(
            torch.is_tensor(v) for v in payload.values()
        ):
            # Checkpoint đời cũ: chỉ có trọng số nên KHÔNG kiểm chứng được split/cấu hình.
            print(
                f"  [cảnh báo] {loaded_from} chỉ chứa trọng số (không có fingerprint): "
                "không kiểm chứng được nó train trên cùng split. Chỉ nạp khi chắc chắn."
            )
            model.load_state_dict(payload)
            skipped_training = True
        else:
            raise RuntimeError(f"{loaded_from} không phải checkpoint GRKT của repo này")
        if skipped_training:
            val_metrics = evaluate_grkt(model, val_items, device=device)
            out = {
                "best_val_auc": val_metrics["auc"],
                "history": {
                    "train_loss": [],
                    "val_loss": [],
                    "val_auc": [],
                    "val_acc": [],
                },
                "seconds": 0.0,
            }
            if verbose:
                print(
                    f"  [resume] nạp {loaded_from}, BỎ QUA HUẤN LUYỆN "
                    f"(val AUC={val_metrics['auc']:.4f}). Đánh giá lại nên số lệch nhẹ so với "
                    "phiên trước là do ``gumbel_softmax`` ở eval, không phải trọng số khác."
                )
    if not skipped_training:
        if resume_ckpt is not None and resume_state is None and verbose:
            print(f"  [resume] không có {resume_ckpt} -> huấn luyện bình thường")
        out = train_grkt(
            model,
            train_items,
            val_items,
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=device,
            tag=name,
            verbose=verbose,
            seed=seed,
            resume_state=resume_state,
            ckpt_path=rolling_ckpt,
            fingerprint=fingerprint,
            deadline=deadline,
            ckpt_every_s=(float(ckpt_every_min) * 60.0 if ckpt_every_min else 0.0),
        )
    test_metrics = evaluate_grkt(model, test_items, device=device)

    result = {
        "name": name,
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
        "resumed_from": loaded_from,
        # None = trọng số vừa train trong lần chạy này; int = nạp từ checkpoint đã kiểm
        # fingerprint (seed trùng với ``seed`` nên không thể lệch split/khởi tạo).
        "ckpt_seed": int(seed) if loaded_from else None,
        "args": args,
    }
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = save_dir / f"{result['name']}.pt"
        # Nguyên tử + kèm fingerprint: phiên sau nạp file này để BỎ QUA train, nên nó phải tự
        # chứng minh được là được train trên đúng split/cấu hình đang dùng.
        _atomic_torch_save(
            {
                "format": FINAL_FORMAT,
                "state_dict": _cpu_state_dict(model.state_dict()),
                "fingerprint": fingerprint,
                "best_val_auc": float(out["best_val_auc"]),
                "epochs": int(epochs),
                "seed": int(seed),
            },
            ckpt,
        )
        result["ckpt"] = str(ckpt)
        if rolling_ckpt is not None and rolling_ckpt.exists():
            # Đã có checkpoint cuối: giữ bản giữa epoch chỉ làm zip resume nặng thêm và dễ
            # khiến người đọc tưởng lần chạy còn dở.
            rolling_ckpt.unlink()
    result["model"] = model.cpu()
    print(
        f">>> {result['name']} test AUC={result['test_auc']:.4f} "
        f"Acc={result['test_acc']:.4f} Loss={result['test_loss']:.4f}"
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
