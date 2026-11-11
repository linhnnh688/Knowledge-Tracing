"""Callable training/eval API for cap do 1. The Kaggle notebook only orchestrates these functions."""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from kt_common.checkpoint import (
    atomic_save,
    check_fingerprint,
    checkpoint_path,
    cpu_state_dict,
    optimizer_state_cpu,
    optimizer_state_to_device,
    record_checkpoint,
)
from kt_common.ids import as_id
from kt_common.interactions import DATA_VERSION
from kt_common.log import get_logger
from kt_common.metrics import auc_acc
from kt_common.seed import set_seed  # noqa: F401 - API công khai, notebook import từ đây

from .paths import ensure_repo_on_path, repo_root

ensure_repo_on_path()

from GKT.metrics import KTLoss  # noqa: E402
from GKT.models import DKT, GKT, MultiHeadAttention  # noqa: E402
from GKT.processing import load_dataset  # noqa: E402

log = get_logger("level1.pipeline")

#: Khoá lịch sử train theo epoch (dùng chung với ``level2.baselines``).
HISTORY_KEYS = (
    "train_loss",
    "val_loss",
    "train_auc",
    "val_auc",
    "train_acc",
    "val_acc",
)
#: Định danh payload "train dở" theo epoch. Giữ nguyên chuỗi cũ của ``level2.baselines`` để
#: checkpoint resume đã có trên Kaggle vẫn nạp được.
RESUME_FORMAT = "level1-resume-v1"


@dataclass
class ExperimentConfig:
    """Cấu hình MỘT lần train DKT/GKT. Một nơi duy nhất cho mặc định của Cấp độ 1.

    Được ghi kèm checkpoint (``checkpoints_manifest.json``) và trả về trong ``result["config"]``.
    """

    model_type: str = "GKT"
    graph_type: str = "Transition"
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    hid_dim: int = 32
    emb_dim: int = 32
    dropout: float = 0.0
    seed: int = 42
    max_seq_len: int = 200
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    patience: int = 5

    @property
    def name(self) -> str:
        return self.model_type if self.model_type == "DKT" else f"GKT-{self.graph_type}"

    def replace(self, **changes) -> "ExperimentConfig":
        data = asdict(self)
        data.update(changes)
        return ExperimentConfig(**data)

    def to_dict(self) -> dict:
        return asdict(self)


_CONFIG_FIELDS = {f.name for f in fields(ExperimentConfig)}


def resolve_data_file(data_file: str) -> str:
    path = Path(data_file)
    if path.exists():
        return str(path)
    candidate = repo_root() / "GKT" / "data" / data_file
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"Cannot find data file: {data_file}")


def load_kt_data(
    data_file,
    batch_size=32,
    graph_type="Transition",
    model_type="GKT",
    train_ratio=0.6,
    val_ratio=0.2,
    seed=42,
    max_seq_len=200,
    min_seq_len=3,
    use_cuda=None,
):
    """Nạp CSV -> bundle ``{concept_num, graph, *_loader, data_path, data_version, data_stats}``."""
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    data_path = resolve_data_file(data_file)
    concept_num, graph, train_loader, val_loader, test_loader = load_dataset(
        data_path,
        batch_size,
        graph_type,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        shuffle=True,
        model_type=model_type,
        use_cuda=use_cuda,
        seed=seed,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
    )
    base = train_loader.dataset.dataset
    return {
        "concept_num": concept_num,
        "graph": graph,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "data_path": data_path,
        "data_version": DATA_VERSION,
        "data_stats": dict(getattr(base, "data_stats", {}) or {}),
    }


def count_parameters(model) -> int:
    """Số tham số huấn luyện được — báo cáo cạnh AUC để so sánh công bằng (mục 0.4)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _user_ids_from_subset(subset):
    base = subset.dataset
    user_ids = getattr(base, "user_ids", list(range(len(base))))
    return [as_id(user_ids[i]) for i in subset.indices]


def save_split(bundle, out_path: Path) -> dict:
    payload = {
        "train_users": _user_ids_from_subset(bundle["train_loader"].dataset),
        "val_users": _user_ids_from_subset(bundle["val_loader"].dataset),
        "test_users": _user_ids_from_subset(bundle["test_loader"].dataset),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_model(
    model_type,
    concept_num,
    graph=None,
    graph_type="Transition",
    hid_dim=32,
    emb_dim=32,
    dropout=0.0,
    edge_types=2,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "DKT":
        model = DKT(2 * concept_num, hid_dim, concept_num, dropout=dropout, bias=True)
    elif model_type == "GKT":
        graph_model = None
        if graph_type == "MHA":
            # MHA suy luận đồ thị động trong lúc chạy nên GKT yêu cầu graph=None.
            graph = None
            graph_model = MultiHeadAttention(
                edge_types, concept_num, emb_dim, emb_dim, dropout=dropout
            )
            if device.type == "cuda":
                graph_model = graph_model.cuda()
        elif graph is not None:
            graph = graph.to(device)
        model = GKT(
            concept_num,
            hid_dim,
            emb_dim,
            edge_types,
            graph_type,
            graph=graph,
            graph_model=graph_model,
            dropout=dropout,
            bias=True,
            binary=True,
            has_cuda=device.type == "cuda",
        )
    else:
        raise ValueError(model_type)
    return model.to(device)


def _forward(model, model_type, batch, device):
    features, questions, answers = batch
    features = features.to(device)
    questions = questions.to(device)
    answers = answers.to(device)
    out = model(features, questions)
    pred = out[0] if isinstance(out, tuple) else out
    return pred, answers


def _epoch_loop(model, loader, model_type, device, kt_loss, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses = []
    all_preds, all_labels = [], []
    n_batches = 0
    bad_loss_batches = 0
    bad_grad_batches = 0
    bad_preds = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            n_batches += 1
            pred, answers = _forward(model, model_type, batch, device)
            loss, _, _ = kt_loss(pred, answers)
            loss_finite = bool(torch.isfinite(loss).item())
            if train and loss_finite:
                optimizer.zero_grad()
                loss.backward()
                if all(
                    p.grad is None or bool(torch.isfinite(p.grad).all().item())
                    for p in model.parameters()
                ):
                    clip_grad_norm_(model.parameters(), 10.0)
                    optimizer.step()
                else:
                    # clip_grad_norm_ chia cho tổng norm; một gradient NaN sẽ biến MỌI tham số
                    # thành NaN. Bỏ bước cập nhật của batch hỏng để giữ trọng số hữu hạn.
                    bad_grad_batches += 1
                    optimizer.zero_grad(set_to_none=True)
            if loss_finite:
                losses.append(float(loss.detach().cpu()))
            else:
                bad_loss_batches += 1
            real = answers[:, 1:]
            mask = (real != -1) & (batch[1].to(device)[:, 1:] != -1)
            batch_pred = pred[mask].detach()
            batch_true = real[mask]
            keep = torch.isfinite(batch_pred)
            if not bool(keep.all().item()):
                bad_preds += int((~keep).sum().item())
                batch_pred = batch_pred[keep]
                batch_true = batch_true[keep]
            all_preds.append(batch_pred.cpu().numpy())
            all_labels.append(batch_true.cpu().numpy())
    if n_batches and bad_loss_batches == n_batches:
        raise RuntimeError(
            f"[{model_type}] toàn bộ {n_batches} batch có loss không hữu hạn — dừng run này "
            "thay vì lưu checkpoint rác. Kiểm tra dữ liệu vào (features/questions); nếu đang "
            "resume từ checkpoint cũ thì xoá ckpt_* của run đó rồi chạy lại."
        )
    if bad_loss_batches or bad_grad_batches or bad_preds:
        log.warning(
            f"  [cảnh báo NaN] {model_type} ({'train' if train else 'eval'}): "
            f"loss hỏng {bad_loss_batches}/{n_batches} batch, "
            f"gradient hỏng {bad_grad_batches} batch, "
            f"dự đoán hỏng {bad_preds} — các batch/vị trí này bị bỏ, trọng số giữ hữu hạn."
        )
    # Global AUC / Accuracy over the whole loader (not mean of per-batch AUCs)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    auc, acc = auc_acc(y_true, y_pred)
    return {
        "loss": float(np.mean(losses) if losses else 0.0),
        "auc": auc,
        "acc": acc,
        "batches": n_batches,
        "bad_loss_batches": bad_loss_batches,
        "bad_grad_batches": bad_grad_batches,
        "bad_preds": bad_preds,
    }


def fit(
    model,
    train_loader,
    val_loader,
    model_type: str = "GKT",
    *,
    epochs: int = 20,
    lr: float = 1e-3,
    device=None,
    patience: int = 5,
    ckpt_path=None,
    fingerprint: Optional[dict] = None,
    fingerprint_hint: str = "Xoá checkpoint cũ rồi chạy lại.",
    resume_state: Optional[dict] = None,
    deadline: Optional[float] = None,
    verbose: bool = True,
) -> dict:
    """Vòng epoch DUY NHẤT của DKT/GKT: early stop theo val AUC toàn cục, resume theo epoch.

    Dùng chung cho :func:`train_one` (không checkpoint) và
    ``level2.baselines.train_baseline_resumable`` (checkpoint mỗi epoch + dừng theo ``deadline``).

    Trả về ``{"history", "best_val_auc", "paused", "seconds"}``. ``paused=True`` nghĩa là đã hết
    ``deadline`` và checkpoint vừa được ghi — người gọi tự raise ``TrainingPaused``.
    """
    if device is None:
        device = next(model.parameters()).device
    kt_loss = KTLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    history = {k: [] for k in HISTORY_KEYS}
    best_auc, best_state, wait, epoch = -1.0, None, 0, 0
    if resume_state:
        check_fingerprint(resume_state.get("fingerprint"), fingerprint, fingerprint_hint)
        model.load_state_dict(resume_state["state_dict"])
        optimizer.load_state_dict(resume_state["optim"])
        optimizer_state_to_device(optimizer, device)
        epoch = int(resume_state.get("epoch", 0))
        best_auc = float(resume_state.get("best_auc", -1.0))
        best_state = resume_state.get("best_state")
        wait = int(resume_state.get("wait", 0))
        history = resume_state.get("history", history)
        if verbose:
            log.info(
                f"  [{model_type}] RESUME: đã xong {epoch}/{epochs} epoch "
                f"| best val AUC={best_auc:.4f} | chờ {wait}/{patience}"
            )

    def _save(ep: int) -> None:
        if ckpt_path is None:
            return
        atomic_save(
            {
                "format": RESUME_FORMAT,
                "state_dict": cpu_state_dict(model),
                "optim": optimizer_state_cpu(optimizer),
                "epoch": int(ep),
                "best_auc": float(best_auc),
                "best_state": best_state,
                "wait": int(wait),
                "history": history,
                "fingerprint": fingerprint or {},
            },
            ckpt_path,
        )

    t0, paused = time.time(), False
    last_epoch_s: Optional[float] = None
    while epoch < int(epochs):
        if deadline is not None and time.time() >= deadline:
            _save(epoch)
            paused = True
            break
        current = epoch + 1
        # Checkpoint chỉ ghi ở ranh giới epoch (một epoch Junyi đã hàng chục phút), nên nếu
        # epoch này không kịp xong thì dừng NGAY: Kaggle kill giữa epoch sẽ mất cả epoch đó.
        if (
            deadline is not None
            and last_epoch_s is not None
            and time.time() + last_epoch_s * 1.1 > deadline
        ):
            _save(epoch)
            paused = True
            if verbose:
                log.info(
                    f"  [{model_type}] dừng TRƯỚC epoch {current}: epoch trước mất "
                    f"{last_epoch_s / 60:.0f} phút, chỉ còn "
                    f"{max(0.0, deadline - time.time()) / 60:.0f} phút ngân sách — checkpoint "
                    f"đã ghi, phiên sau train tiếp từ epoch {epoch}"
                )
            break
        t_epoch = time.time()
        tr = _epoch_loop(model, train_loader, model_type, device, kt_loss, optimizer)
        va = _epoch_loop(model, val_loader, model_type, device, kt_loss, optimizer=None)
        last_epoch_s = time.time() - t_epoch
        for split, res in (("train", tr), ("val", va)):
            for metric in ("loss", "auc", "acc"):
                history[f"{split}_{metric}"].append(res[metric])
        improved = va["auc"] > best_auc
        if improved:
            best_auc = va["auc"]
            best_state = cpu_state_dict(model)
            wait = 0
        else:
            wait += 1
        epoch = current
        _save(epoch)
        if verbose:
            log.info(
                f"  [{model_type}] epoch {epoch:02d}/{epochs} "
                f"train loss={tr['loss']:.4f} auc={tr['auc']:.4f} "
                f"val loss={va['loss']:.4f} auc={va['auc']:.4f} acc={va['acc']:.4f}"
                + (" *" if improved else "")
            )
        if wait >= patience:
            if verbose:
                log.info(f"  [{model_type}] early stopping tại epoch {epoch}")
            break
    if not paused and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    elapsed = time.time() - t0
    if verbose:
        log.info(
            f"  [{model_type}] {'DỪNG theo ngân sách' if paused else 'xong'} sau "
            f"{elapsed:.1f}s | best val AUC={best_auc:.4f}"
        )
    return {
        "history": history,
        "best_val_auc": float(best_auc),
        "paused": paused,
        "seconds": elapsed,
    }


def train_one(
    model,
    train_loader,
    val_loader,
    model_type="GKT",
    epochs=20,
    lr=1e-3,
    device=None,
    patience=5,
):
    """Train không checkpoint; trả về ``(history, best_val_auc)`` như API cũ."""
    out = fit(
        model,
        train_loader,
        val_loader,
        model_type,
        epochs=epochs,
        lr=lr,
        device=device,
        patience=patience,
    )
    return out["history"], out["best_val_auc"]


def evaluate(model, loader, model_type="GKT", device=None):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    return _epoch_loop(model, loader, model_type, device, KTLoss(), optimizer=None)


def run_experiment(
    data_file,
    model_type="GKT",
    graph_type="Transition",
    epochs=20,
    batch_size=32,
    lr=1e-3,
    hid_dim=32,
    emb_dim=32,
    dropout=0.0,
    seed=42,
    max_seq_len=200,
    train_ratio=0.6,
    val_ratio=0.2,
    save_dir=None,
    bundle=None,
    device=None,
    config: Optional[ExperimentConfig] = None,
):
    """Train + test một mô hình. Truyền ``config=ExperimentConfig(...)`` hoặc từng kwargs như cũ.

    Khi có ``config``, mọi trường của nó thắng kwargs cùng tên.
    """
    if config is None:
        config = ExperimentConfig(
            model_type=model_type,
            graph_type=graph_type,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hid_dim=hid_dim,
            emb_dim=emb_dim,
            dropout=dropout,
            seed=seed,
            max_seq_len=max_seq_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
    cfg = config
    set_seed(cfg.seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bundle is None:
        bundle = load_kt_data(
            data_file,
            batch_size=cfg.batch_size,
            graph_type=cfg.graph_type if cfg.model_type == "GKT" else "Dense",
            model_type="GKT" if cfg.model_type == "GKT" else "DKT",
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
            max_seq_len=cfg.max_seq_len,
            use_cuda=device.type == "cuda",
        )
    model = build_model(
        cfg.model_type,
        bundle["concept_num"],
        graph=bundle.get("graph"),
        graph_type=cfg.graph_type,
        hid_dim=cfg.hid_dim,
        emb_dim=cfg.emb_dim,
        dropout=cfg.dropout,
        device=device,
    )
    history, best_val_auc = train_one(
        model,
        bundle["train_loader"],
        bundle["val_loader"],
        model_type=cfg.model_type,
        epochs=cfg.epochs,
        lr=cfg.lr,
        device=device,
        patience=cfg.patience,
    )
    test_metrics = evaluate(
        model, bundle["test_loader"], model_type=cfg.model_type, device=device
    )
    name = cfg.name
    result = {
        "name": name,
        "model_type": cfg.model_type,
        "graph_type": cfg.graph_type if cfg.model_type == "GKT" else "N/A",
        "concept_num": bundle["concept_num"],
        "best_val_auc": best_val_auc,
        "test_loss": test_metrics["loss"],
        "test_auc": test_metrics["auc"],
        "test_acc": test_metrics["acc"],
        "history": history,
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "n_params": count_parameters(model),
    }
    log.info(
        f">>> {name} test AUC={result['test_auc']:.4f} Acc={result['test_acc']:.4f} "
        f"Loss={result['test_loss']:.4f} | tham số={result['n_params']:,}"
    )
    if save_dir:
        # Tên file LUÔN mang seed (kể cả 42): quy ước cũ "seed 42 không hậu tố" từng để vòng
        # đa seed ghi đè checkpoint seed 42. Bên đọc dùng kt_common.checkpoint.find_checkpoint,
        # vẫn tìm được file tên cũ.
        ckpt = checkpoint_path(save_dir, name, cfg.seed)
        atomic_save(model.state_dict(), ckpt)
        record_checkpoint(
            ckpt,
            name=name,
            seed=cfg.seed,
            config=cfg.to_dict(),
            data_version=DATA_VERSION,
            data_path=str(bundle.get("data_path")),
            best_val_auc=best_val_auc,
            test_auc=result["test_auc"],
        )
        result["ckpt"] = str(ckpt)
    result["model"] = model.cpu()
    result["bundle"] = bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


#: Bộ mô hình mặc định của :func:`run_level1` — khớp ``TRAIN_MODELS`` của notebook Cấp độ 1
#: (trừ ``ShuffledRow``: đồ thị đó được dựng trong notebook, chưa có trong ``GKT/processing``).
DEFAULT_GRAPH_TYPES = ("Transition", "Dense", "Identity", "Zero", "Shuffled", "MHA")


def run_level1(
    data_file="skill_builder_data.csv",
    graph_types=DEFAULT_GRAPH_TYPES,
    seeds=(42,),
    save_dir=None,
    device=None,
    config: Optional[ExperimentConfig] = None,
    **overrides,
):
    """Entry-point không cần notebook (cũng là CLI ``python -m level1``).

    Train DKT + GKT với từng ``graph_types`` × ``seeds`` trên MỘT split cố định (``seed`` của
    split = ``config.seed``, chỉ seed khởi tạo đổi theo ``seeds``), ghi ``split.json``,
    checkpoint có hậu tố seed và ``metrics.json``. ``overrides`` là trường bất kỳ của
    :class:`ExperimentConfig` (``epochs=...``, ``max_seq_len=...``).
    """
    unknown = set(overrides) - _CONFIG_FIELDS
    if unknown:
        raise TypeError(f"run_level1: tham số lạ {sorted(unknown)}")
    base = (config or ExperimentConfig()).replace(**overrides)
    save_dir = Path(save_dir) if save_dir is not None else repo_root() / "outputs" / "level1"
    save_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    log.info(f"device={device} data={data_file} config={base.to_dict()}")

    def _bundle(graph_type, model_type="GKT"):
        # Split chỉ phụ thuộc split seed (base.seed), không phụ thuộc đồ thị (xem
        # level2.baselines.graph_bundle) nên mọi bundle có cùng người học train/val/test.
        return load_kt_data(
            data_file,
            batch_size=base.batch_size,
            graph_type=graph_type,
            model_type=model_type,
            train_ratio=base.train_ratio,
            val_ratio=base.val_ratio,
            seed=base.seed,
            max_seq_len=base.max_seq_len,
            use_cuda=use_cuda,
        )

    shared = _bundle("Transition")
    log.info(f"data_stats: {shared['data_stats']}")
    split = save_split(shared, save_dir / "split.json")
    log.info(
        f"split users train/val/test = {len(split['train_users'])}/{len(split['val_users'])}/{len(split['test_users'])}"
    )

    jobs = [(base.replace(model_type="DKT", graph_type="N/A"), shared)]
    for graph_type in graph_types:
        bundle = shared if graph_type in ("Transition", "MHA") else _bundle(graph_type)
        jobs.append((base.replace(model_type="GKT", graph_type=graph_type), bundle))

    results = []
    for cfg, bundle in jobs:
        for s in seeds:
            results.append(
                run_experiment(
                    data_file,
                    config=cfg.replace(seed=int(s)),
                    save_dir=save_dir,
                    bundle=bundle,
                    device=device,
                )
            )

    table = [
        {
            "model": r["name"],
            "graph": r["graph_type"],
            "seed": r["seed"],
            "n_params": r["n_params"],
            "test_auc": r["test_auc"],
            "test_acc": r["test_acc"],
            "test_loss": r["test_loss"],
            "best_val_auc": r["best_val_auc"],
        }
        for r in results
    ]
    (save_dir / "metrics.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    (save_dir / "run_config.json").write_text(
        json.dumps({"data_file": str(data_file), "config": base.to_dict(),
                    "data_stats": shared["data_stats"],
                    "graph_types": list(graph_types), "seeds": [int(s) for s in seeds]},
                   indent=2),
        encoding="utf-8",
    )
    return {
        "results": results,
        "table": table,
        "split": split,
        "save_dir": str(save_dir),
        "shared": shared,
    }
