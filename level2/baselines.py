"""Tái sử dụng DKT/GKT của Cấp độ 1 làm đường nền cho bảng so sánh Cấp độ 2.

Mục đích: bảng so sánh DKT / GKT / GRKT phải được đo trên CÙNG MỘT split và CÙNG MỘT file
chuẩn hoá màGRKT dùng ở Cấp độ 2. Vì vậy mô-đun này không "train lại từ đầu theo bài khác",
mà nạp checkpoint đã có trong ``outputs/level1`` (hoặc ``/kaggle/working/level1_out``), và
chỉ huấn luyện lại nếu checkpoint không khớp.

Hàm thu thập xác suất ``collect_level1_probs`` dùng lại đúng mặt nạ của
``level1.pipeline._epoch_loop`` nên số hiệu chỉnh (ECE/Brier) của Cấp độ 1 và Cấp độ 2 so
sánh được trực tiếp.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.optim as optim

from level1.pipeline import (
    _epoch_loop,
    _forward,
    _user_ids_from_subset,
    build_model,
    evaluate,
    load_kt_data,
)

# Cấp độ 1 train với hai giá trị này (kaggle_level1.ipynb, các cell run_experiment)
LEVEL1_HID_DIM = 32
LEVEL1_EMB_DIM = 32

LEVEL1_SPECS: Tuple[Tuple[str, str], ...] = (
    ("DKT", "N/A"),
    ("GKT", "Transition"),
    ("GKT", "Zero"),
)


def _name(model_type: str, graph_type: str) -> str:
    return model_type if model_type == "DKT" else f"GKT-{graph_type}"


def graph_bundle(
    shared: dict,
    graph_type: str,
    shared_graph_type: str = "Transition",
    batch_size: int = 32,
    max_seq_len: int = 200,
    min_seq_len: int = 3,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    device=None,
    verbose: bool = True,
) -> dict:
    """Nạp lại bundle với đồ thị khác nhưng CHẮC CHẮN giữ nguyên split của ``shared``.

    ``GKT/processing.load_dataset`` dựng đồ thị trước rồi mới ``random_split`` bằng generator
    có seed cố định, nên đổi ``graph_type`` không đổi split. Lời assert dưới đây biến giả định
    đó thành bằng chứng: nếu người dùng sau này sửa thứ tự trong pipeline, chỗ này sẽ fail
    thay vì âm thầm so sánh hai split khác nhau.

    ``shared_graph_type`` là kiểu đồ thị đã dùng khi nạp ``shared`` (mặc định "Transition"
    như trong ``kaggle_level1.ipynb``). Đường nền DKT không dùng đồ thị nên dùng thẳng
    ``shared``.

    ``min_seq_len`` / ``train_ratio`` / ``val_ratio`` không được lưu lại trong bundle nên
    người gọi PHẢI truyền đúng như lần nạp ``shared``; lệch một giá trị là lời assert bên
    dưới fail ngay thay vì âm thầm so sánh hai split khác nhau.
    """
    if graph_type in ("N/A", None) or graph_type == shared_graph_type:
        return shared
    bundle = load_kt_data(
        shared["data_path"],
        batch_size=batch_size,
        graph_type=graph_type,
        model_type="GKT",
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
        max_seq_len=max_seq_len,
        min_seq_len=min_seq_len,
        use_cuda=(
            device is not None and str(getattr(device, "type", device)) == "cuda"
        ),
    )
    for split in ("train", "val", "test"):
        a = _user_ids_from_subset(shared[f"{split}_loader"].dataset)
        b = _user_ids_from_subset(bundle[f"{split}_loader"].dataset)
        if a != b:
            raise AssertionError(
                f"Đổi graph_type={graph_type} làm thay đổi split ({split}) so với bundle gốc. "
                "Bảng so sánh Cấp độ 2 sẽ vô nghĩa — phải dựng đồ thị SAU khi cắt split."
            )
    if bundle["concept_num"] != shared["concept_num"]:
        raise AssertionError("Số khái niệm khác nhau giữa hai lần nạp dữ liệu")
    if verbose:
        print(
            f"  bundle[{graph_type}]: split khớp với bundle gốc, concept_num={bundle['concept_num']}"
        )
    return bundle


def load_level1_state(model, ckpt_path: Path, device) -> bool:
    """Nạp ``state_dict`` từ checkpoint. Trả về False nếu hình trạng không khớp."""
    if not ckpt_path or not Path(ckpt_path).exists():
        return False
    try:
        # weights_only=True: chỉ chấp nhận tensor, không thực thi pickle tuỳ ý
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
        return True
    except (
        Exception
    ) as exc:  # noqa: BLE001 - chủ đích: mọi lỗi khớp trọng số đều rơi về train lại
        print(
            f"  [checkpoint] {ckpt_path.name} không nạp được ({type(exc).__name__}: {exc}) -> train lại"
        )
        return False


def checkpoint_seed(ckpt_path) -> Optional[int]:
    """Seed đã train ra checkpoint, đọc từ tên file hoặc metadata; ``None`` nếu không rõ.

    Ba nguồn, theo thứ tự tin cậy: hậu tố ``-seed{s}.pt`` (do ``level1.pipeline.run_experiment``
    đặt cho seed ≠ 42), ``<tên>.json`` cạnh file (fingerprint ghi khi Cấp độ 2 tự train lại),
    rồi ``checkpoint_seeds.json`` trong cùng thư mục (khai báo thủ công cho checkpoint cũ).

    ``None`` phải được coi là "KHÔNG BIẾT", không phải 42: checkpoint Cấp độ 1 từng bị vòng
    đa seed ghi đè bằng seed cuối cùng (xem GAP_LEVEL2.md §F2).
    """
    if not ckpt_path:
        return None
    path = Path(ckpt_path)
    m = re.search(r"-seed(\d+)\.pt$", path.name)
    if m:
        return int(m.group(1))
    meta = path.with_suffix(".json")
    if meta.exists():
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        fingerprint = payload.get("fingerprint") if isinstance(payload, dict) else None
        if isinstance(fingerprint, dict) and fingerprint.get("seed") is not None:
            return int(fingerprint["seed"])
    sidecar = path.parent / "checkpoint_seeds.json"
    if sidecar.exists():
        try:
            mapping = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            mapping = {}
        if isinstance(mapping, dict):
            value = mapping.get(path.name, mapping.get(path.stem))
            if value is not None:
                return int(value)
    return None


def collect_level1_probs(
    model, loader, model_type: str, device=None
) -> Dict[str, object]:
    """(probs, labels) trên toàn bộ vị trí có nhãn của ``loader`` + AUC/ACC/loss."""
    from level1.paths import ensure_gkt_on_path

    ensure_gkt_on_path()
    from metrics import KTLoss

    if device is None:
        device = next(model.parameters()).device
    model.eval()
    kt_loss = KTLoss()
    preds, labels, losses = [], [], []
    with torch.no_grad():
        for batch in loader:
            pred, answers = _forward(model, model_type, batch, device)
            loss, _, _ = kt_loss(pred, answers)
            losses.append(float(loss.detach().cpu()))
            real = answers[:, 1:]
            mask = real != -1
            mask = mask & (batch[1].to(device)[:, 1:] != -1)
            preds.append(pred[mask].detach().cpu().numpy())
            labels.append(real[mask].detach().cpu().numpy())
    probs = np.concatenate(preds) if preds else np.array([])
    y = np.concatenate(labels) if labels else np.array([])
    from sklearn.metrics import roc_auc_score

    if y.size and len(np.unique(y)) > 1:
        auc = float(roc_auc_score(y, probs))
        acc = float(((probs >= 0.5) == (y >= 0.5)).mean())
    else:
        auc, acc = -1.0, -1.0
    return {
        "probs": probs,
        "labels": y.astype(float),
        "auc": auc,
        "acc": acc,
        "loss": float(np.mean(losses) if losses else 0.0),
    }


# ---------------------------------------------------------------- resume đường nền
#: Định danh payload "train dở" của đường nền (khác ``baselines/<tên>.pt`` chỉ chứa trọng số).
BASELINE_RESUME_FORMAT = "level1-resume-v1"
HISTORY_KEYS = (
    "train_loss",
    "val_loss",
    "train_auc",
    "val_auc",
    "train_acc",
    "val_acc",
)


def _atomic_save(payload, path: Path) -> None:
    """File tạm + ``os.replace``: session bị kill giữa lúc ghi không làm hỏng checkpoint cũ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _cpu_state(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _optim_cpu(optimizer) -> dict:
    state = optimizer.state_dict()
    return {
        "state": {
            i: {
                k: (v.detach().cpu().clone() if torch.is_tensor(v) else v)
                for k, v in entry.items()
            }
            for i, entry in state["state"].items()
        },
        "param_groups": state["param_groups"],
    }


def _optim_to_device(optimizer, device) -> None:
    for entry in optimizer.state.values():
        for k, v in entry.items():
            if torch.is_tensor(v):
                entry[k] = v.to(device)


def baseline_fingerprint(
    shared,
    model_type,
    graph_type,
    *,
    batch_size,
    max_seq_len,
    min_seq_len,
    train_ratio,
    val_ratio,
    seed,
) -> dict:
    """Cấu hình làm thay đổi ý nghĩa trọng số đường nền, ghi kèm checkpoint.

    Không có nó, ``baselines/DKT.pt`` sinh từ split khác vẫn nạp được (cùng shape) và bảng so
    sánh trộn hai split — đúng loại lỗi ``_assert_align`` của GRKT chặn bên nhánh GRKT.
    """
    counts = {
        s: len(_user_ids_from_subset(shared[f"{s}_loader"].dataset))
        for s in ("train", "val", "test")
    }
    return {
        "data_path": str(shared["data_path"]),
        "concept_num": int(shared["concept_num"]),
        "model_type": model_type,
        "graph_type": graph_type,
        "n_users": counts,
        "batch_size": int(batch_size),
        "max_seq_len": int(max_seq_len),
        "min_seq_len": int(min_seq_len),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "seed": int(seed),
    }


def _check_baseline_fingerprint(
    previous: Optional[dict], current: Optional[dict], name: str
) -> None:
    if not previous or not current:
        return
    diff = {k: (previous.get(k), v) for k, v in current.items() if previous.get(k) != v}
    if diff:
        detail = ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in diff.items())
        raise RuntimeError(
            f"Checkpoint đường nền {name} được tạo với cấu hình khác ({detail}). Xoá "
            f"baselines/{name}.pt, baselines/{name}.json và baselines/ckpt_{name}.pt rồi chạy "
            "lại — trộn hai split/cấu hình sẽ làm hỏng bảng so sánh."
        )


def load_retrained_baseline(
    model, pt_path: Path, fingerprint: Optional[dict], device
) -> bool:
    """Nạp ``baselines/<tên>.pt`` do lần chạy trước train (khác checkpoint Cấp độ 1)."""
    pt_path = Path(pt_path)
    if not pt_path.exists():
        return False
    meta_path = pt_path.with_suffix(".json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _check_baseline_fingerprint(meta.get("fingerprint"), fingerprint, pt_path.stem)
    else:
        print(
            f"  [cảnh báo] {pt_path.name} không có {meta_path.name}: không kiểm chứng được "
            "split/cấu hình đã train — chỉ nạp khi chắc chắn nó cùng split."
        )
    state = torch.load(pt_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    return True


def train_baseline_resumable(
    model,
    bundle: dict,
    model_type: str,
    *,
    epochs: int,
    lr: float,
    device,
    ckpt_path: Optional[Path],
    fingerprint: Optional[dict],
    resume_state: Optional[dict] = None,
    deadline: Optional[float] = None,
    patience: int = 5,
    verbose: bool = True,
) -> Tuple[dict, float, bool]:
    """Như ``level1.pipeline.train_one`` nhưng ghi checkpoint cuối MỖI epoch và dừng theo giờ.

    Đường nền phải train lại (Junyi) chạy 20 epoch × 3 mô hình trên 16,2 triệu lượt — dài hơn
    một phiên Kaggle. Checkpoint theo epoch (không theo batch: một epoch đã hàng chục phút) đủ
    để phiên sau mất tối đa một epoch mỗi mô hình.

    Trả về ``(history, best_val_auc, paused)``; ``paused=True`` nghĩa là đã hết ``deadline`` và
    checkpoint vừa được ghi — người gọi tự raise ``TrainingPaused``.
    """
    from level1.paths import ensure_gkt_on_path

    ensure_gkt_on_path()
    from metrics import KTLoss

    kt_loss = KTLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    history = {k: [] for k in HISTORY_KEYS}
    best_auc, best_state, wait, epoch = -1.0, None, 0, 0
    if resume_state:
        _check_baseline_fingerprint(
            resume_state.get("fingerprint"), fingerprint, model_type
        )
        model.load_state_dict(resume_state["state_dict"])
        optimizer.load_state_dict(resume_state["optim"])
        _optim_to_device(optimizer, device)
        epoch = int(resume_state.get("epoch", 0))
        best_auc = float(resume_state.get("best_auc", -1.0))
        best_state = resume_state.get("best_state")
        wait = int(resume_state.get("wait", 0))
        history = resume_state.get("history", history)
        if verbose:
            print(
                f"  [{model_type}] RESUME: đã xong {epoch}/{epochs} epoch "
                f"| best val AUC={best_auc:.4f} | chờ {wait}/{patience}"
            )

    def _save(ep: int) -> None:
        if ckpt_path is None:
            return
        _atomic_save(
            {
                "format": BASELINE_RESUME_FORMAT,
                "state_dict": _cpu_state(model),
                "optim": _optim_cpu(optimizer),
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
                print(
                    f"  [{model_type}] dừng TRƯỚC epoch {current}: epoch trước mất "
                    f"{last_epoch_s / 60:.0f} phút, chỉ còn "
                    f"{max(0.0, deadline - time.time()) / 60:.0f} phút ngân sách — checkpoint "
                    f"đã ghi, phiên sau train tiếp từ epoch {epoch}"
                )
            break
        t_epoch = time.time()
        tr = _epoch_loop(
            model, bundle["train_loader"], model_type, device, kt_loss, optimizer
        )
        va = _epoch_loop(
            model, bundle["val_loader"], model_type, device, kt_loss, optimizer=None
        )
        last_epoch_s = time.time() - t_epoch
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_auc"].append(tr["auc"])
        history["val_auc"].append(va["auc"])
        history["train_acc"].append(tr["acc"])
        history["val_acc"].append(va["acc"])
        improved = va["auc"] > best_auc
        if improved:
            best_auc = va["auc"]
            best_state = _cpu_state(model)
            wait = 0
        else:
            wait += 1
        epoch = current
        _save(epoch)
        if verbose:
            print(
                f"  [{model_type}] epoch {epoch:02d}/{epochs} "
                f"train loss={tr['loss']:.4f} auc={tr['auc']:.4f} "
                f"val loss={va['loss']:.4f} auc={va['auc']:.4f} acc={va['acc']:.4f}"
                + (" *" if improved else "")
            )
        if wait >= patience:
            if verbose:
                print(f"  [{model_type}] early stopping tại epoch {epoch}")
            break
    if not paused and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    if verbose:
        print(
            f"  [{model_type}] {'DỪNG theo ngân sách' if paused else 'xong'} sau "
            f"{time.time() - t0:.1f}s | best val AUC={best_auc:.4f}"
        )
    return history, float(best_auc), paused


def level1_baseline(
    shared: dict,
    specs: Sequence[Tuple[str, str]] = LEVEL1_SPECS,
    ckpt_dir=None,
    epochs: int = 20,
    batch_size: int = 32,
    max_seq_len: int = 200,
    min_seq_len: int = 3,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    lr: float = 1e-3,
    seed: int = 42,
    device=None,
    allow_retrain: bool = True,
    collect_probs: bool = True,
    shared_graph_type: str = "Transition",
    save_dir=None,
    deadline: Optional[float] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Đánh giá (và nếu cần, huấn luyện lại) các đường nền Cấp độ 1 trên bundle đang dùng.

    Trả về ``{"results": [...], "probs": {tên: (probs, labels)}}``; mỗi phần tử của
    ``results`` có ĐÚNG bộ khoá của ``level1.pipeline.run_experiment`` để notebook nối
    thẳng vào bảng so sánh với GRKT.

    ``ckpt_dir`` là nơi ĐỌC checkpoint của Cấp độ 1; ``save_dir`` là nơi GHI trọng số vừa
    huấn luyện lại (thường là thư mục kết quả của tập dữ liệu đang chạy). Không bao giờ ghi
    vào ``ckpt_dir`` — tên file trùng với sản phẩm Cấp độ 1 (``DKT.pt``), ghi đè là mất.

    ``max_seq_len`` / ``min_seq_len`` / ``train_ratio`` / ``val_ratio`` phải trùng với cấu
    hình đã sinh ra ``shared`` (xem docstring của :func:`graph_bundle`).

    Resume: khi phải train lại, trọng số ghi vào ``save_dir/baselines/<tên>.pt`` kèm
    ``<tên>.json`` (fingerprint split/cấu hình) và checkpoint giữa chừng
    ``ckpt_<tên>.pt``. Lần chạy sau nạp lại thay vì train lại; hết ``deadline`` thì ghi
    checkpoint rồi raise :class:`~level2.grkt_api.TrainingPaused`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
    results: List[dict] = []
    probs: Dict[str, tuple] = {}

    for model_type, graph_type in specs:
        name = _name(model_type, graph_type)
        bundle = graph_bundle(
            shared,
            graph_type,
            shared_graph_type=shared_graph_type,
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            min_seq_len=min_seq_len,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            device=device,
            verbose=verbose,
        )
        model = build_model(
            model_type,
            bundle["concept_num"],
            graph=bundle.get("graph"),
            graph_type=graph_type if model_type == "GKT" else "Transition",
            hid_dim=LEVEL1_HID_DIM,
            emb_dim=LEVEL1_EMB_DIM,
            dropout=0.0,
            device=device,
        )
        ckpt = (ckpt_dir / f"{name}.pt") if ckpt_dir else None
        base_dir = (Path(save_dir) / "baselines") if save_dir is not None else None
        final_ckpt = (base_dir / f"{name}.pt") if base_dir is not None else None
        rolling_ckpt = (base_dir / f"ckpt_{name}.pt") if base_dir is not None else None
        fingerprint = (
            None
            if base_dir is None
            else baseline_fingerprint(
                bundle,
                model_type,
                graph_type,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                min_seq_len=min_seq_len,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                seed=seed,
            )
        )
        loaded_from = None
        if ckpt and load_level1_state(model, ckpt, device):
            loaded_from = ckpt
        elif final_ckpt is not None and load_retrained_baseline(
            model, final_ckpt, fingerprint, device
        ):
            loaded_from = final_ckpt
        loaded = loaded_from is not None
        ckpt_seed = None
        resume_state = None
        if not loaded and rolling_ckpt is not None and rolling_ckpt.exists():
            payload = torch.load(rolling_ckpt, map_location="cpu", weights_only=False)
            if (
                isinstance(payload, dict)
                and payload.get("format") == BASELINE_RESUME_FORMAT
            ):
                resume_state = payload
        if loaded:
            ckpt_seed = checkpoint_seed(loaded_from)
            if ckpt_seed is None:
                print(
                    f"  [checkpoint-seed] {name}: KHÔNG xác định được seed của "
                    f"{loaded_from.name} -> bảng ghi seed=None. Sửa bằng cách train lại "
                    "đường nền (baseline_ckpt_dir=None) hoặc khai báo checkpoint_seeds.json."
                )
            elif int(ckpt_seed) != int(seed):
                print(
                    f"  [checkpoint-seed] {name}: checkpoint là seed {ckpt_seed}, KHÁC "
                    f"split_seed={seed} -> dòng này không cùng seed khởi tạo với GRKT; "
                    "phải ghi rõ trong báo cáo."
                )
            if verbose:
                print(f"  [baseline] {name}: dùng checkpoint {loaded_from}")
            history, best_val_auc = {}, None
        else:
            if not allow_retrain:
                if verbose:
                    print(
                        f"  [baseline] {name}: bỏ qua (không có checkpoint, allow_retrain=False)"
                    )
                continue
            if resume_state is not None:
                print(
                    f"  [baseline] {name}: tiếp tục từ {rolling_ckpt.name} "
                    f"(epoch {resume_state.get('epoch')}/{epochs})"
                )
            else:
                print(
                    f"  [baseline] {name}: không có checkpoint -> huấn luyện lại {epochs} epoch"
                )
            history, best_val_auc, paused = train_baseline_resumable(
                model,
                bundle,
                model_type,
                epochs=epochs,
                lr=lr,
                device=device,
                ckpt_path=rolling_ckpt,
                fingerprint=fingerprint,
                resume_state=resume_state,
                deadline=deadline,
                verbose=verbose,
            )
            if paused:
                from .grkt_api import TrainingPaused

                raise TrainingPaused(
                    f"[baseline {name}] hết ngân sách thời gian: đã ghi {rolling_ckpt}. "
                    "Phiên sau giữ nguyên zip resume + `RESUME_GRKT_CKPT=True` để chạy tiếp "
                    "đúng epoch còn lại (đường nền cũng resume theo epoch)."
                )
            if final_ckpt is not None:
                # Ghi vào thư mục Cấp độ 2, TUYỆT ĐỐI không ghi đè ckpt_dir: thư mục đó là
                # sản phẩm của Cấp độ 1 (và tên file trùng khớp, vd DKT.pt).
                _atomic_save(_cpu_state(model), final_ckpt)
                final_ckpt.with_suffix(".json").write_text(
                    json.dumps(
                        {
                            "fingerprint": fingerprint,
                            "best_val_auc": best_val_auc,
                            "epochs": int(epochs),
                        },
                        ensure_ascii=False,
                        indent=1,
                    ),
                    encoding="utf-8",
                )
                if rolling_ckpt is not None and rolling_ckpt.exists():
                    rolling_ckpt.unlink()
            ckpt_seed = int(seed)  # trọng số vừa train trong lần chạy này
        if collect_probs:
            got = collect_level1_probs(
                model, bundle["test_loader"], model_type, device=device
            )
            test_metrics = {"auc": got["auc"], "acc": got["acc"], "loss": got["loss"]}
        else:
            got = None
            test_metrics = evaluate(
                model, bundle["test_loader"], model_type=model_type, device=device
            )
        res = {
            "name": name,
            "model_type": model_type,
            "graph_type": graph_type if model_type == "GKT" else "N/A",
            "concept_num": bundle["concept_num"],
            "best_val_auc": best_val_auc,
            "test_loss": test_metrics["loss"],
            "test_auc": test_metrics["auc"],
            "test_acc": test_metrics["acc"],
            "history": history,
            # ``seed`` = seed đã TRAIN ra trọng số đang được đánh giá (không phải split seed).
            # ``None`` = checkpoint cũ không ghi seed — bảng phải để trống thay vì đoán 42.
            "seed": int(ckpt_seed) if ckpt_seed is not None else None,
            "split_seed": int(seed),
            "ckpt_seed": ckpt_seed,
            "ckpt_source": str(loaded_from) if loaded_from is not None else None,
            "from_checkpoint": bool(loaded),
        }
        if got is not None:
            res["test_n"] = int(got["labels"].size)
            probs[name] = (got["probs"], got["labels"])
        results.append(res)
        print(
            f">>> {name} test AUC={res['test_auc']:.4f} Acc={res['test_acc']:.4f} "
            f"Loss={res['test_loss']:.4f} (checkpoint={'có' if loaded else 'train lại'})"
        )
        model.cpu()
    return {"results": results, "probs": probs}
