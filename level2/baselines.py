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

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from level1.pipeline import (
    _forward,
    _user_ids_from_subset,
    build_model,
    evaluate,
    load_kt_data,
    train_one,
)

# Cấp độ 1 train với hai giá trị này (kaggle_level1.ipynb, các cell run_experiment)
LEVEL1_HID_DIM = 32
LEVEL1_EMB_DIM = 32

LEVEL1_SPECS: Tuple[Tuple[str, str], ...] = (("DKT", "N/A"), ("GKT", "Transition"), ("GKT", "Zero"))


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
        use_cuda=(device is not None and str(getattr(device, "type", device)) == "cuda"),
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
        print(f"  bundle[{graph_type}]: split khớp với bundle gốc, concept_num={bundle['concept_num']}")
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
    except Exception as exc:  # noqa: BLE001 - chủ đích: mọi lỗi khớp trọng số đều rơi về train lại
        print(f"  [checkpoint] {ckpt_path.name} không nạp được ({type(exc).__name__}: {exc}) -> train lại")
        return False


def collect_level1_probs(model, loader, model_type: str, device=None) -> Dict[str, object]:
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
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
    results: List[dict] = []
    probs: Dict[str, tuple] = {}

    for model_type, graph_type in specs:
        name = _name(model_type, graph_type)
        bundle = graph_bundle(shared, graph_type, shared_graph_type=shared_graph_type,
                              batch_size=batch_size, max_seq_len=max_seq_len,
                              min_seq_len=min_seq_len, train_ratio=train_ratio,
                              val_ratio=val_ratio, seed=seed, device=device, verbose=verbose)
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
        loaded = load_level1_state(model, ckpt, device) if ckpt else False
        if loaded:
            if verbose:
                print(f"  [baseline] {name}: dùng checkpoint {ckpt}")
            history, best_val_auc = {}, None
        else:
            if not allow_retrain:
                if verbose:
                    print(f"  [baseline] {name}: bỏ qua (không có checkpoint, allow_retrain=False)")
                continue
            print(f"  [baseline] {name}: không có checkpoint -> huấn luyện lại {epochs} epoch")
            history, best_val_auc = train_one(
                model, bundle["train_loader"], bundle["val_loader"],
                model_type=model_type, epochs=epochs, lr=lr, device=device,
            )
            if save_dir is not None:
                # Ghi vào thư mục Cấp độ 2, TUYỆT ĐỐI không ghi đè ckpt_dir: thư mục đó là
                # sản phẩm của Cấp độ 1 (và tên file Trùng khớp, vd DKT.pt).
                wd = Path(save_dir) / "baselines"
                wd.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), wd / f"{name}.pt")
        if collect_probs:
            got = collect_level1_probs(model, bundle["test_loader"], model_type, device=device)
            test_metrics = {"auc": got["auc"], "acc": got["acc"], "loss": got["loss"]}
        else:
            got = None
            test_metrics = evaluate(model, bundle["test_loader"], model_type=model_type, device=device)
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
            "seed": seed,
            "from_checkpoint": bool(loaded),
        }
        if got is not None:
            res["test_n"] = int(got["labels"].size)
            probs[name] = (got["probs"], got["labels"])
        results.append(res)
        print(f">>> {name} test AUC={res['test_auc']:.4f} Acc={res['test_acc']:.4f} "
              f"Loss={res['test_loss']:.4f} (checkpoint={'có' if loaded else 'train lại'})")
        model.cpu()
    return {"results": results, "probs": probs}
