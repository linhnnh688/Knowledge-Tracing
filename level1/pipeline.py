"""Callable training/eval API for cap do 1. The Kaggle notebook only orchestrates these functions."""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_

from .paths import ensure_gkt_on_path, repo_root


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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
    ensure_gkt_on_path()
    from processing import load_dataset

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
    return {
        "concept_num": concept_num,
        "graph": graph,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "data_path": data_path,
    }


def _as_id(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _user_ids_from_subset(subset):
    base = subset.dataset
    user_ids = getattr(base, "user_ids", list(range(len(base))))
    return [_as_id(user_ids[i]) for i in subset.indices]


def save_split(bundle, out_path: Path) -> None:
    payload = {
        "train_users": _user_ids_from_subset(bundle["train_loader"].dataset),
        "val_users": _user_ids_from_subset(bundle["val_loader"].dataset),
        "test_users": _user_ids_from_subset(bundle["test_loader"].dataset),
    }
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
    ensure_gkt_on_path()
    from models import DKT, GKT

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "DKT":
        model = DKT(2 * concept_num, hid_dim, concept_num, dropout=dropout, bias=True)
    elif model_type == "GKT":
        if graph is not None:
            graph = graph.to(device)
        model = GKT(
            concept_num,
            hid_dim,
            emb_dim,
            edge_types,
            graph_type,
            graph=graph,
            graph_model=None,
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
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch in loader:
            pred, answers = _forward(model, model_type, batch, device)
            loss, _, _ = kt_loss(pred, answers)
            if train:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            real = answers[:, 1:]
            mask = real != -1
            if model_type == "GIKT":
                questions = batch[0].to(device)
                mask = mask & (questions[:, 1:] > 0)
            else:
                mask = mask & (batch[1].to(device)[:, 1:] != -1)
            all_preds.append(pred[mask].detach().cpu().numpy())
            all_labels.append(real[mask].detach().cpu().numpy())
    # Global AUC / Accuracy over the whole loader (not mean of per-batch AUCs)
    y_pred = np.concatenate(all_preds) if all_preds else np.array([])
    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    if y_true.size > 0 and len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_pred))
        acc = float(((y_pred >= 0.5) == y_true).mean())
    else:
        auc, acc = -1.0, -1.0
    return {
        "loss": float(np.mean(losses) if losses else 0.0),
        "auc": auc,
        "acc": acc,
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
    ensure_gkt_on_path()
    from metrics import KTLoss

    if device is None:
        device = next(model.parameters()).device
    kt_loss = KTLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {
        k: []
        for k in [
            "train_loss",
            "val_loss",
            "train_auc",
            "val_auc",
            "train_acc",
            "val_acc",
        ]
    }
    best_auc = -1.0
    best_state = None
    wait = 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = _epoch_loop(model, train_loader, model_type, device, kt_loss, optimizer)
        va = _epoch_loop(model, val_loader, model_type, device, kt_loss, optimizer=None)
        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_auc"].append(tr["auc"])
        history["val_auc"].append(va["auc"])
        history["train_acc"].append(tr["acc"])
        history["val_acc"].append(va["acc"])
        improved = va["auc"] > best_auc
        if improved:
            best_auc = va["auc"]
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
        print(
            f"  [{model_type}] epoch {epoch:02d}/{epochs} "
            f"train loss={tr['loss']:.4f} auc={tr['auc']:.4f} "
            f"val loss={va['loss']:.4f} auc={va['auc']:.4f} acc={va['acc']:.4f}"
            + (" *" if improved else "")
        )
        if wait >= patience:
            print(f"  early stopping at epoch {epoch}")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
    print(f"  done in {time.time() - t0:.1f}s | best val AUC={best_auc:.4f}")
    return history, best_auc


def evaluate(model, loader, model_type="GKT", device=None):
    ensure_gkt_on_path()
    from metrics import KTLoss

    if device is None:
        device = next(model.parameters()).device
    kt_loss = KTLoss()
    model.eval()
    return _epoch_loop(model, loader, model_type, device, kt_loss, optimizer=None)


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
):
    set_seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if bundle is None:
        bundle = load_kt_data(
            data_file,
            batch_size=batch_size,
            graph_type=graph_type if model_type == "GKT" else "Dense",
            model_type="GKT" if model_type == "GKT" else "DKT",
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            max_seq_len=max_seq_len,
            use_cuda=device.type == "cuda",
        )
    model = build_model(
        model_type,
        bundle["concept_num"],
        graph=bundle.get("graph"),
        graph_type=graph_type,
        hid_dim=hid_dim,
        emb_dim=emb_dim,
        dropout=dropout,
        device=device,
    )
    history, best_val_auc = train_one(
        model,
        bundle["train_loader"],
        bundle["val_loader"],
        model_type=model_type,
        epochs=epochs,
        lr=lr,
        device=device,
    )
    test_metrics = evaluate(
        model, bundle["test_loader"], model_type=model_type, device=device
    )
    name = model_type if model_type == "DKT" else f"GKT-{graph_type}"
    result = {
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
    }
    print(
        f">>> {name} test AUC={result['test_auc']:.4f} Acc={result['test_acc']:.4f} Loss={result['test_loss']:.4f}"
    )
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        ckpt = save_dir / f"{name}.pt"
        torch.save(model.state_dict(), ckpt)
        result["ckpt"] = str(ckpt)
    result["model"] = model.cpu()
    result["bundle"] = bundle
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_gikt_experiment(*args, **kwargs):
    raise NotImplementedError(
        "GIKT đã được loại bỏ khỏi pipeline. Hệ thống tập trung chuẩn hóa hoàn toàn trên nền tảng GKT & DKT."
    )


def run_level1(
    data_file="assistment_test15.csv",
    epochs=20,
    batch_size=32,
    lr=1e-3,
    hid_dim=32,
    emb_dim=32,
    seed=42,
    max_seq_len=200,
    save_dir=None,
    device=None,
):
    """Train DKT, GKT-Transition, GKT graph variants & ablations (Dense, Identity, Zero)."""
    if save_dir is None:
        save_dir = repo_root() / "outputs" / "level1"
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device={device} data={data_file} epochs={epochs} max_seq_len={max_seq_len}")
    shared = load_kt_data(
        data_file,
        batch_size=batch_size,
        graph_type="Transition",
        model_type="GKT",
        seed=seed,
        max_seq_len=max_seq_len,
        use_cuda=device.type == "cuda",
    )
    split = save_split(shared, save_dir / "split.json")
    print(
        f"split users train/val/test = {len(split['train_users'])}/{len(split['val_users'])}/{len(split['test_users'])}"
    )

    results = []
    results.append(
        run_experiment(
            data_file,
            model_type="DKT",
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hid_dim=hid_dim,
            emb_dim=emb_dim,
            seed=seed,
            max_seq_len=max_seq_len,
            save_dir=save_dir,
            bundle=shared,
            device=device,
        )
    )
    for graph_type in ["Transition", "Dense", "Identity", "Zero"]:
        bundle = load_kt_data(
            data_file,
            batch_size=batch_size,
            graph_type=graph_type,
            model_type="GKT",
            seed=seed,
            max_seq_len=max_seq_len,
            use_cuda=device.type == "cuda",
        )
        results.append(
            run_experiment(
                data_file,
                model_type="GKT",
                graph_type=graph_type,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                hid_dim=hid_dim,
                emb_dim=emb_dim,
                seed=seed,
                max_seq_len=max_seq_len,
                save_dir=save_dir,
                bundle=bundle,
                device=device,
            )
        )

    table = [
        {
            "model": r["name"],
            "graph": r["graph_type"],
            "test_auc": r["test_auc"],
            "test_acc": r["test_acc"],
            "test_loss": r["test_loss"],
            "best_val_auc": r["best_val_auc"],
        }
        for r in results
    ]
    (save_dir / "metrics.json").write_text(
        json.dumps(table, indent=2), encoding="utf-8"
    )
    return {
        "results": results,
        "table": table,
        "split": split,
        "save_dir": str(save_dir),
        "shared": shared,
    }
