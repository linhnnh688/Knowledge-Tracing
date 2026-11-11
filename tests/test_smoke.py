"""Smoke test đầu-cuối trên CPU (file 15 người học): train, lưu, nạp lại, resume, Level 2.

Test có TRAIN gắn marker ``train`` và bị bỏ qua mặc định (train chỉ chạy trên Kaggle);
chạy bằng ``pytest -m train`` khi cần.
"""

import time

import numpy as np
import pytest
import torch

from tests.conftest import SMOKE_CSV


@pytest.mark.train
def test_run_experiment_saves_seeded_checkpoint_and_reloads(tmp_path, smoke_bundle, cpu):
    from level1.pipeline import ExperimentConfig, run_experiment
    from level2.baselines import level1_baseline

    cfg = ExperimentConfig(model_type="DKT", graph_type="N/A", epochs=1, batch_size=4, max_seq_len=80)
    r = run_experiment(str(SMOKE_CSV), config=cfg, save_dir=tmp_path, bundle=smoke_bundle, device=cpu)
    assert np.isfinite(r["test_loss"]) and r["config"]["epochs"] == 1
    assert (tmp_path / "DKT-seed42.pt").exists()
    assert not (tmp_path / "DKT.pt").exists()
    # Level 2 nạp lại checkpoint Level 1 (tên mới) mà không train lại.
    got = level1_baseline(
        smoke_bundle, specs=(("DKT", "N/A"),), ckpt_dir=tmp_path, epochs=1, batch_size=4,
        max_seq_len=80, seed=42, device=cpu, allow_retrain=False, verbose=False,
    )
    row = got["results"][0]
    assert row["from_checkpoint"] and row["ckpt_seed"] == 42
    assert row["test_auc"] == r["test_auc"]


def test_level1_baseline_refuses_checkpoint_without_data_version(tmp_path, smoke_bundle, cpu):
    """Checkpoint cũ (không manifest/data_version) KHÔNG được nạp dù cùng shape."""
    from level1.pipeline import build_model
    from level2.baselines import level1_baseline

    model = build_model("DKT", smoke_bundle["concept_num"], device=cpu)
    torch.save(model.state_dict(), tmp_path / "DKT.pt")
    got = level1_baseline(smoke_bundle, specs=(("DKT", "N/A"),), ckpt_dir=tmp_path, batch_size=4,
                          max_seq_len=80, device=cpu, allow_retrain=False, verbose=False)
    assert got["results"] == []  # không nạp; allow_retrain=False -> bỏ qua


def test_level1_baseline_loads_current_checkpoint(tmp_path, smoke_bundle, cpu):
    from kt_common.checkpoint import checkpoint_path, record_checkpoint
    from kt_common.interactions import DATA_VERSION
    from level1.pipeline import build_model
    from level2.baselines import level1_baseline

    model = build_model("DKT", smoke_bundle["concept_num"], device=cpu)
    path = checkpoint_path(tmp_path, "DKT", 42)
    torch.save(model.state_dict(), path)
    record_checkpoint(path, name="DKT", data_version=DATA_VERSION)
    got = level1_baseline(smoke_bundle, specs=(("DKT", "N/A"),), ckpt_dir=tmp_path, batch_size=4,
                          max_seq_len=80, device=cpu, allow_retrain=False, verbose=False)
    row = got["results"][0]
    assert row["from_checkpoint"] and row["ckpt_seed"] == 42


@pytest.mark.train
def test_fit_pauses_on_deadline_and_resumes(tmp_path, smoke_bundle, cpu):
    from level1.pipeline import RESUME_FORMAT, build_model, fit, set_seed

    set_seed(1)
    model = build_model("DKT", smoke_bundle["concept_num"], device=cpu)
    ckpt = tmp_path / "ckpt_DKT-seed1.pt"
    out = fit(model, smoke_bundle["train_loader"], smoke_bundle["val_loader"], "DKT", epochs=2,
              device=cpu, ckpt_path=ckpt, fingerprint={"seed": 1}, deadline=time.time() - 1,
              verbose=False)
    assert out["paused"] and ckpt.exists()
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert state["format"] == RESUME_FORMAT and state["epoch"] == 0
    out2 = fit(model, smoke_bundle["train_loader"], smoke_bundle["val_loader"], "DKT", epochs=2,
               device=cpu, ckpt_path=ckpt, fingerprint={"seed": 1}, resume_state=state, verbose=False)
    assert not out2["paused"] and len(out2["history"]["val_auc"]) == 2
    # Resume với cấu hình khác phải bị chặn.
    with pytest.raises(RuntimeError):
        fit(model, smoke_bundle["train_loader"], smoke_bundle["val_loader"], "DKT", epochs=2,
            device=cpu, fingerprint={"seed": 2}, resume_state=state, verbose=False)


@pytest.mark.train
def test_level2_run_dataset_smoke(tmp_path, cpu):
    from level2.experiment import Level2Config, run_dataset

    res = run_dataset(
        "assist2009_smoke", device=cpu,
        config=Level2Config(save_root=str(tmp_path), epochs=1, batch_size=4, max_seq_len=40,
                            d_hidden=8, k_hidden=4, cold_thresh=1, graph_thresh=0.1,
                            baseline_epochs=1, trajectory_users=2, trajectory_batch=2,
                            verbose=False),
    )
    d = tmp_path / "assist2009_smoke"
    assert (d / "GRKT-train-seed42.pt").exists()
    assert (d / "baselines" / "DKT-seed42.pt").exists()
    assert set(res["table"]["model"]) == {"DKT", "GKT-Transition", "GKT-Zero", "GRKT-train"}
    assert res["table"]["auc"].between(0, 1).all()
    # Chạy lại với resume: GRKT được nạp (không train), cùng số liệu.
    res2 = run_dataset(
        "assist2009_smoke", device=cpu,
        config=Level2Config(save_root=str(tmp_path), epochs=1, batch_size=4, max_seq_len=40,
                            d_hidden=8, k_hidden=4, cold_thresh=1, graph_thresh=0.1,
                            baseline_epochs=1, record_trajectory=False, resume_grkt_ckpt=True,
                            verbose=False),
    )
    g1 = res["table"].set_index("model").loc["GRKT-train"]
    g2 = res2["table"].set_index("model").loc["GRKT-train"]
    assert g2["seconds"] == 0.0 and g2["checkpoint_seed"] == 42
    assert abs(g1["auc"] - g2["auc"]) < 0.05  # gumbel_softmax ở eval -> có thể lệch nhẹ

