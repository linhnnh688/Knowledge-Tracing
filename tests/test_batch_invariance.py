"""Dự đoán lúc eval của một người học KHÔNG được phụ thuộc người học khác trong cùng batch.

Đây là nguyên nhân của GAP_LEVEL1 §6.4.3 (10/10 case xAI không tái lập được): ``MLP.batch_norm``
bỏ chuẩn hoá khi chỉ còn MỘT người học hoạt động tại một bước, kể cả lúc eval — nên chạy lại
một người học riêng lẻ cho p khác với lúc chạy cả batch.
"""

import pytest
import torch

from GKT.processing import pad_collate


def _trained_like_batchnorm(model, seed=0):
    """BatchNorm mới khởi tạo có mean=0, var=1 ≈ phép đồng nhất -> đặt thống kê giống đã train."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=g))
            m.running_var.copy_(torch.rand(m.running_var.shape, generator=g) * 3 + 0.2)


@pytest.mark.parametrize("graph_type", ["Transition", "MHA"])
def test_gkt_eval_prediction_independent_of_batch(smoke_bundle, cpu, graph_type):
    from level1.pipeline import build_model, set_seed

    set_seed(0)
    model = build_model("GKT", smoke_bundle["concept_num"], graph=smoke_bundle["graph"],
                        graph_type=graph_type, device=cpu)
    _trained_like_batchnorm(model)
    model.eval()
    base = smoke_bundle["train_loader"].dataset.dataset
    lens = [len(q) for q in base.questions]
    long_i = max(range(len(lens)), key=lens.__getitem__)
    short_i = min(range(len(lens)), key=lens.__getitem__)
    assert lens[long_i] > lens[short_i]
    with torch.no_grad():
        alone = model(*pad_collate([base[long_i]])[:2])
        paired = model(*pad_collate([base[long_i], base[short_i]])[:2])
    alone = alone[0] if isinstance(alone, tuple) else alone
    paired = paired[0] if isinstance(paired, tuple) else paired
    n = lens[long_i] - 1
    assert torch.allclose(alone[0, :n], paired[0, :n], atol=1e-5), (
        "p của người học đổi theo batch: "
        f"max |Δp| = {(alone[0, :n] - paired[0, :n]).abs().max():.4f}"
    )


def test_case_evidence_is_reproducible(smoke_bundle, cpu):
    """GAP_LEVEL1 §6.4.3: p chạy lại một người học phải trùng p lúc chọn case (theo batch)."""
    from level1.explain import collect_cases
    from level1.pipeline import build_model, set_seed

    set_seed(0)
    model = build_model("GKT", smoke_bundle["concept_num"], graph=smoke_bundle["graph"],
                        graph_type="Transition", device=cpu)
    _trained_like_batchnorm(model)
    model.eval()
    buckets = collect_cases(model, smoke_bundle["test_loader"], graph=smoke_bundle["graph"],
                            n_each=3, device=cpu)
    recs = [r for rows in buckets.values() for r in rows]
    assert recs
    for r in recs:
        ev = r["evidence"]
        assert ev["ok"], ev
        assert abs(ev["p_recheck"] - r["y_prob"]) < 1e-5, ev
