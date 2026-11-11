"""Bất biến chống rò rỉ (YEU_CAU.md §6) — chạy trên CPU với file 15 người học."""

import numpy as np
import pandas as pd
import pytest
import torch

from tests.conftest import SMOKE_CSV


def _users(bundle, split):
    from level1.pipeline import _user_ids_from_subset

    return _user_ids_from_subset(bundle[f"{split}_loader"].dataset)


def test_split_by_learner_is_disjoint_and_complete(smoke_bundle):
    tr, va, te = (set(_users(smoke_bundle, s)) for s in ("train", "val", "test"))
    assert not (tr & va) and not (tr & te) and not (va & te)
    all_users = set(smoke_bundle["train_loader"].dataset.dataset.user_ids)
    assert tr | va | te == all_users


def test_split_does_not_depend_on_graph_type():
    from level1.pipeline import load_kt_data

    ref = None
    for g in ("Transition", "Dense", "Zero", "Shuffled"):
        b = load_kt_data(str(SMOKE_CSV), batch_size=4, graph_type=g, seed=42, max_seq_len=80, use_cuda=False)
        got = [_users(b, s) for s in ("train", "val", "test")]
        ref = ref or got
        assert got == ref, g


def test_transition_graph_uses_train_learners_only(smoke_bundle):
    from GKT.processing import build_transition_graph

    base = smoke_bundle["train_loader"].dataset.dataset
    train_idx = list(smoke_bundle["train_loader"].dataset.indices)
    test_idx = list(smoke_bundle["test_loader"].dataset.indices)
    qs = [list(q) for q in base.questions]
    lens = [len(q) for q in qs]
    K = smoke_bundle["concept_num"]
    g0 = build_transition_graph(qs, lens, train_idx, len(qs), K)
    # Xáo trộn hoàn toàn chuỗi của người học TEST: đồ thị không được đổi.
    rng = np.random.RandomState(0)
    for i in test_idx:
        qs[i] = rng.randint(0, K, size=lens[i]).tolist()
    g1 = build_transition_graph(qs, lens, train_idx, len(qs), K)
    assert torch.equal(g0, g1)
    # Đối chứng: đổi chuỗi người học TRAIN thì đồ thị phải đổi (test có khả năng thất bại).
    qs[train_idx[0]] = rng.randint(0, K, size=lens[train_idx[0]]).tolist()
    g2 = build_transition_graph(qs, lens, train_idx, len(qs), K)
    assert not torch.equal(g0, g2)


def test_transition_graph_rows_are_normalized(smoke_bundle):
    g = smoke_bundle["graph"].cpu().numpy()
    sums = g.sum(axis=1)
    assert np.all(np.isclose(sums, 1.0, atol=1e-5) | (sums == 0))
    assert np.all(np.diag(g) == 0)


@pytest.mark.parametrize("graph_type", ["N/A", "Transition", "MHA"])
def test_prediction_is_causal(smoke_bundle, cpu, graph_type):
    """pred[:, j] (dự đoán câu j+1) chỉ được phụ thuộc nhãn ở vị trí <= j."""
    from level1.pipeline import build_model, set_seed

    set_seed(0)
    model_type = "DKT" if graph_type == "N/A" else "GKT"
    model = build_model(
        model_type, smoke_bundle["concept_num"], graph=smoke_bundle["graph"],
        graph_type=graph_type if graph_type != "N/A" else "Transition", device=cpu,
    ).eval()
    feats, qs, _ = next(iter(smoke_bundle["test_loader"]))
    feats, qs = feats[:1, :30].clone(), qs[:1, :30].clone()
    k = 15
    flipped = feats.clone()
    flipped[0, k] = flipped[0, k] ^ 1  # đổi nhãn đúng/sai ở bước k (feature = skill*2+correct)
    with torch.no_grad():
        p0 = model(feats, qs)
        p1 = model(flipped, qs)
    p0 = p0[0] if isinstance(p0, tuple) else p0
    p1 = p1[0] if isinstance(p1, tuple) else p1
    assert torch.equal(p0[:, :k], p1[:, :k]), "dự đoán trước bước k thay đổi -> rò nhãn tương lai"
    assert not torch.equal(p0[:, k:], p1[:, k:]), "đổi nhãn không ảnh hưởng gì -> test vô nghĩa"


def test_level2_sequences_align_with_level1(smoke_bundle):
    from level2.grkt_data import build_grkt_data, split_sequences
    from level1.pipeline import save_split
    import tempfile
    from pathlib import Path

    data = build_grkt_data(smoke_bundle, SMOKE_CSV, max_seq_len=80, verbose=False)
    with tempfile.TemporaryDirectory() as d:
        split = save_split(smoke_bundle, Path(d) / "split.json")
    sets = split_sequences(data, split)
    for s in ("train", "val", "test"):
        assert [q.user_id for q in sets[s]["sequences"]] == split[f"{s}_users"]


# ------------------------------------------------------------------ tiền xử lý (checklist Mức 0)
def test_no_consecutive_duplicate_interactions(full_csv):
    from kt_common.interactions import interaction_stats, load_interactions

    stats = interaction_stats(load_interactions(full_csv))
    assert stats["dup_adjacent"] == 0, f"{stats['dup_adjacent']} cặp bước liền nhau là CÙNG một lượt"


def test_sequences_are_chronological(full_csv):
    from kt_common.interactions import interaction_stats, load_interactions

    stats = interaction_stats(load_interactions(full_csv))
    assert stats["order_backsteps"] == 0, f"{stats['order_backsteps']} lần order_id đi lùi"


def test_raw_file_really_has_the_defects(full_csv):
    """Đối chứng: file gốc CÓ lỗi -> hai test trên đo được việc gộp/sắp, không đúng vì tình cờ."""
    raw = pd.read_csv(full_csv, low_memory=False).dropna(subset=["skill_id"])
    diff = raw.groupby("user_id", sort=False)["order_id"].diff()
    assert (diff == 0).sum() > 100_000 and (diff < 0).sum() > 10_000


def test_preprocessing_invariants_on_smoke():
    from kt_common.interactions import load_interactions

    raw = pd.read_csv(SMOKE_CSV).dropna(subset=["skill_id"])
    df = load_interactions(SMOKE_CSV)
    key = ["user_id", "order_id"]
    # Mỗi lượt đúng một dòng, chuỗi tăng dần theo order_id.
    assert not df.duplicated(key).any()
    assert all(g["order_id"].is_monotonic_increasing for _, g in df.groupby("user_id"))
    # Không bịa dòng: mọi lượt đều có trong file gốc, giữ đúng nhãn và kỹ năng ĐẦU TIÊN của lượt.
    first = raw.drop_duplicates(key, keep="first")[key + ["correct", "skill_id"]]
    merged = df.merge(first, on=key, suffixes=("", "_raw"))
    assert len(merged) == len(df)
    assert (merged["correct"] == merged["correct_raw"]).all()
    assert (merged["skill_id"] == merged["skill_id_raw"]).all()
    # Lượt của người học đủ điều kiện không bị mất.
    counts = first.groupby("user_id").size()
    assert len(df) == int(counts[counts >= 3].sum())
    # Thứ tự người học = lần xuất hiện đầu tiên trong file.
    order = [u for u in dict.fromkeys(raw["user_id"]) if u in set(df["user_id"])]
    assert list(dict.fromkeys(df["user_id"])) == order


def test_preprocessing_without_order_id_keeps_file_order(tmp_path):
    from kt_common.interactions import load_interactions

    df = pd.read_csv(SMOKE_CSV).drop(columns=["order_id"])
    path = tmp_path / "no_order.csv"
    df.to_csv(path, index=False)
    got = load_interactions(path)
    ref = df.dropna(subset=["skill_id"])
    ref = ref[ref.groupby("user_id")["user_id"].transform("size") >= 3].copy()
    ref["skill"], _ = pd.factorize(ref["skill_id"], sort=True)
    pd.testing.assert_frame_equal(got, ref)


def test_bundle_reports_data_quality(smoke_bundle):
    from kt_common.interactions import DATA_VERSION

    st = smoke_bundle["data_stats"]
    assert smoke_bundle["data_version"] == DATA_VERSION == st["data_version"]
    assert st["dup_adjacent"] == 0 and st["order_backsteps"] == 0
    assert 0 <= st["truncated_frac"] < 1


def test_fingerprints_carry_data_version(smoke_bundle):
    """Checkpoint train trên dữ liệu trước bản sửa (không có khoá này) phải bị từ chối."""
    from kt_common.checkpoint import check_fingerprint
    from kt_common.interactions import DATA_VERSION
    from level2.baselines import baseline_fingerprint

    fp = baseline_fingerprint(smoke_bundle, "DKT", "N/A", batch_size=4, max_seq_len=80,
                              min_seq_len=3, train_ratio=0.6, val_ratio=0.2, seed=42)
    assert fp["data_version"] == DATA_VERSION
    old = {k: v for k, v in fp.items() if k != "data_version"}
    with pytest.raises(RuntimeError, match="data_version"):
        check_fingerprint(old, fp, "hint")
