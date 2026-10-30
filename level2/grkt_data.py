"""Cầu nối Cấp độ 2: biến bundle của Cấp độ 1 thành dữ liệu + đồ thị cho GRKT.

Hai bất đẳng ích của bản gốc được xử lý ở đây:

1. ``GRKT/Runner.py`` cắt chuỗi mỗi người học thành các cửa sổ 100 tương tác và chia
   k-fold trên cửa sổ — cùng một người học xuất hiện ở cả train lẫn test (phạm
   ``YEU_CAU.md`` §6.4). Ở đây MỖI NGƯỜI HỌC LÀ MỘT chuỗi, và danh sách người học lấy
   thẳng từ split của Cấp độ 1 (``level1.pipeline.save_split``).
2. ``GRKT/data/assist09/assist09.bin`` dựng ``rel_map``/``pre_map`` từ TOÀN BỘ dữ liệu
   (phạm §6.2). Ở đây hai ma trận được đếm lại chỉ từ tập train, trong không gian khái
   niệm của Cấp độ 1 (Cấp độ 1 đánh số 0-based; GRKT dành chỉ số 0 cho padding nên mọi
   chỉ số được dịch +1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- sequences của từng người học


@dataclass
class UserSequence:
    """Một chuỗi tương tác của MỘT người học, đã ở không gian chỉ số của GRKT (1-based)."""

    user_id: object
    probs: List[int]              # câu hỏi (problem_id đã ánh xạ), 1..n_probs
    knows: List[List[int]]        # khái niệm của từng lượt, 1..n_knows (0 không xuất hiện)
    corrs: List[int]              # 0/1
    times: List[int]              # mốc thời gian NGUYÊN theo phút, đơn điệu không giảm
    lags: List[int]               # thời gian làm bài (giây) — GRKT không dùng tới
    skills0: List[int]            # khái niệm theo không gian 0-based của Cấp độ 1 (đối chiếu)

    @property
    def length(self) -> int:
        return len(self.corrs)

    def as_grkt_items(self) -> List[tuple]:
        """Định dạng upstream: mỗi lượt là (prob, [knows], corr(bool), time, lag)."""
        return [
            (int(p), [int(k) for k in kn], bool(c), int(t), int(lag))
            for p, kn, c, t, lag in zip(self.probs, self.knows, self.corrs, self.times, self.lags)
        ]

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "probs": np.asarray(self.probs, dtype="int64"),
            "knows": np.asarray(self.knows, dtype="int64"),          # [S, K]
            "skills0": np.asarray(self.skills0, dtype="int64"),      # [S]
            "corrs": np.asarray(self.corrs, dtype="int64"),          # [S]
            "times": np.asarray(self.times, dtype="int64"),          # [S]
        }


@dataclass
class GRKTData:
    """Toàn bộ chuỗi người học (theo đúng thứ tự position của KTDataset Cấp độ 1)."""

    dataset: str
    concept_num: int
    n_probs: int
    sequences: List[UserSequence]
    skill_lookup: Dict[int, str] = field(default_factory=dict)
    time_mode: str = "duration_ms"
    gap_cap_min: int = 240
    stats: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self):
        self._by_user = {}
        for i, seq in enumerate(self.sequences):
            self._by_user.setdefault(seq.user_id, i)

    def by_users(self, user_ids: Sequence) -> List[UserSequence]:
        missing = [u for u in user_ids if u not in self._by_user]
        if missing:
            raise KeyError(
                f"{len(missing)} người học không có trong dữ liệu GRKT (vd {missing[:5]}). "
                "Split của Cấp độ 1 và dữ liệu đã nạp không cùng một lượt tiền xử lý."
            )
        return [self.sequences[self._by_user[u]] for u in user_ids]

    def items_of(self, sequences: Sequence[UserSequence]) -> List[List[tuple]]:
        return [s.as_grkt_items() for s in sequences]

    def max_concepts_per_step(self) -> int:
        return max(len(k) for s in self.sequences for k in s.knows)

    def n_interactions(self, sequences: Sequence[UserSequence]) -> int:
        return int(sum(s.length for s in sequences))


# ---------------------------------------------------------------- đường time


def _cumulative_times(deltas_min: np.ndarray, gap_cap_min: int) -> List[int]:
    """Cộng dồn khoảng cách (phút) đã chặn trên, trả về ĐÚNG S mốc (``times[0] == 0``).

    ``deltas_min[i]`` là thời gian trôi VÀO GIỮA lượt ``i-1`` và lượt ``i`` của cùng một
    người học (nên ``deltas_min[0]`` luôn bằng 0). GRKT nội suy ``delta_time =
    times[i+1] - times[i]`` rồi tính ``exp(-(learn_count+1) * delta_time * kernel)``, vì
    vậy:

    * bắt buộc chặn trên ở ``gap_cap_min`` — với delta thô (order_id, microgiây) số mũ âm
      hàng triệu -> underflow, mọi khái niệm bị xoá sạch trạng thái ngay bước kế tiếp và
      AUC chỉ còn bằng tỉ lệ đúng của dữ liệu;
    * làm tròn SAU khi cộng dồn, không làm tròn từng delta — thời gian làm bài trung bình
      0.5-5 phút, nếu ``rint`` từng delta thì mọi lượt dưới 1 phút thành 0 và nhân tử
      ``1 - exp(0)`` bằng 0, mô hình không học được gì từ giai đoạn học/quên.
    """
    clipped = np.clip(np.asarray(deltas_min, dtype="float64"), 0.0, float(gap_cap_min))
    return [int(x) for x in np.rint(np.cumsum(clipped))]


def _user_delta_minutes(series: pd.DataFrame, mode: str) -> np.ndarray:
    """Delta (phút) theo từng LƯỢT của một người học — tuyệt đối không tính trên cả khung.

    ``np.diff`` trên toàn bộ CSV đã nối nhiều người học sẽ nhét khoảng cách giữa người học
    này với người học kế tiếp vào delta đầu tiên của mỗi chuỗi (và ở ``timestamp_ms`` thì
    khoảng đó có thể hàng triệu phút, bị cap thành hằng số vô nghĩa).
    """
    n = len(series)
    if mode == "step":
        deltas = np.ones(n, dtype="float64")
        deltas[0] = 0.0
        return deltas
    if mode == "duration_ms":
        if "ms_first_response" not in series.columns:
            raise KeyError("Cột 'ms_first_response' không tồn tại — không dùng được time_mode='duration_ms'")
        ms = pd.to_numeric(series["ms_first_response"], errors="coerce").fillna(0.0).clip(lower=0.0)
        # Độ dài làm bài của lượt TRƯỚC xấp xỉ khoảng cách tới lượt sau; lượt đầu chưa có gì.
        return np.concatenate(([0.0], ms.to_numpy(dtype="float64")[:-1])) / 60000.0
    if mode == "timestamp_ms":
        if "timestamp_ms" not in series.columns:
            raise KeyError("Cột 'timestamp_ms' không tồn tại — không dùng được time_mode='timestamp_ms'")
        ts = pd.to_numeric(series["timestamp_ms"], errors="coerce").fillna(0.0).to_numpy(dtype="float64")
        deltas = np.diff(ts, prepend=ts[0]) / 60000.0
        deltas[0] = 0.0
        return deltas
    raise ValueError(f"time_mode không hợp lệ: {mode}")


def resolve_time_mode(df: pd.DataFrame, time_mode: str, verbose: bool = True) -> str:
    """Chọn lại thang thời gian nếu CSV thiếu cột, kèm cảnh báo in ra notebook.

    ``duration_ms`` và ``timestamp_ms`` là hai nguồn độc lập; chỉ khi hỏng cả hai mới lùi
    về ``step`` (mọi Δt = 1 phút) vì khi đó GRKT vẫn chạy được nhưng giai đoạn học/quên
    không còn thông tin thời gian để suy giảm.
    """
    if time_mode == "duration_ms" and "ms_first_response" not in df.columns:
        fallback = "timestamp_ms" if "timestamp_ms" in df.columns else "step"
        if verbose:
            print(f"  [cảnh báo] CSV không có ms_first_response -> chuyển sang time_mode='{fallback}'")
        return fallback
    if time_mode == "timestamp_ms" and "timestamp_ms" not in df.columns:
        fallback = "duration_ms" if "ms_first_response" in df.columns else "step"
        if verbose:
            print(f"  [cảnh báo] CSV không có timestamp_ms -> chuyển sang time_mode='{fallback}'")
        return fallback
    return time_mode


# ---------------------------------------------------------------- build


def build_grkt_data(
    bundle: dict,
    canonical_path,
    time_mode: str = "duration_ms",
    gap_cap_min: int = 240,
    max_seq_len: int = 200,
    min_seq_len: int = 3,
    max_concepts: int = 1,
    dataset: str = "assist2009",
    verbose: bool = True,
) -> GRKTData:
    """Nạp CSV chuẩn hoá và dựng chuỗi cho GRKT, ĐỐI CHIẾU với bundle của Cấp độ 1.

    Chuỗi tiền xử lý lặp lại đúng ``GKT/processing.load_dataset`` (bỏ dòng thiếu skill,
    lọc người có ít hơn ``min_seq_len`` tương tác, ``pd.factorize(skill_id, sort=True)``,
    ``groupby('user_id', sort=False)``, cắt ``max_seq_len``) để thứ tự người học và chỉ số
    khái niệm giống hệt nhau; kết quả được kiểm tra lại từng người học bằng
    ``_assert_align`` — bằng chứng split/đồ thị/không gian khái niệm là MỘT.
    """
    path = str(canonical_path)
    if max_concepts > 1:
        raise NotImplementedError(
            "CSV chuẩn hoá của Cấp độ 2 lưu MỘT khái niệm mỗi tương tác (cột skill_id), "
            "nên max_concepts > 1 không có thông tin để mở rộng. Muốn dùng đa khái niệm "
            "thì phải thêm cột danh sách kỹ năng và sửa _assert_align cùng logic 1-based."
        )
    base = bundle["train_loader"].dataset.dataset
    base_users = list(getattr(base, "user_ids", []))
    base_questions = list(base.questions)
    base_answers = list(base.answers)
    concept_num = int(bundle["concept_num"])

    df = pd.read_csv(path)
    for col in ("user_id", "skill_id", "correct"):
        if col not in df.columns:
            raise KeyError(f"CSV chuẩn hoá thiếu cột bắt buộc '{col}': {path}")
    df.dropna(subset=["skill_id"], inplace=True)
    df = df.groupby("user_id").filter(lambda q: len(q) >= min_seq_len).copy()
    df["skill"], _ = pd.factorize(df["skill_id"], sort=True)
    if "problem_id" in df.columns:
        df["prob"], _ = pd.factorize(df["problem_id"], sort=True)
    else:
        df["prob"] = 0
    df = df.reset_index(drop=True)

    derived_num = int(df["skill"].max()) + 1
    if derived_num != concept_num:
        raise AssertionError(
            f"Không gian khái niệm khác Cấp độ 1: {derived_num} so với {concept_num}. "
            "Cấp độ 2 phải dùng lại ĐÚNG bundle của Cấp độ 1 (cùng file, cùng min_seq_len)."
        )

    time_mode = resolve_time_mode(df, time_mode, verbose=verbose)
    has_ms = "ms_first_response" in df.columns

    sequences: List[UserSequence] = []
    positions = 0
    for user_id, series in df.groupby("user_id", sort=False):
        # 0/None = KHÔNG cắt, giống `if max_seq_len and len(feats) > max_seq_len` của Cấp độ 1
        limit = int(max_seq_len) if max_seq_len else len(series)
        take = min(limit, len(series))
        skills0 = series["skill"].to_numpy(dtype="int64")[:take]
        corrs = series["correct"].eq(1).to_numpy(dtype="int64")[:take]
        probs = series["prob"].to_numpy(dtype="int64")[:take]
        times = _cumulative_times(_user_delta_minutes(series, time_mode)[:take], gap_cap_min)
        if has_ms:
            lags = pd.to_numeric(
                series["ms_first_response"].head(take), errors="coerce").fillna(0.0).clip(lower=0.0) / 1000.0
        else:
            lags = np.zeros(take, dtype="float64")
        if positions < len(base_users):
            _assert_align(positions, user_id, skills0, corrs, base_users, base_questions, base_answers)
        knows = [[int(s + 1)] for s in skills0]
        sequences.append(
            UserSequence(
                user_id=_as_id(user_id),
                probs=[int(p + 1) for p in probs],
                knows=knows,
                corrs=[int(c) for c in corrs],
                times=times,
                lags=[int(x) for x in lags],
                skills0=[int(s) for s in skills0],
            )
        )
        positions += 1

    if positions != len(base_users):
        raise AssertionError(
            f"Số người học khác nhau: Cấp độ 2 tạo {positions}, bundle Cấp độ 1 có {len(base_users)}. "
            "Hai bên phải dùng cùng file chuẩn hoá + cùng min_seq_len/max_seq_len."
        )

    n_probs = int(df["prob"].max()) + 1
    skill_lookup = {int(k): v for k, v in (getattr(base, "skill_lookup", {}) or {}).items()}
    gap_stats = _gap_stats(sequences)
    data = GRKTData(
        dataset=dataset,
        concept_num=concept_num,
        n_probs=n_probs,
        sequences=sequences,
        skill_lookup=skill_lookup,
        time_mode=time_mode,
        gap_cap_min=gap_cap_min,
        stats={
            "users": len(sequences),
            "interactions": int(sum(s.length for s in sequences)),
            "n_probs": n_probs,
            "n_knows": concept_num,
            "max_seq_len_seen": max(s.length for s in sequences),
            **gap_stats,
        },
    )
    if verbose:
        print(f"  GRKT data: {data.stats['users']} người học | {data.stats['interactions']} tương tác | "
              f"n_knows={concept_num} | n_probs={n_probs} | chuỗi dài nhất={data.stats['max_seq_len_seen']}")
        print(f"  time_mode={time_mode}, Δt phút: p50={data.stats['dt_p50']:.1f} "
              f"p95={data.stats['dt_p95']:.1f} max={data.stats['dt_max']:.1f} (cap {gap_cap_min})")
    return data


def _as_id(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return x


def _assert_align(pos, user_id, skills0, corrs, base_users, base_questions, base_answers):
    """Bằng chứng Cấp độ 2 đang nhìn đúng người học + đúng nhãn như Cấp độ 1."""
    if _as_id(base_users[pos]) != _as_id(user_id):
        raise AssertionError(
            f"Thứ tự người học lệch ở vị trí {pos}: Cấp độ 1 = {base_users[pos]}, Cấp độ 2 = {user_id}"
        )
    q1 = list(base_questions[pos])
    a1 = list(base_answers[pos])
    s2 = [int(x) for x in skills0]
    c2 = [int(x) for x in corrs]
    if q1[: len(s2)] != s2:
        raise AssertionError(f"Chuỗi khái niệm của user {user_id} khác Cấp độ 1 tại vị trí {pos}")
    if a1[: len(c2)] != c2:
        raise AssertionError(f"Nhãn đúng/sai của user {user_id} khác Cấp độ 1 tại vị trí {pos}")


def _gap_stats(sequences: Sequence[UserSequence]) -> dict:
    deltas = np.asarray([b - a for s in sequences for a, b in zip(s.times[:-1], s.times[1:])], dtype="float64")
    if deltas.size == 0:
        return {"dt_p50": 0.0, "dt_p95": 0.0, "dt_max": 0.0, "dt_zero_frac": 1.0}
    return {
        "dt_p50": float(np.percentile(deltas, 50)),
        "dt_p95": float(np.percentile(deltas, 95)),
        "dt_max": float(deltas.max()),
        "dt_zero_frac": float((deltas == 0).mean()),
    }


# ---------------------------------------------------------------- đồ thị khái niệm


def build_relevance_maps(sequences: Sequence[UserSequence], n_knows: int, cold_thresh: int = 5) -> dict:
    """Đếm đồng xuất hiện / tiên quyết theo công thức của upstream, CHỈ trên ``sequences``.

    Khác upstream: upstream chỉ dùng khái niệm đầu tiên của lượt làm bài (``knows[0]``);
    ASSISTments 2009 mỗi lượt có một khái niệm nên tương đương, nhưng với dữ liệu đa khái
    niệm thì phải duyệt mọi khái niệm và chỉ cộng dồn bộ đếm mới SAU khi đã cập nhật hết
    bộ đếm của lượt đó (tránh để khái niệm đầu của cùng một lượt ảnh hưởng khái niệm thứ
    hai).
    """
    nk = n_knows + 1
    tt = np.zeros((nk, nk))
    tf = np.zeros((nk, nk))
    ft = np.zeros((nk, nk))
    ff = np.zeros((nk, nk))

    for seq in sequences:
        t = np.zeros(nk)
        f = np.zeros(nk)
        for knows, corr in zip(seq.knows, seq.corrs):
            if corr:
                for know in knows:
                    tt[:, know] += t
                    tt[know, :] += t
                    ft[:, know] += f
                    tf[know, :] += f
            else:
                for know in knows:
                    ff[:, know] += f
                    ff[know, :] += f
                    tf[:, know] += t
                    ft[know, :] += t
            for know in knows:
                if corr:
                    t[know] += 1.0
                else:
                    f[know] += 1.0

    rel_filt = (tt + ff + tf + ft) >= 4 * cold_thresh
    pre_filt = (tf + ft) >= 2 * cold_thresh
    rel = (tt + ff) / np.clip(tt + ff + tf + ft, a_min=1.0, a_max=None)
    pre = ft / np.clip(tf + ft, a_min=1.0, a_max=None)
    rel = rel * rel_filt
    pre = pre * pre_filt
    for m in (rel, pre):
        np.fill_diagonal(m, 0.0)
        m[0, :] = 0.0
        m[:, 0] = 0.0
    return {
        "rel": rel.astype(np.float64),
        "pre": pre.astype(np.float64),
        "counts": {"tt": float(tt.sum()), "tf": float(tf.sum()), "ft": float(ft.sum()), "ff": float(ff.sum())},
        "cold_thresh": cold_thresh,
        "sequences": len(sequences),
    }


def binarize(mat: np.ndarray, thresh: float) -> np.ndarray:
    out = (np.asarray(mat) > thresh).astype(np.float32)
    out[0, :] = 0.0
    out[:, 0] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def _shuffle_binary(mat: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Ablation: giữ nguyên số cạnh ra của từng khái niệm, đổi ngẫu nhiên đích đến."""
    rows, cols = np.nonzero(mat)
    if rows.size == 0:
        return np.zeros_like(mat, dtype=np.float32)
    new_cols = rng.permutation(cols)
    out = np.zeros_like(mat, dtype=np.float32)
    for r, c in zip(rows, new_cols):
        out[r, c] = 1.0
    out[0, :] = 0.0
    out[:, 0] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def graph_variant(maps: dict, kind: str, n_knows: int, thresh: float = 0.6, seed: int = 42) -> dict:
    """Trả về (rel_map, pre_map) dạng 0/1 cho một kiểu đồ thị, kèm số cạnh để báo cáo.

    * ``train``   : cạnh đồng xuất hiện + tiên quyết đếm từ tập train, ngưỡng ``thresh``
    * ``dense``   : đồ thị đầy đủ (không có cấu trúc) — mọi cặp khái niệm nối với nhau
    * ``zero``    : bỏ toàn bộ cạnh (GRKT khi đó chỉ cập nhật từng khái niệm riêng lẻ)
    * ``shuffled``: same sparsity, dstshuffle — phá cấu trúc nhưng giữ bậc của từng đỉnh

    ``args.thresh`` của GRKT được đặt 0.5 để ``(map > thresh)`` giữ nguyên ma trận 0/1 ở
    trên; KHÔNG dùng ``thresh=0`` vì nhánh đó của upstream gán toàn bộ ma trận = 1, bao
    gồm cả hàng/cột padding số 0.
    """
    nk = n_knows + 1
    rng = np.random.RandomState(seed)
    if kind == "train":
        rel = binarize(maps["rel"], thresh)
        pre = binarize(maps["pre"], thresh)
    elif kind == "dense":
        rel = np.ones((nk, nk), dtype=np.float32)
        pre = np.ones((nk, nk), dtype=np.float32)
        for m in (rel, pre):
            m[0, :] = 0.0
            m[:, 0] = 0.0
            np.fill_diagonal(m, 0.0)
    elif kind == "zero":
        rel = np.zeros((nk, nk), dtype=np.float32)
        pre = np.zeros((nk, nk), dtype=np.float32)
    elif kind == "shuffled":
        rel = _shuffle_binary(binarize(maps["rel"], thresh), rng)
        pre = _shuffle_binary(binarize(maps["pre"], thresh), np.random.RandomState(seed + 1))
    else:
        raise ValueError(f"Kiểu đồ thị không hợp lệ: {kind}")
    return {
        "rel_map": rel,
        "pre_map": pre,
        "edges_rel": int(rel.sum()),
        "edges_pre": int(pre.sum()),
        "kind": kind,
        "thresh": thresh,
    }


def adjacency_from_graphs(rel_map: np.ndarray, pre_map: np.ndarray) -> np.ndarray:
    """Láng giềng 1-hop (hỗn hợp rel + pre hai chiều) dùng cho tiêu chí tính cục bộ."""
    adj = ((np.asarray(rel_map) > 0) | (np.asarray(pre_map) > 0) | (np.asarray(pre_map).T > 0)).astype(np.float32)
    adj[0, :] = 0.0
    adj[:, 0] = 0.0
    np.fill_diagonal(adj, 0.0)
    return adj


# ---------------------------------------------------------------- split + chống rò rỉ


def split_sequences(data: GRKTData, split_payload: dict) -> Dict[str, dict]:
    """Ánh xạ danh sách người học của Cấp độ 1 sang chuỗi GRKT theo từng split."""
    out = {}
    for name in ("train_users", "val_users", "test_users"):
        users = [_as_id(u) for u in split_payload[name]]
        seqs = data.by_users(users)
        out[name.replace("_users", "")] = {"user_ids": users, "sequences": seqs, "items": data.items_of(seqs)}
    return out


def leak_report(
    data: GRKTData,
    split_sets: Dict[str, dict],
    bundle: dict,
    maps_train: dict,
    maps_val: dict,
    graph_info: Dict[str, dict],
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> str:
    """Báo cáo chống rò rỉ bằng số, in ra notebook và ghi ``leak_report.txt``."""
    tr = set(split_sets["train"]["user_ids"])
    va = set(split_sets["val"]["user_ids"])
    te = set(split_sets["test"]["user_ids"])
    lines: List[str] = []
    lines.append("=== BÁO CÁO CHỐNG RÒ RÌ (Cấp độ 2 / GRKT) ===")
    lines.append(f"File chuẩn hoá          : {data.dataset} | time_mode={data.time_mode}")
    lines.append(
        f"Người học train/val/test: {len(tr)}/{len(va)}/{len(te)} "
        f"(split seed={seed}, tỉ lệ {train_ratio}/{val_ratio})"
    )
    lines.append(
        f"Giám sát giao nhau      : train∩val={len(tr & va)} train∩test={len(tr & te)} val∩test={len(va & te)} "
        "(phải = 0)"
    )
    n_int = {k: int(sum(s.length for s in v["sequences"])) for k, v in split_sets.items()}
    lines.append(f"Số tương tác            : {n_int}")
    lines.append(
        f"Không gian khái niệm    : n_knows={data.concept_num} (Cấp độ 1 + 1 cho padding), "
        f"n_probs={data.n_probs}"
    )
    e_tr = int((maps_train["rel"] > 0).sum() + (maps_train["pre"] > 0).sum())
    e_va = int((maps_val["rel"] > 0).sum() + (maps_val["pre"] > 0).sum())
    lines.append(f"Số cạnh đồ thị từ train : {e_tr}")
    lines.append(f"Số cạnh nếu dựng từ val : {e_va} (khác train => đồ thị phụ thuộc tập huấn luyện)")
    for kind, info in graph_info.items():
        lines.append(f"  biến thể {kind:9s}: rel={info['edges_rel']:6d} pre={info['edges_pre']:6d}")
    lines.append("Nhãn hiện tại/tương lai : không dùng. GRKT chỉ dùng knows[i+1] và times[i+1] "
                 "(danh tính + thời điểm câu hỏi kế tiếp, KHÔNG dùng kết quả) cho giai đoạn học/quên.")
    lines.append("Padding                 : chỉ số 0 (khái niệm & câu hỏi); hàng/cột 0 của đồ thị luôn = 0.")
    return "\n".join(lines)
