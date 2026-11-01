"""Nhiệm vụ 2 Cấp độ 2: quỹ đạo mastery m(u, c, t) và các tiêu chí hợp lý.

Ba mốc đo trong một bước làm bài i (xem các chỗ "SỬA so với bản gốc" ở
``GRKT/model/GRKT.py::forward``):

``m_pre[i]``    mastery MỌI khái niệm trước khi cập nhật — đúng là trạng thái mô hình đã
                dùng để dự đoán ``score[i]``, nên so được với ``m_strong[i]``.
``m_tilde[i]``  mastery SAU giai đoạn truy xuất (aggregation lan truyền trên đồ thị) —
                vector thật sự đi vào head dự đoán; khác ``m_pre`` ở đúng phần lan truyền
                trên cạnh đồ thị.
``m_strong[i]`` mastery SAU giai đoạn củng cố (đúng → gain, sai → loss) của bước i.

Suy ra, mỗi bước tách được hai cơ chế:

* ``Δm_củng_cố = m_strong[i] - m_pre[i]``            : ảnh hưởng của VIỆC LÀM ĐÚNG/SAI
* ``Δm_học_quên = m_pre[i+1] - m_strong[i]``         : ảnh hưởng của HỌC + QUÊN theo thời
                                          gian giữa hai bước

GRKT định nghĩa mastery là một vô hướng KHÔNG bị chặn [0, 1] (``Positive_Linear`` với
trọng số softmax dương, nhân với trạng thái ẩn đã kẹp ±10). Để vẽ và so sánh với mức độ
thành thạo của DKT/GKT Cấp độ 1, mọi đầu ra ở đây dùng ``sigmoid(m)`` — một phép biến đổi
đơn điệu nên không đổi thứ tự, không đổi dấu của các hiệu vi phân, và không đổi kết luận
của các tiêu chí bên dưới. Giá trị thô cũng được lưu để đối chiếu.

Phân loại tiêu chí (quan trọng khi viết báo cáo):

* **Kiểm chứng cài đặt** — ``consistency_strengthen``, ``consistency_spread``, ``locality``.
  Ba số này BẰNG 1 theo cấu trúc của GRKT (mọi ``Positive_Linear`` có trọng số dương, cập
  nhật ``h += ±relu(...)``, và ``k_hop=1``), nên chúng chỉ có thể phát hiện lỗi hook/tiền
  xử lý. Chúng KHÔNG chứng minh vai trò của đồ thị và KHÔNG phải "tiêu chí hợp lý" theo
  nghĩa đề bài.
* **Tiêu chí thật** — ``precedence_gap`` (nhất quán tiên quyết so với hoán vị cạnh) và
  ``stability_spearman`` (ổn định khi lật một kết quả làm bài có bước kế tiếp), cùng
  ``forget_slope``/``pearson_dt_dm``. Ba nhóm này có thể sai thật và là bằng chứng cấp 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

EPS = 1e-9

#: Ba số đo bị cấu trúc mô hình ép về 1 — chỉ dùng để kiểm chứng cài đặt, không dùng làm
#: bằng chứng vai trò đồ thị (xem docstring module và GAP_LEVEL2.md §F1).
INSTALLATION_CHECK_COLUMNS = (
    "consistency_strengthen",
    "consistency_spread",
    "locality",
)


# ---------------------------------------------------------------- ghi quỹ đạo
def _new_rec() -> dict:
    return {"m_pre": [], "m_tilde": [], "m_strong": []}


def _stack_rec(rec: dict, n_steps: int) -> Dict[str, np.ndarray]:
    """[S, B, NK] (float16, CPU) -> dict mỗi mục [B, S, NK] float32.

    Cột số 0 (padding) bị ép về 0: upstream ``scatter`` theo ``know_index`` có ghi đè lên
    hàng padding khi một lượt có ít hơn K khái niệm, nên cột đó chứa rác của ``init_hidden``
    chứ không phải độ thành thạo của bất kỳ khái niệm nào. Loại nó khỏi mọi chỉ số bên dưới.
    """
    out = {}
    for key in ("m_pre", "m_tilde", "m_strong"):
        arr = torch.stack(rec[key], dim=1).to(torch.float32).numpy()  # [B, S, NK]
        assert arr.shape[1] == n_steps, (arr.shape, n_steps)
        arr[:, :, 0] = 0.0
        out[key] = arr
    return out


@torch.no_grad()
def record_trajectories(
    model,
    sequences: Sequence,
    device=None,
    batch_size: int = 32,
    max_users: Optional[int] = None,
    seed: int = 0,
    concept_num: Optional[int] = None,
    graph_kind: str = "train",
    perturb: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Chạy forward một lượt trên ``sequences`` và ghi m(u, c, t) cho từng người học.

    ``sequences`` là danh sách ``level2.grkt_data.UserSequence`` (thông thường lấy từ
    ``split_sets[...]["sequences"]``). Kết quả là dict ``{"users": [...], "concept_num":
    int, "graph_kind": str}`` với mỗi người học là các mảng ``[S_u, NK]`` đã cắt bỏ padding.

    ``perturb="flip_penultimate"`` lật đúng/sai của bước áp chót (bước cuối cùng còn có
    bước kế tiếp để lan truyền), dùng cho tiêu chí ổn định — xem
    :func:`stability_under_perturbation`.

    Ghi chú: GRKT dùng ``F.gumbel_softmax(..., hard=True)`` nên chiều "học thêm" giữa hai
    bước là ngẫu nhiên ngay ở chế độ eval. Vì vậy cần ``seed`` cố định để quỹ đạo tái lập
    được, và nên giải thích độ biến thiên giữa các lần ghi như một phần của kết luận.
    """
    from level1.pipeline import set_seed

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    model.eval()
    model.to(device)
    model.args.device = device

    seqs = list(sequences)
    if max_users is not None:
        seqs = seqs[: int(max_users)]
    items = [_items_with_perturbation(s, perturb) for s in seqs]

    users: List[dict] = []
    offset = 0
    for batch, batch_seqs in _batch_with_meta(items, seqs, batch_size):
        feed = model.get_feed_dict(batch, "valid")
        for key in ("probs", "knows", "corrs", "times", "lags", "filt", "labels"):
            feed[key] = feed[key].to(device)
        feed["rec"] = _new_rec()
        model.forward(feed)
        n_steps = max(len(b) for b in batch)
        stacked = _stack_rec(feed["rec"], n_steps)
        for j, seq in enumerate(batch_seqs):
            L = seq.length
            users.append(
                {
                    "user_id": seq.user_id,
                    # "_raw" = vô hướng mastery chưa bị chặn của GRKT; biến không đuôi = sigmoid(_)
                    # ∈ [0, 1] để vẽ/so sánh được với độ thành thạo của Cấp độ 1.
                    "m_pre_raw": stacked["m_pre"][j, :L],
                    "m_strong_raw": stacked["m_strong"][j, :L],
                    "m_tilde_raw": stacked["m_tilde"][j, :L],
                    "m_pre": _sigmoid(stacked["m_pre"][j, :L]),
                    "m_strong": _sigmoid(stacked["m_strong"][j, :L]),
                    "m_tilde": _sigmoid(stacked["m_tilde"][j, :L]),
                    "knows": np.asarray(seq.knows, dtype="int64"),  # [S, K] 1-based
                    "corrs": np.asarray(seq.corrs, dtype="int64"),  # [S]
                    "times": np.asarray(seq.times, dtype="int64"),  # [S] phút cộng dồn
                    "skills0": np.asarray(
                        seq.skills0, dtype="int64"
                    ),  # [S] không gian Cấp độ 1
                }
            )
        offset += len(batch_seqs)
        if verbose and (offset // 200) != (offset - len(batch_seqs)) // 200:
            print(f"  đã ghi quỹ đạo {offset}/{len(seqs)} người học ...")
    if verbose:
        print(f"  đã ghi quỹ đạo {len(users)}/{len(seqs)} người học")
    return {
        "users": users,
        "concept_num": int(
            concept_num if concept_num is not None else model.args.n_knows
        ),
        "graph_kind": str(graph_kind),
        "n_knows_plus_pad": int(model.args.n_knows) + 1,
        "seed": int(seed),
    }


def _sigmoid(m: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(m, -30.0, 30.0)))


def _items_with_perturbation(seq, perturb: Optional[str]) -> List[tuple]:
    """Chuỗi upstream, có thể lật kết quả bước áp chót (giữ nguyên độ dài)."""
    items = seq.as_grkt_items()
    if perturb in (None, "none"):
        return items
    if perturb != "flip_penultimate":
        raise ValueError(
            f"perturb không hợp lệ: {perturb!r} (chỉ nhận None hoặc 'flip_penultimate')"
        )
    idx = len(items) - 2
    if idx < 0:
        return items  # chuỗi 1 bước: không có bước kế tiếp để lan truyền, giữ nguyên
    prob, knows, corr, time, lag = items[idx]
    items[idx] = (prob, knows, not corr, time, lag)
    return items


def _batch_with_meta(items: List[list], seqs: List, batch_size: int):
    """Giữ nguyên thứ tự người học để mảng ghi được khớp với ``user_id``."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size], seqs[i : i + batch_size]


# ---------------------------------------------------------------- tiêu chí hợp lý
def directional_consistency(
    traj: Dict[str, object], adj: Optional[np.ndarray] = None
) -> dict:
    """Hướng thay đổi mastery có đi đúng chiều với kết quả làm bài không.

    **CẢ HAI số đo đều là kiểm chứng cài đặt, không phải bằng chứng vai trò đồ thị.**
    GRKT cộng/trừ state bằng tensor đã ``relu()``, còn mọi trọng số của ``know_master_proj``
    / ``agg_*`` / ``gain_*`` / ``loss_*`` đều dương (``Positive_Linear`` + ``softmax``);
    với ``k_hop=1`` phép lan truyền chỉ chạm đúng các láng giềng 1-hop. Vì vậy:

    * ``strengthen`` — tại ĐÚNG khái niệm vừa làm: ``Δm = m_strong - m_pre`` tăng khi làm
      đúng và giảm khi làm sai **theo cấu trúc**;
    * ``spread`` — tại các khái niệm LÁNG GIỀNG 1-hop: ``Δm`` cùng dấu với kết quả làm bài
      **cũng theo cấu trúc** (ma trận trọng số dương ⇒ không thể đảo dấu), và bằng 0 ngoài
      1-hop.

    Cả hai chỉ có thể phát hiện lỗi hook/tiền xử lý (kỳ vọng = 1). Muốn đo tác dụng thật
    của cấu trúc đồ thị, dùng :func:`precedence_consistency`,
    :func:`stability_under_perturbation` hoặc so ``abs_delta_*`` giữa các biến thể.

    Trả về tỉ lệ trên các cặp (bước, khái niệm) CÓ thay đổi khác 0; không có mẫu nào thì
    trả ``None`` (xảy ra với đồ thị ``zero``: không cạnh nên không lan truyền).
    """
    ok_strong = tot_strong = 0
    ok_spread = tot_spread = 0
    n_steps_used = 0
    for user in traj["users"]:
        m_pre, m_strong = user["m_pre"], user["m_strong"]
        knows, corrs = user["knows"], user["corrs"]
        dm = m_strong - m_pre  # [S, NK+1]
        for i in range(len(corrs)):
            ans = [
                int(k) for k in knows[i] if k > 0
            ]  # đã là chỉ số không gian có padding
            if not ans:
                continue
            n_steps_used += 1
            valence = 1.0 if corrs[i] else -1.0
            for a in ans:
                d = float(dm[i, a])
                if abs(d) > EPS:
                    tot_strong += 1
                    ok_strong += int(np.sign(d) == valence)
            if adj is not None:
                nb = set()
                for a in ans:
                    nb.update(int(c) for c in np.nonzero(np.asarray(adj)[a])[0])
                nb -= set(ans)
                nb.discard(0)
                if nb:
                    idx = np.fromiter(nb, dtype="int64")
                    vals = dm[i, idx]
                    nz = vals[np.abs(vals) > EPS]
                    if nz.size:
                        tot_spread += int(nz.size)
                        ok_spread += int((np.sign(nz) == valence).sum())
    return {
        "strengthen": (ok_strong / tot_strong) if tot_strong else None,
        "n_strengthen_samples": tot_strong,
        "spread": (ok_spread / tot_spread) if tot_spread else None,
        "n_spread_samples": tot_spread,
        "n_steps": n_steps_used,
    }


def locality(traj: Dict[str, object], adj: Optional[np.ndarray] = None) -> dict:
    """Tỉ lệ thay đổi mastery nằm trong {khái niệm đang làm ∪ láng giềng 1-hop}.

    ``locality = Σ_{c ∈ đang làm ∪ N(đang làm)} |Δm_c| / Σ_c |Δm_c|``. Với ``k_hop=1``,
    ``Δm`` ngoài tập này **bằng 0 theo cấu trúc** ⇒ ``locality ≡ 1`` trên mọi đồ thị có
    cạnh. Đây là kiểm chứng cài đặt (kỳ vọng = 1), KHÔNG phải bằng chứng về vai trò đồ thị
    và không phân biệt được ``train`` với ``shuffled``. Muốn đo độ lan toả thật, dùng
    ``abs_delta_retrieval``/``stability_changed_frac`` giữa các biến thể.
    """
    num = den = 0.0
    per_step: List[float] = []
    adj_np = None if adj is None else np.asarray(adj)
    for user in traj["users"]:
        dm = user["m_strong"] - user["m_pre"]
        knows = user["knows"]
        for i in range(dm.shape[0]):
            ans = [int(k) for k in knows[i] if k > 0]
            if not ans:
                continue
            focus = set(ans)
            if adj_np is not None:
                for a in ans:
                    focus.update(int(c) for c in np.nonzero(adj_np[a])[0])
            focus.discard(0)
            absdm = np.abs(dm[i])
            tot = float(absdm.sum())
            if tot <= EPS:
                continue
            idx = np.fromiter(focus, dtype="int64")
            share = float(absdm[idx].sum()) / tot
            num += share * tot
            den += tot
            per_step.append(share)
    return {
        "locality": (num / den) if den > EPS else None,
        "n_steps": len(per_step),
        "locality_median": float(np.median(per_step)) if per_step else None,
    }


def precedence_consistency(
    traj: Dict[str, object],
    pre_map: Optional[np.ndarray],
    n_permutations: int = 5,
    seed: int = 42,
) -> dict:
    """Nhất quán tiên quyết: mastery khái niệm TIÊN QUYẾT có cao hơn khái niệm PHỤ thuộc?

    Với mỗi cạnh ``c → d`` của ``pre_map`` (``c`` là tiên quyết của ``d``), xét trạng thái
    ``m_pre[i]`` — trạng thái mô hình dùng để dự đoán bước ``i``, trước khi bước đó làm tăng
    mastery của ``d``. Đếm tỉ lệ ``m_pre[i, c] ≥ m_pre[i, d]`` trên các bước người học trả
    lời ĐÚNG ``d``, với điều kiện cả ``c`` lẫn ``d`` đã từng được người đó làm trước bước
    ``i`` (khái niệm chưa luyện có prior giống nhau nên chỉ tạo tie vô nghĩa).

    Đây là tiêu chí **có thể sai thật**: không bị ép bởi dấu của trọng số như ``spread``.
    Để biết "cao" có ý nghĩa hay chỉ là ngẫu nhiên, so với baseline hoán vị đích của các
    cạnh (giữ nguyên số cạnh ra của mỗi đỉnh): ``precedence_gap = precedence −
    precedence_permuted``. Gap dương mới là bằng chứng cấu trúc tiên quyết có tác dụng.

    Trả ``None`` cho các số đo nếu đồ thị rỗng hoặc không có mẫu.
    """
    empty = {
        "precedence": None,
        "precedence_permuted": None,
        "precedence_gap": None,
        "precedence_tie_frac": None,
        "n_precedence_samples": 0,
        "n_precedence_edges": 0,
    }
    if pre_map is None:
        return empty
    pre = np.asarray(pre_map) > 0
    if pre.ndim != 2 or pre.shape[0] != pre.shape[1]:
        raise ValueError(f"pre_map phải là ma trận vuông, nhận {pre.shape}")
    pre[0, :] = False
    pre[:, 0] = False
    np.fill_diagonal(pre, False)
    src, dst = np.nonzero(pre)
    if src.size == 0:
        return empty

    lookups = [_sources_by_dst(src, dst)]
    rng = np.random.RandomState(seed)
    for _ in range(max(0, int(n_permutations))):
        lookups.append(_sources_by_dst(src, rng.permutation(dst)))

    stats = []
    for lut in lookups:
        ok = tot = ties = 0
        for user in traj["users"]:
            m_pre = user["m_pre"]
            knows, corrs = user["knows"], user["corrs"]
            nk = m_pre.shape[1]
            active = np.zeros(nk, dtype=bool)
            for i in range(len(corrs)):
                if int(corrs[i]):
                    for d in {int(k) for k in knows[i] if k > 0}:
                        sources = lut.get(d)
                        if sources is None or not active[d]:
                            continue
                        src_arr = sources[active[sources]]
                        if src_arr.size == 0:
                            continue
                        vals = m_pre[i, src_arr]
                        md = m_pre[i, d]
                        ok += int((vals >= md).sum())
                        ties += int((vals == md).sum())
                        tot += int(src_arr.size)
                for k in knows[i]:
                    if k > 0:
                        active[int(k)] = True
        stats.append((ok, tot, ties))

    base_ok, base_tot, base_ties = stats[0]
    if base_tot == 0:
        return empty
    perm = [
        ok / tot for ok, tot, _ in stats[1:] if tot > 0
    ]
    prec = base_ok / base_tot
    perm_mean = float(np.mean(perm)) if perm else None
    return {
        "precedence": prec,
        "precedence_permuted": perm_mean,
        "precedence_gap": (prec - perm_mean) if perm_mean is not None else None,
        "precedence_tie_frac": base_ties / base_tot,
        "n_precedence_samples": int(base_tot),
        "n_precedence_edges": int(src.size),
    }


def _sources_by_dst(src: np.ndarray, dst: np.ndarray) -> Dict[int, np.ndarray]:
    """``{đích: mảng nguồn}`` cho một tập cạnh (dùng để tra nhanh cạnh vào của mỗi đỉnh)."""
    buckets: Dict[int, List[int]] = {}
    for s, d in zip(src.tolist(), dst.tolist()):
        buckets.setdefault(int(d), []).append(int(s))
    return {d: np.asarray(v, dtype="int64") for d, v in buckets.items()}


@torch.no_grad()
def stability_under_perturbation(
    model,
    sequences: Sequence,
    device=None,
    batch_size: int = 16,
    max_users: Optional[int] = 150,
    seed: int = 42,
    concept_num: Optional[int] = None,
    graph_kind: str = "train",
    verbose: bool = True,
) -> dict:
    """Đổi kết quả một bước làm bài thì trạng thái tri thức cuối chuỗi thay đổi bao nhiêu?

    Lật đúng/sai của bước **áp chót** (``L-2``) rồi chạy lại forward; so vector mastery
    ``m_strong[L-1]`` với lần chạy gốc. Bước áp chót được chọn vì nó còn một bước kế tiếp
    để phần thay đổi lan qua cơ chế truy xuất trên đồ thị — lật bước cuối sẽ không lan đi
    đâu và mọi biến thể (kể cả ``zero``) sẽ cho cùng kết quả.

    Hai số đo, đều CÓ THỂ SAI THẬT và phụ thuộc cấu trúc đồ thị:

    * ``stability_spearman`` — tương quan hạng Spearman giữa hai vector mastery trên toàn
      bộ khái niệm (bỏ padding). Càng gần 1 càng ổn định.
    * ``stability_changed_frac`` — tỉ lệ khái niệm có mastery thay đổi. Đồ thị có cạnh
      lan thay đổi sang láng giềng ⇒ tỉ lệ này cao hơn đồ thị ``zero``.

    GRKT dùng ``gumbel_softmax`` ngay khi eval nên ``seed`` phải cố định; hàm tự gọi
    :func:`record_trajectories` hai lần với cùng ``seed`` để hai lượt forward nhận cùng
    chuỗi ngẫu nhiên.
    """
    seqs = list(sequences)
    if max_users is not None:
        seqs = seqs[: int(max_users)]
    kwargs = dict(
        device=device,
        batch_size=batch_size,
        concept_num=concept_num,
        graph_kind=graph_kind,
        seed=seed,
        verbose=False,
    )
    base = record_trajectories(model, seqs, **kwargs)
    flipped = record_trajectories(model, seqs, perturb="flip_penultimate", **kwargs)

    rhos, changed, deltas, n_used = [], [], [], 0
    for before, after in zip(base["users"], flipped["users"]):
        if before["m_strong"].shape != after["m_strong"].shape:
            continue
        if before["m_strong"].shape[0] < 2:
            continue  # không có bước áp chót để lật
        m0 = before["m_strong"][-1, 1:]
        m1 = after["m_strong"][-1, 1:]
        if m0.size < 3:
            continue
        rho = _spearman(m0, m1)
        if rho is not None:
            rhos.append(rho)
        diff = np.abs(m1 - m0)
        changed.append(float((diff > 1e-6).mean()))
        deltas.append(float(diff.mean()))
        n_used += 1
    if verbose:
        print(
            f"  [stability] lật bước áp chót trên {n_used} người học: "
            f"Spearman={_fmt(rhos)} | đổi={_fmt(changed)}"
        )
    return {
        "stability_spearman": float(np.mean(rhos)) if rhos else None,
        "stability_changed_frac": float(np.mean(changed)) if changed else None,
        "stability_abs_delta": float(np.mean(deltas)) if deltas else None,
        "n_stability_users": int(n_used),
        "perturbation": "flip_penultimate",
    }


def _fmt(values) -> str:
    return f"{np.mean(values):.4f}" if values else "n/a"


def _spearman(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Tương quan hạng Spearman với xử lý tie bằng hạng trung bình; ``None`` nếu hằng số."""
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    if ra.std() <= EPS or rb.std() <= EPS:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _average_ranks(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float64")
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype="float64")
    ranks[order] = np.arange(1, x.size + 1, dtype="float64")
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def forgetting_monotonicity(traj: Dict[str, object], n_buckets: int = 5) -> dict:
    """Quỹ đạo mastery phải suy giảm (hoặc đi ngang) theo khoảng cách thời gian.

    Xét ``Δm = m_pre[i+1] - m_strong[i]`` tại các khái niệm KHÔNG bị chạm tới ở bước i và
    i+1 (không phải khái niệm đang làm, không phải khái niệm kế tiếp) — phần thay đổi còn
    lại chính là cơ chế học/quên theo thời gian của GRKT. Nhóm theo ``Δt`` (phút) rồi hồi
    quy trung bình ``Δm`` theo ``Δt``.  Kỳ vọng vật lý: độ dốc ≤ 0 (quên dần theo thời
    gian).  Độ dốc dương có nghĩa mô hình cho rằng "không làm gì cả" cũng tăng độ thành
    thạo — một tiên đề sai, cần báo cáo.
    """
    dt_all, dm_all = [], []
    for user in traj["users"]:
        m_pre, m_strong = user["m_pre"], user["m_strong"]
        knows, times = user["knows"], user["times"]
        if m_pre.shape[0] < 2:
            continue
        touched = np.zeros(m_pre.shape[1], dtype=bool)
        touched[0] = True  # cột padding: không phải khái niệm thật
        for i in range(m_pre.shape[0] - 1):
            touched[1:] = False
            for k in list(knows[i]) + list(knows[i + 1]):
                if k > 0:
                    touched[int(k)] = True
            dm = m_pre[i + 1] - m_strong[i]
            free = ~touched
            if not free.any():
                continue
            gap = float(times[i + 1] - times[i])
            dt_all.append(gap)
            dm_all.append(float(dm[free].mean()))
    if not dt_all:
        return {
            "slope": None,
            "buckets": [],
            "n": 0,
            "pearson_dt_vs_dm": None,
            "dt_zero_frac": 1.0,
        }
    dt = np.asarray(dt_all, dtype="float64")
    dm = np.asarray(dm_all, dtype="float64")
    qs = np.quantile(dt, np.linspace(0, 1, n_buckets + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    bucket_id = np.clip(np.digitize(dt, qs[1:-1], right=True), 0, n_buckets - 1)
    buckets = []
    means, centers = [], []
    for b in range(n_buckets):
        m = bucket_id == b
        if not m.any():
            continue
        mean_dm = float(dm[m].mean())
        med_dt = float(np.median(dt[m]))
        buckets.append(
            {
                "bucket": b,
                "n": int(m.sum()),
                "dt_min_median": med_dt,
                "mean_delta_m": mean_dm,
            }
        )
        means.append(mean_dm)
        centers.append(med_dt)
    slope = None
    if len(means) >= 2 and np.ptp(centers) > 0:
        slope = float(np.polyfit(centers, means, 1)[0])
    return {
        "slope": slope,
        "pearson_dt_vs_dm": float(np.corrcoef(dt, dm)[0, 1]) if len(dt) > 2 else None,
        "buckets": buckets,
        "n": int(dt.size),
        "dt_zero_frac": float((dt == 0).mean()),
    }


def stage_attribution(traj: Dict[str, object]) -> dict:
    """Độ lớn thay đổi mastery trung bình của từng giai đoạn, tại khái niệm đang làm.

    Cho biết mô hình đang "học từ kết quả làm bài" (củng cố) hay "suy luận theo thời
    gian" (học/quên) — phục vụ phần biện luận trong báo cáo.
    """
    strong, hf, retr = [], [], []
    for user in traj["users"]:
        knows = user["knows"]
        dm_s = user["m_strong"] - user["m_pre"]
        for i in range(dm_s.shape[0]):
            ans = [int(k) for k in knows[i] if k > 0]
            if not ans:
                continue
            strong.append(float(np.mean([abs(dm_s[i, a]) for a in ans])))
        if user["m_pre"].shape[0] >= 2:
            hf.append(
                float(np.abs(user["m_pre"][1:, 1:] - user["m_strong"][:-1, 1:]).mean())
            )
        retr.append(float(np.abs(user["m_tilde"][:, 1:] - user["m_pre"][:, 1:]).mean()))
    return {
        "mean_abs_delta_strengthen": float(np.mean(strong)) if strong else None,
        "mean_abs_delta_learn_forget": float(np.mean(hf)) if hf else None,
        "mean_abs_delta_retrieval": float(np.mean(retr)) if retr else None,
        "n_steps": len(strong),
    }


def trajectory_metrics_table(
    runs: Dict[str, Dict[str, object]],
    adj: Optional[np.ndarray] = None,
    pre_map: Optional[np.ndarray] = None,
    stability: Optional[Dict[str, dict]] = None,
    precedence_permutations: int = 5,
    seed: int = 42,
):
    """Bảng tiêu chí cho nhiều lần chạy ``{tên biến thể đồ thị: trajectory dict}``.

    Cột **tiêu chí thật** (có thể sai, dùng làm bằng chứng): ``precedence``,
    ``precedence_gap``, ``stability_spearman``, ``stability_changed_frac``,
    ``forget_slope``, ``pearson_dt_dm``. Cột **kiểm chứng cài đặt** (kỳ vọng = 1, không
    dùng làm bằng chứng): ``consistency_strengthen``, ``consistency_spread``, ``locality``.

    ``adj`` là mặt nạ láng giềng 1-hop của biến thể ``train`` và ``pre_map`` là đồ thị
    tiên quyết nhị phân của ``train`` (``level2.grkt_data.graph_variant``). Cả hai dùng
    CHUNG cho mọi biến thể để "tập láng giềng"/"tập cạnh tiên quyết" cố định, nhờ đó cột
    so sánh được giữa các đồ thị. ``stability`` là ``{biến thể: kết quả
    stability_under_perturbation}``; thiếu thì các cột stability để ``NaN``.
    """
    import pandas as pd

    rows = []
    stability = stability or {}
    for name, traj in runs.items():
        # Chủ đích dùng LẠI adj/pre_map của đồ thị train cho mọi biến thể: "láng giềng" và
        # "cạnh tiên quyết" phải là tập cố định thì số đo mới so sánh được giữa các đồ thị.
        dc = directional_consistency(traj, adj)
        lo = locality(traj, adj)
        fm = forgetting_monotonicity(traj)
        sa = stage_attribution(traj)
        # Tính RIÊNG cho từng biến thể: m_pre phụ thuộc đồ thị, nên "nhất quán tiên quyết"
        # mới là phép so giữa các đồ thị.
        prec = precedence_consistency(
            traj, pre_map, n_permutations=precedence_permutations, seed=seed
        )
        st = stability.get(name) or stability.get(str(traj.get("graph_kind", name))) or {}
        rows.append(
            {
                "graph": str(traj.get("graph_kind", name)),
                "users": len(traj["users"]),
                "steps": dc["n_steps"],
                "precedence": prec["precedence"],
                "precedence_gap": prec["precedence_gap"],
                "n_precedence_edges": prec["n_precedence_edges"],
                "stability_spearman": st.get("stability_spearman"),
                "stability_changed_frac": st.get("stability_changed_frac"),
                "forget_slope": fm["slope"],
                "pearson_dt_dm": fm["pearson_dt_vs_dm"],
                "consistency_strengthen": dc["strengthen"],
                "consistency_spread": dc["spread"],
                "locality": lo["locality"],
                "abs_delta_strengthen": sa["mean_abs_delta_strengthen"],
                "abs_delta_learn_forget": sa["mean_abs_delta_learn_forget"],
                "abs_delta_retrieval": sa["mean_abs_delta_retrieval"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- xuất & vẽ
def save_trajectories(traj: Dict[str, object], out_path) -> Path:
    """Lưu quỹ đạo dạng nén: mỗi người học một biến ``u{i}`` với các mảng theo thứ tự cố định.

    Dùng ``np.savez_compressed`` và KHÔNG dùng ``allow_pickle``: các chuỗi có độ dài khác
    nhau nên được lưu thành mảng riêng theo tên, thay vì gom vào một object array.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        user_ids = np.asarray([u["user_id"] for u in traj["users"]], dtype="int64")
    except (TypeError, ValueError):
        # id dạng chuỗi: lưu thành mảng ký tự cố định, vẫn an toàn khi đọc (không pickle)
        user_ids = np.asarray([str(u["user_id"]) for u in traj["users"]], dtype="<U64")
    payloads = {
        "graph_kind": str(traj.get("graph_kind", "train")),
        "concept_num": int(traj["concept_num"]),
        "n_knows_plus_pad": int(
            traj.get("n_knows_plus_pad", int(traj["concept_num"]) + 1)
        ),
        "seed": int(traj.get("seed", 0)),
        "user_ids": user_ids,
    }
    for i, u in enumerate(traj["users"]):
        payloads[f"u{i}__m_pre"] = u["m_pre"].astype(np.float16)
        payloads[f"u{i}__m_strong"] = u["m_strong"].astype(np.float16)
        payloads[f"u{i}__m_tilde"] = u["m_tilde"].astype(np.float16)
        payloads[f"u{i}__knows"] = u["knows"].astype(np.int32)
        payloads[f"u{i}__corrs"] = u["corrs"].astype(np.int8)
        payloads[f"u{i}__times"] = u["times"].astype(np.int32)
    np.savez_compressed(out_path, **payloads)
    return out_path


# Các mục lưu lại cho từng người học (mastery float16, chỉ số/nguyên float32-or-int)
NPZ_SLOTS = ("m_pre", "m_strong", "m_tilde", "knows", "corrs", "times")
NPZ_DTYPES = {
    "m_pre": np.float32,
    "m_strong": np.float32,
    "m_tilde": np.float32,
    "knows": np.int64,
    "corrs": np.int64,
    "times": np.int64,
}


def load_trajectories(npz_path) -> Dict[str, object]:
    """Đọc lại quỹ đạo đã lưu bằng :func:`save_trajectories` để vẽ/đo mà không train lại."""
    z = np.load(npz_path, allow_pickle=False)
    users = []
    for i, uid in enumerate(z["user_ids"]):
        item = {"user_id": _maybe_id(uid)}
        for slot in NPZ_SLOTS:
            item[slot] = z[f"u{i}__{slot}"].astype(NPZ_DTYPES[slot])
        users.append(item)
    return {
        "users": users,
        "concept_num": int(z["concept_num"]),
        "graph_kind": str(z["graph_kind"]),
        "seed": int(z["seed"]),
    }


def plot_mastery_heatmap(
    traj: Dict[str, object],
    out_path,
    user_index: int = 0,
    max_concepts: int = 24,
    skill_lookup: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
) -> Path:
    """Nhiệt độ mastery theo (khái niệm × thời điểm) của MỘT người học.

    Hàm KHÔNG gọi ``matplotlib.use("Agg")`` — ép backend giữa lúc notebook đang hiển thị
    inline sẽ làm các cell ``plt.show()`` sau đó vẽ ra khoảng trắng. ``fig.savefig`` hoạt động
    với mọi backend, và fig được ``plt.close`` ngay nên không tích tụ figure khi vẽ 4 biến thể.
    """
    import matplotlib.pyplot as plt

    user = traj["users"][user_index]
    concepts = _active_concepts(user, max_concepts)
    mat = (
        user["m_strong"][:, concepts].T
        if concepts.size
        else np.zeros((0, user["m_strong"].shape[1]))
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(max(6.0, 0.22 * mat.shape[1] + 3), max(3.0, 0.26 * mat.shape[0] + 1.6))
    )
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_ylabel("khái niệm")
    ax.set_xlabel("bước tương tác")
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(
        [_concept_name(int(c), skill_lookup) for c in concepts], fontsize=7
    )
    ttl = (
        title
        or f"Mastery m(u,c,t) — user {user['user_id']} [{traj.get('graph_kind', 'train')}]"
    )
    ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=ax, label="sigmoid(m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def plot_mastery_curves(
    traj: Dict[str, object],
    out_path,
    user_indices: Sequence[int] = (0,),
    concepts_per_user: int = 4,
    skill_lookup: Optional[Dict[int, str]] = None,
    title: Optional[str] = None,
) -> Path:
    """Đường mastery theo thời gian của vài khái niệm, mỗi người học một subplot."""
    import matplotlib.pyplot as plt

    user_indices = list(user_indices)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(1, len(user_indices))
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.2), squeeze=False, sharey=True)
    for ax, ui in zip(axes[0], user_indices):
        user = traj["users"][ui]
        concepts = _active_concepts(user, concepts_per_user)
        for c in concepts:
            line = user["m_strong"][:, int(c)]
            ax.plot(
                np.arange(len(line)),
                line,
                linewidth=1.4,
                label=_concept_name(int(c), skill_lookup),
            )
        corr = user["corrs"]
        for i, v in enumerate(corr):
            ax.axvline(
                i, color=("#2e7d32" if v else "#c62828"), alpha=0.10, linewidth=0.9
            )
        ax.set_xlabel("bước")
        ax.set_title(
            f"user {user['user_id']} [{traj.get('graph_kind', 'train')}]", fontsize=9
        )
        ax.legend(fontsize=6, ncol=2, loc="best")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("sigmoid(m)")
    fig.suptitle(
        title or "Quỹ đạo mastery theo thời gian (nền đậm: đúng=xanh, sai=đỏ)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def export_mastery_csv(
    traj: Dict[str, object],
    out_path,
    max_users: int = 50,
    skill_lookup: Optional[Dict[int, str]] = None,
) -> Path:
    """Xuất bảng dài ``m(u,c,t)``: một dòng = (người học, bước, khái niệm, m_pre, m_strong, m_tilde, nhãn).

    Chỉ xuất các khái niệm thực sự xuất hiện trong chuỗi của người học; nếu xuất mọi khái
    niệm thì số dòng nhân lên ``n_knows`` lần mà phần lớn toàn giá trị không đổi.
    """
    import pandas as pd

    rows = []
    for user in traj["users"][: int(max_users)]:
        concepts = set(int(c) for c in np.unique(user["knows"]) if c > 0)
        cols = np.asarray(sorted(concepts), dtype="int64")
        if not cols.size:
            continue
        for i in range(user["corrs"].shape[0]):
            for c in cols:
                rows.append(
                    {
                        "graph": str(traj.get("graph_kind", "train")),
                        "user_id": user["user_id"],
                        "step": i,
                        "concept0": int(c) - 1,
                        "skill": _concept_name(int(c), skill_lookup),
                        "answered": int(bool(np.any(user["knows"][i] == c))),
                        "correct": int(user["corrs"][i]),
                        "time_min": int(user["times"][i]),
                        "m_pre": float(user["m_pre"][i, c]),
                        "m_strong": float(user["m_strong"][i, c]),
                        "m_tilde": float(user["m_tilde"][i, c]),
                    }
                )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def _active_concepts(user: dict, limit: int) -> np.ndarray:
    """Các khái niệm hay được dùng nhất, trả về KHÔNG GIAN CHỈ SỐ CÓ PADDING (1..NK).

    Đây đúng là chỉ số cột của ``knows`` và của các mảng mastery, nên dùng trực tiếp để lấy
    cột; muốn tra tên khái niệm thì trừ 1 để về không gian 0-based của Cấp độ 1.
    """
    knows = user["knows"]
    counts: Dict[int, int] = {}
    first: Dict[int, int] = {}
    for i, ks in enumerate(knows):
        for k in ks:
            if k <= 0:
                continue
            c = int(k)
            counts[c] = counts.get(c, 0) + 1
            first.setdefault(c, i)
    order = sorted(counts, key=lambda c: (-counts[c], first[c]))[: max(1, int(limit))]
    return np.asarray(order, dtype="int64")


def _concept_name(padded_index: int, skill_lookup: Optional[Dict[int, str]]) -> str:
    """Nhãn khái niệm: tra tên gốc theo không gian 0-based của Cấp độ 1, fallback là số."""
    key = padded_index - 1
    name = (skill_lookup or {}).get(key)
    return str(name if name is not None else f"c{key}")[:18]


def _maybe_id(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return str(x)
