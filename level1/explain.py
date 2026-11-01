"""Pick 5 correct and 5 incorrect next-step predictions and attach history/graph evidence."""

from __future__ import annotations

from collections import defaultdict

import torch


def skill_lookup_from_loader(loader):
    ds = loader.dataset
    if hasattr(ds, "dataset"):
        ds = ds.dataset
    return getattr(ds, "skill_lookup", {}) or {}


def user_ids_from_loader(loader):
    """Map each batch position to the underlying user_id (loader must be non-shuffled, e.g. test_loader)."""
    ds = loader.dataset
    if hasattr(ds, "indices") and hasattr(ds.dataset, "user_ids"):
        base_ids = ds.dataset.user_ids
        return [base_ids[i] for i in ds.indices]
    return list(range(len(ds)))


def _skill_name(lookup, skill_id):
    if lookup and skill_id in lookup:
        return f"{skill_id}:{lookup[skill_id]}"
    return str(skill_id)


def _neighbors(graph, skill_id, topk=5):
    if graph is None:
        return []
    row = graph[skill_id].detach().cpu()
    if row.numel() == 0:
        return []
    values, idx = torch.topk(row, k=min(topk, row.numel()))
    out = []
    for v, i in zip(values.tolist(), idx.tolist()):
        if i == skill_id or v <= 0:
            continue
        out.append((int(i), float(v)))
    return out


def _dataset_rows(loader):
    """Rows of the underlying dataset, aligned to the positions of a NON-shuffled loader.

    collect_cases walks test batches in order (DataLoader(shuffle=False)), so position
    batch_offset+b identifies exactly one learner sequence.
    """
    ds = loader.dataset
    base = getattr(ds, "dataset", ds)
    indices = getattr(ds, "indices", range(len(base)))
    rows = []
    for i in indices:
        feats, qs, ans = base[int(i)]
        to_list = lambda x: x.tolist() if hasattr(x, "tolist") else list(x)
        rows.append((to_list(feats), to_list(qs), to_list(ans)))
    return rows


def _flip(feats, qs, h, res_len=2):
    """Counterfactual at step h: same skill, opposite outcome. Returns new feature value or None."""
    skill = qs[h]
    if skill < 0 or feats[h] < 0:
        return None
    correct = feats[h] - res_len * skill  # binary encoding: feature = skill*2 + correct
    if correct not in (0, 1):
        return None
    return res_len * skill + (1 - correct)


def _edge_evidence(graph, skill, history, topk=3):
    """Graph edges out of/in to the target skill, annotated with when the learner touched them."""
    if graph is None:
        return []
    row = graph[skill].detach().cpu()
    col = graph[:, skill].detach().cpu()
    touched = {}
    for h in history:
        prev = touched.get(h["skill"])
        if prev is None or h["t"] > prev["t"]:
            touched[h["skill"]] = h
    scored = []
    for c in range(row.numel()):
        if c == skill:
            continue
        w = max(float(row[c]), float(col[c]))
        if w <= 0:
            continue
        scored.append((w, c))
    scored.sort(reverse=True)
    out = []
    for w, c in scored[:topk]:
        hit = touched.get(c)
        out.append(
            {
                "skill": int(c),
                "weight": w,
                "direction": "out" if float(row[c]) >= float(col[c]) else "in",
                "last_t": None if hit is None else hit["t"],
                "last_correct": None if hit is None else hit["correct"],
            }
        )
    return out


@torch.no_grad()
def annotate_evidence(
    model,
    loader,
    buckets,
    graph=None,
    device=None,
    max_flips=8,
    graph_ablation=True,
    res_len=2,
    tol=0.02,
):
    """Attach computed counterfactual evidence to each case record (in place).

    Two numbers per case, both from real re-runs of the model:
      * delta_flip  — change in P(correct) when one past attempt is flipped correct<->wrong.
      * delta_graph — change in P(correct) when every graph edge is removed (A := 0) while the
        learner's own history stays untouched.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    rows = _dataset_rows(loader)
    has_graph = (
        graph_ablation
        and hasattr(model, "graph")
        and isinstance(getattr(model, "graph"), torch.nn.Parameter)
    )
    for rec in [r for bucket in buckets.values() for r in bucket]:
        pos, t = rec.get("pos"), rec.get("t")
        if pos is None or t is None or pos >= len(rows):
            rec["evidence"] = {"ok": False, "reason": "position not available"}
            continue
        feats, qs, _ = rows[pos]
        idx = t - 1  # pred index t-1 predicts the attempt at time t
        if idx < 0 or idx >= len(feats) - 1:
            rec["evidence"] = {"ok": False, "reason": "target step outside sequence"}
            continue
        batch = torch.LongTensor([feats]).to(device)
        qbatch = torch.LongTensor([qs]).to(device)
        p_recheck = float(_forward_pred(model, batch, qbatch, idx, device))
        rec["evidence"] = {
            "ok": abs(p_recheck - rec["y_prob"]) <= tol,
            "p_recheck": p_recheck,
        }
        if not rec["evidence"]["ok"]:
            rec["evidence"][
                "reason"
            ] = f"re-run p={p_recheck:.4f} != logged p={rec['y_prob']:.4f}"
            continue
        history = [h for h in rec["history"] if 0 <= h["t"] <= idx]
        flips = []
        for h in history[-max_flips:]:
            new_feat = _flip(feats, qs, h["t"], res_len=res_len)
            if new_feat is None:
                continue
            trial = batch.clone()
            trial[0, h["t"]] = new_feat
            p_flip = float(_forward_pred(model, trial, qbatch, idx, device))
            flips.append(
                {
                    "t": h["t"],
                    "skill": h["skill"],
                    "skill_name": h["skill_name"],
                    "was_correct": h["correct"],
                    "p": p_flip,
                    "delta": p_flip - p_recheck,
                }
            )
        flips.sort(key=lambda f: abs(f["delta"]), reverse=True)
        rec["evidence"]["flips"] = flips
        if has_graph:
            old = model.graph.data
            try:
                model.graph.data = torch.zeros_like(old)
                p_nograph = float(_forward_pred(model, batch, qbatch, idx, device))
            finally:
                model.graph.data = old
            rec["evidence"]["nograph_p"] = p_nograph
            rec["evidence"]["delta_graph"] = p_recheck - p_nograph
        rec["evidence"]["edges"] = _edge_evidence(graph, rec["skill"], history)
    return buckets


def _forward_pred(model, features, questions, idx, device):
    out = model(features, questions)
    pred = out[0] if isinstance(out, tuple) else out
    return pred[0, idx]


def collect_cases(
    model,
    loader,
    graph=None,
    skill_lookup=None,
    n_each=5,
    history_k=8,
    device=None,
    model_type="GKT",
    evidence=True,
    distinct_skills=True,
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    buckets = {"correct": [], "incorrect": []}
    user_ids = user_ids_from_loader(loader)
    batch_offset = 0
    with torch.no_grad():
        for batch in loader:
            features, questions, answers = [x.to(device) for x in batch]
            out = model(features, questions)
            pred = out[0] if isinstance(out, tuple) else out
            q_seq = questions
            real = answers[:, 1:]
            mask = (real != -1) & (q_seq[:, 1:] != -1)
            bsz, tlen = pred.shape
            for b in range(bsz):
                pos = batch_offset + b
                uid = user_ids[pos] if pos < len(user_ids) else None
                for t in range(tlen):
                    if not bool(mask[b, t]):
                        continue
                    yhat = float(pred[b, t].cpu())
                    y = int(real[b, t].cpu())
                    pred_bin = int(yhat >= 0.5)
                    ok = pred_bin == y
                    skill = int(q_seq[b, t + 1].cpu())
                    hist = []
                    start = max(0, t + 1 - history_k)
                    for h in range(start, t + 1):
                        sid = int(q_seq[b, h].cpu())
                        if sid < 0:
                            continue
                        hist.append(
                            {
                                "t": h,
                                "skill": sid,
                                "skill_name": _skill_name(skill_lookup, sid),
                                "correct": int(answers[b, h].cpu()),
                            }
                        )
                    rec = {
                        "pos": pos,
                        "user_id": uid,
                        "t": t + 1,
                        "skill": skill,
                        "skill_name": _skill_name(skill_lookup, skill),
                        "y_true": y,
                        "y_prob": yhat,
                        "y_pred": pred_bin,
                        "confidence": abs(yhat - 0.5),
                        "history": hist,
                        "neighbors": _neighbors(graph, skill),
                    }
                    key = "correct" if ok else "incorrect"
                    buckets[key].append(rec)
            batch_offset += bsz

    for key in buckets:
        buckets[key].sort(key=lambda r: r["confidence"], reverse=True)
        buckets[key] = _select(buckets[key], n_each, distinct_skills)
    if evidence:
        try:
            annotate_evidence(model, loader, buckets, graph=graph, device=device)
        except Exception as exc:  # never break a training run over the extras
            print(
                f"[explain] bỏ qua bằng chứng phản chứng: {type(exc).__name__}: {exc}"
            )
    return buckets


def _select(rows, n_each, distinct_skills):
    if not distinct_skills:
        return rows[:n_each]
    picked, seen = [], set()
    for r in rows:
        if r["skill"] in seen:
            continue
        seen.add(r["skill"])
        picked.append(r)
        if len(picked) == n_each:
            break
    if len(picked) < n_each:
        chosen = {id(r) for r in picked}
        picked += [r for r in rows if id(r) not in chosen][: n_each - len(picked)]
    return picked


def format_cases(buckets, graph_note="transition graph (train only)") -> str:
    lines = []
    for label, rows in buckets.items():
        lines.append(f"## {len(rows)} {label} predictions")
        for i, rec in enumerate(rows, 1):
            lines.append(
                f"\n### {label} #{i}  user={rec.get('user_id')}  t={rec.get('t')}  skill={rec['skill_name']}  "
                f"y={rec['y_true']}  p={rec['y_prob']:.3f}  pred={rec['y_pred']}"
            )
            hist_txt = ", ".join(
                f"t{h['t']}:{h['skill_name']}={'OK' if h['correct']==1 else 'NG'}"
                for h in rec["history"]
            )
            lines.append(f"- history ({len(rec['history'])} steps): {hist_txt}")
            same = [h for h in rec["history"] if h["skill"] == rec["skill"]]
            if same:
                rate = sum(h["correct"] for h in same) / len(same)
                lines.append(
                    f"- same-skill history: {len(same)} tries, accuracy={rate:.2f}"
                )
            else:
                lines.append("- same-skill history: none (cold skill for this student)")
            if rec["neighbors"]:
                neigh = ", ".join(f"{sid}:{w:.3f}" for sid, w in rec["neighbors"][:5])
                lines.append(f"- graph neighbors ({graph_note}): {neigh}")
            lines.extend(_evidence_lines(rec))
            why = _explain_one(rec)
            lines.append(f"- why: {why}")
    return "\n".join(lines)


def _evidence_lines(rec):
    ev = rec.get("evidence")
    if ev is None:
        return []
    if not ev.get("ok"):
        return [f"- counterfactual: not available ({ev.get('reason', 'not computed')})"]
    lines = []
    flips = ev.get("flips") or []
    if flips:
        top = flips[0]
        others = "; ".join(
            f"t{f['t']} {f['skill_name']} "
            f"({'OK' if f['was_correct'] else 'NG'}->{'NG' if f['was_correct'] else 'OK'}): "
            f"dp={f['delta']:+.3f}"
            for f in flips[1:4]
        )
        lines.append(
            f"- counterfactual (flip one past attempt): t{top['t']} {top['skill_name']} "
            f"({'correct' if top['was_correct'] else 'wrong'} -> opposite) moves p "
            f"{rec['y_prob']:.3f} -> {top['p']:.3f} (dp={top['delta']:+.3f})"
        )
        if others:
            lines.append(f"  - next largest: {others}")
        same = [f for f in flips if f["skill"] == rec["skill"]]
        if same:
            mean = sum(f["delta"] for f in same) / len(same)
            lines.append(
                f"  - same-skill attempts: {len(same)} flipped, mean dp={mean:+.3f} "
                f"(opposite of all of them at once is not modelled)"
            )
        else:
            lines.append(
                "  - no past attempt on this skill in the window: p moves only via other skills"
            )
    if "delta_graph" in ev:
        lines.append(
            f"- graph ablation for THIS prediction: A := 0 moves p {rec['y_prob']:.3f} -> "
            f"{ev['nograph_p']:.3f} (dp={ev['nograph_p'] - rec['y_prob']:+.3f})"
        )
    edges = ev.get("edges") or []
    if edges:
        txt = ", ".join(
            f"{e['skill']} w={e['weight']:.3f} {e['direction']}"
            + (
                f" (learner last touched t{e['last_t']}, {'OK' if e['last_correct'] else 'NG'})"
                if e["last_t"] is not None
                else " (never touched by this learner)"
            )
            for e in edges
        )
        lines.append(f"- strongest edges at the target concept: {txt}")
    return lines


def _explain_one(rec) -> str:
    ev = rec.get("evidence") or {}
    if ev.get("ok"):
        flips = ev.get("flips") or []
        parts = []
        if flips:
            strongest = flips[0]
            if abs(strongest["delta"]) < 1e-3:
                parts.append(
                    "no single past attempt in the window moves p by more than 0.001"
                )
            else:
                direction = "raises" if strongest["delta"] > 0 else "lowers"
                parts.append(
                    f"the single most influential step is t{strongest['t']} ({strongest['skill_name']}): "
                    f"flipping its outcome {direction} p by {abs(strongest['delta']):.3f}"
                )
        if "delta_graph" in ev:
            dg = ev["delta_graph"]
            if abs(dg) < 1e-3:
                parts.append(
                    "removing every edge leaves this score unchanged (|dp| < 0.001)"
                )
            else:
                role = "raises" if dg > 0 else "lowers"
                parts.append(
                    f"the graph {role} this score by {abs(dg):.3f} "
                    f"(p {rec['y_prob']:.3f} vs {ev['nograph_p']:.3f} with A := 0)"
                )
        same = [h for h in rec["history"] if h["skill"] == rec["skill"]]
        if same:
            rate = sum(h["correct"] for h in same) / len(same)
            parts.append(
                f"same-skill record in window = {rate:.2f} over {len(same)} attempts"
            )
        if parts:
            head = "Correct call:" if rec["y_pred"] == rec["y_true"] else "Wrong call:"
            return head + " " + "; ".join(parts) + "."
    same = [h for h in rec["history"] if h["skill"] == rec["skill"]]
    recent_ok = sum(h["correct"] for h in rec["history"][-3:]) if rec["history"] else 0
    if rec["y_pred"] == rec["y_true"] and rec["y_true"] == 1:
        if same and sum(h["correct"] for h in same) >= 1:
            return "History already has correct answers on this skill, so the model raises P(correct)."
        return "Recent overall success is high even without the same skill, so the model extrapolates."
    if rec["y_pred"] == rec["y_true"] and rec["y_true"] == 0:
        return "Past mistakes on related/same skills keep predicted mastery low."
    if rec["y_pred"] == 1 and rec["y_true"] == 0:
        return (
            f"False positive: model saw {recent_ok}/3 recent successes and over-generalized "
            f"to skill {rec['skill_name']}."
        )
    return (
        f"False negative: true label is correct but predicted p={rec['y_prob']:.3f}; "
        "sparse same-skill evidence or conflicting neighbors likely pulled the score down."
    )


def summarize_skill_evidence(buckets) -> dict:
    """Count how often same-skill history agrees with the prediction."""
    stats = defaultdict(int)
    for rows in buckets.values():
        for rec in rows:
            same = [h for h in rec["history"] if h["skill"] == rec["skill"]]
            stats["cases"] += 1
            stats["with_same_skill"] += int(len(same) > 0)
    return dict(stats)
