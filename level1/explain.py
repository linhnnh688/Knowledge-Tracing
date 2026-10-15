"""Pick 5 correct and 5 incorrect next-step predictions and attach history/graph evidence."""

from __future__ import annotations

from collections import defaultdict

import torch


def skill_lookup_from_loader(loader):
    ds = loader.dataset
    if hasattr(ds, "dataset"):
        ds = ds.dataset
    return getattr(ds, "skill_lookup", {}) or {}


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


def collect_cases(
    model,
    loader,
    graph=None,
    skill_lookup=None,
    n_each=5,
    history_k=8,
    device=None,
    model_type="GKT",
):
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    buckets = {"correct": [], "incorrect": []}
    with torch.no_grad():
        for batch in loader:
            if model_type == "GIKT":
                questions, skills, answers = batch
                questions = questions.to(device)
                answers = answers.to(device)
                pred = model(questions, answers)
                q_seq = questions
            else:
                features, questions, answers = [x.to(device) for x in batch]
                out = model(features, questions)
                pred = out[0] if isinstance(out, tuple) else out
                q_seq = questions
            real = answers[:, 1:]
            mask = real != -1
            if model_type == "GIKT":
                mask = mask & (q_seq[:, 1:] > 0)
            else:
                mask = mask & (q_seq[:, 1:] != -1)
            bsz, tlen = pred.shape
            for b in range(bsz):
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
                        if model_type == "GIKT" and sid <= 0:
                            continue
                        if model_type != "GIKT" and sid < 0:
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
                        "skill": skill,
                        "skill_name": _skill_name(skill_lookup, skill),
                        "y_true": y,
                        "y_prob": yhat,
                        "y_pred": pred_bin,
                        "confidence": abs(yhat - 0.5),
                        "history": hist,
                        "neighbors": _neighbors(graph, skill) if model_type != "GIKT" else [],
                    }
                    key = "correct" if ok else "incorrect"
                    buckets[key].append(rec)

    for key in buckets:
        buckets[key].sort(key=lambda r: r["confidence"], reverse=True)
        buckets[key] = buckets[key][:n_each]
    return buckets


def format_cases(buckets, graph_note="transition graph (train only)") -> str:
    lines = []
    for label, rows in buckets.items():
        lines.append(f"## {len(rows)} {label} predictions")
        for i, rec in enumerate(rows, 1):
            lines.append(
                f"\n### {label} #{i}  skill={rec['skill_name']}  "
                f"y={rec['y_true']}  p={rec['y_prob']:.3f}  pred={rec['y_pred']}"
            )
            hist_txt = ", ".join(
                f"t{h['t']}:{h['skill_name']}={'OK' if h['correct']==1 else 'NG'}" for h in rec["history"]
            )
            lines.append(f"- history ({len(rec['history'])} steps): {hist_txt}")
            same = [h for h in rec["history"] if h["skill"] == rec["skill"]]
            if same:
                rate = sum(h["correct"] for h in same) / len(same)
                lines.append(f"- same-skill history: {len(same)} tries, accuracy={rate:.2f}")
            else:
                lines.append("- same-skill history: none (cold skill for this student)")
            if rec["neighbors"]:
                neigh = ", ".join(f"{sid}:{w:.3f}" for sid, w in rec["neighbors"][:5])
                lines.append(f"- graph neighbors ({graph_note}): {neigh}")
            why = _explain_one(rec)
            lines.append(f"- why: {why}")
    return "\n".join(lines)


def _explain_one(rec) -> str:
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
