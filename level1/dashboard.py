"""Learner-level view: pick a learner and inspect predictions, history, knowledge state, evidence.

Produces a JSON artifact (learner_view.json) and, when ipywidgets is available,
an in-notebook dropdown picker. Falls back to printing top learners otherwise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


def _skill_name(lookup, skill_id):
    if lookup and skill_id in lookup:
        return f"{skill_id}:{lookup[skill_id]}"
    return str(skill_id)


def export_learner_view(
    model,
    loader,
    skill_lookup=None,
    graph=None,
    device=None,
    model_type="GKT",
    out_path=None,
    max_learners=None,
):
    """Run the model on every learner in `loader` and collect per-step records.

    Returns a list of dicts, one per learner:
      {user_id, steps: [{t, skill, skill_name, y_true, y_prob, y_pred}...]}
    For GKT, also stores per-concept knowledge-state norms at each step (state_norm).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    from .explain import user_ids_from_loader

    user_ids = user_ids_from_loader(loader)
    views = []
    batch_offset = 0
    with torch.no_grad():
        for batch in loader:
            features, questions, answers = [x.to(device) for x in batch]
            if model_type == "GKT" and getattr(model, "graph_type", None) is not None:
                out = model(features, questions, return_state=True)
                pred = out[0]
                states = out[4]
            else:
                out = model(features, questions)
                pred = out[0] if isinstance(out, tuple) else out
                states = None
            q_seq = questions
            real = answers[:, 1:]
            mask = (real != -1) & (q_seq[:, 1:] != -1)
            bsz, tlen = pred.shape
            for b in range(bsz):
                uid = (
                    user_ids[batch_offset + b]
                    if batch_offset + b < len(user_ids)
                    else None
                )
                steps = []
                for t in range(tlen):
                    if not bool(mask[b, t]):
                        continue
                    skill = int(q_seq[b, t + 1].cpu())
                    yhat = float(pred[b, t].cpu())
                    y = int(real[b, t].cpu())
                    step = {
                        "t": t + 1,
                        "skill": skill,
                        "skill_name": _skill_name(skill_lookup, skill),
                        "y_true": y,
                        "y_prob": round(yhat, 4),
                        "y_pred": int(yhat >= 0.5),
                    }
                    if states is not None and t < len(states):
                        # Frobenius-style per-concept norm of the hidden knowledge state at this step
                        h = states[t][b]  # [concept_num, hidden_dim]
                        step["state_norm"] = [
                            round(float(x), 4) for x in h.norm(dim=-1).cpu().tolist()
                        ]
                    steps.append(step)
                views.append({"user_id": uid, "n_steps": len(steps), "steps": steps})
                if max_learners is not None and len(views) >= max_learners:
                    break
            batch_offset += bsz
            if max_learners is not None and len(views) >= max_learners:
                break
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(views, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return views


def render_learner_view(views, learner_index=0):
    """Plain-text rendering of one learner's view (used as fallback when no widgets)."""
    if not views:
        return "(no learners)"
    v = views[learner_index]
    lines = [f"Learner user_id={v['user_id']} | {v['n_steps']} steps"]
    for s in v["steps"]:
        marker = "OK" if s["y_pred"] == s["y_true"] else "MISS"
        lines.append(
            f"  t={s['t']:>3}  skill={s['skill_name']:<20}  p={s['y_prob']:.3f}  "
            f"true={s['y_true']}  pred={s['y_pred']}  [{marker}]"
        )
    return "\n".join(lines)


def interactive_learner_view(views):
    """Show an ipywidgets dropdown if available; otherwise print the first learner."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print(render_learner_view(views, 0))
        return
    options = [
        (f"user {v['user_id']} ({v['n_steps']} steps)", i) for i, v in enumerate(views)
    ]
    dd = widgets.Dropdown(options=options, description="Learner:")
    out = widgets.Output()

    def _on_change(change):
        out.clear_output(wait=True)
        with out:
            print(render_learner_view(views, change["new"]))

    dd.observe(_on_change, names="value")
    display(dd, out)
    print(render_learner_view(views, options[0][1]))
