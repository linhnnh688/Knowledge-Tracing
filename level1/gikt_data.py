"""Build question–skill sequences and a train-only bipartite graph for GIKT."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class GIKTDataset(Dataset):
    def __init__(self, questions, skills, answers, user_ids):
        self.questions = questions
        self.skills = skills
        self.answers = answers
        self.user_ids = user_ids

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.skills[idx], self.answers[idx]


def _pad_collate(batch):
    qs, ss, ans = zip(*batch)
    qs = [torch.LongTensor(x) for x in qs]
    ss = [torch.LongTensor(x) for x in ss]
    ans = [torch.LongTensor(x) for x in ans]
    q_pad = pad_sequence(qs, batch_first=True, padding_value=0)
    s_pad = pad_sequence(ss, batch_first=True, padding_value=0)
    a_pad = pad_sequence(ans, batch_first=True, padding_value=-1)
    return q_pad, s_pad, a_pad


def load_gikt_dataset(
    file_path,
    train_user_ids,
    val_user_ids,
    test_user_ids,
    batch_size=32,
    max_seq_len=200,
    min_seq_len=3,
    question_neighbor_num=4,
    skill_neighbor_num=4,
    seed=42,
):
    df = pd.read_csv(file_path)
    if "problem_id" not in df.columns:
        df["problem_id"] = df["skill_id"]
    df = df.dropna(subset=["skill_id", "problem_id", "correct", "user_id"])
    df = df.groupby("user_id").filter(lambda q: len(q) >= min_seq_len).copy()
    df["skill"], _ = pd.factorize(df["skill_id"], sort=True)
    df["question"], _ = pd.factorize(df["problem_id"], sort=True)
    # reserve 0 for padding
    df["skill"] = df["skill"] + 1
    df["question"] = df["question"] + 1
    df["correct"] = df["correct"].eq(1).astype(int)

    num_skills = int(df["skill"].max()) + 1
    num_questions = int(df["question"].max()) + 1

    seqs = {}
    for user_id, series in df.groupby("user_id", sort=False):
        q = series["question"].tolist()
        s = series["skill"].tolist()
        a = series["correct"].tolist()
        if max_seq_len and len(q) > max_seq_len:
            q, s, a = q[:max_seq_len], s[:max_seq_len], a[:max_seq_len]
        try:
            uid = int(user_id)
        except (TypeError, ValueError):
            uid = user_id
        seqs[uid] = (q, s, a)

    def subset(user_ids):
        questions, skills, answers, kept = [], [], [], []
        for uid in user_ids:
            if uid not in seqs:
                continue
            q, s, a = seqs[uid]
            questions.append(q)
            skills.append(s)
            answers.append(a)
            kept.append(uid)
        return GIKTDataset(questions, skills, answers, kept)

    train_ds = subset(train_user_ids)
    val_ds = subset(val_user_ids)
    test_ds = subset(test_user_ids)

    rng = np.random.RandomState(seed)
    q_neighbors, s_neighbors = _build_neighbors(
        df[df["user_id"].isin(set(train_ds.user_ids))],
        num_questions,
        num_skills,
        question_neighbor_num,
        skill_neighbor_num,
        rng,
    )

    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_pad_collate),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_pad_collate),
        "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=_pad_collate),
    }
    return {
        "num_questions": num_questions,
        "num_skills": num_skills,
        "q_neighbors": q_neighbors,
        "s_neighbors": s_neighbors,
        "loaders": loaders,
        "datasets": {"train": train_ds, "val": val_ds, "test": test_ds},
    }


def _build_neighbors(train_df, num_questions, num_skills, q_num, s_num, rng):
    skill_to_q = defaultdict(set)
    q_to_skill = defaultdict(set)
    for q, s in zip(train_df["question"].tolist(), train_df["skill"].tolist()):
        q_to_skill[int(q)].add(int(s))
        skill_to_q[int(s)].add(int(q))

    q_neighbors = torch.zeros(num_questions, q_num, dtype=torch.long)
    s_neighbors = torch.zeros(num_skills, s_num, dtype=torch.long)
    for q, skills in q_to_skill.items():
        arr = np.array(list(skills), dtype=np.int64)
        if len(arr) == 0:
            continue
        choice = rng.choice(arr, size=q_num, replace=len(arr) < q_num)
        q_neighbors[q] = torch.from_numpy(choice)
    for s, questions in skill_to_q.items():
        arr = np.array(list(questions), dtype=np.int64)
        if len(arr) == 0:
            continue
        choice = rng.choice(arr, size=s_num, replace=len(arr) < s_num)
        s_neighbors[s] = torch.from_numpy(choice)
    return q_neighbors, s_neighbors


def identity_neighbors(num_questions, num_skills, q_num, s_num):
    """Ablation: empty bipartite graph (padding neighbors only)."""
    return (
        torch.zeros(num_questions, q_num, dtype=torch.long),
        torch.zeros(num_skills, s_num, dtype=torch.long),
    )
