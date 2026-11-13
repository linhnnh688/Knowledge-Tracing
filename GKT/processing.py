import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import (
    build_dense_graph,
    build_identity_graph,
    build_zero_graph,
    build_shuffled_graph,
)

# Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
# For more information, please refer to https://dl.acm.org/doi/10.1145/3350546.3352513
# Author: jhljx
# Email: jhljx8918@gmail.com


class KTDataset(Dataset):
    def __init__(self, features, questions, answers):
        super(KTDataset, self).__init__()
        self.features = features
        self.questions = questions
        self.answers = answers

    def __getitem__(self, index):
        return self.features[index], self.questions[index], self.answers[index]

    def __len__(self):
        return len(self.features)


def pad_collate(batch):
    features, questions, answers = zip(*batch)
    features = [torch.LongTensor(feat) for feat in features]
    questions = [torch.LongTensor(qt) for qt in questions]
    answers = [torch.LongTensor(ans) for ans in answers]
    feature_pad = pad_sequence(features, batch_first=True, padding_value=-1)
    question_pad = pad_sequence(questions, batch_first=True, padding_value=-1)
    answer_pad = pad_sequence(answers, batch_first=True, padding_value=-1)
    return feature_pad, question_pad, answer_pad


def load_dataset(
    file_path,
    batch_size,
    graph_type,
    dkt_graph_path=None,
    train_ratio=0.7,
    val_ratio=0.2,
    shuffle=True,
    model_type="GKT",
    use_binary=True,
    res_len=2,
    use_cuda=True,
    seed=42,
    max_seq_len=200,
    min_seq_len=3,
):
    r"""
    Parameters:
        file_path: input file path of knowledge tracing data
        batch_size: the size of a student batch
        graph_type: the type of the concept graph
        shuffle: whether to shuffle the dataset or not
        use_cuda: whether to use GPU to accelerate training speed
        seed: RNG seed for the user-wise train/val/test split
        max_seq_len: truncate each student sequence to this length (0 = no truncate)
        min_seq_len: drop students with fewer interactions than this
    Return:
        concept_num: the number of all concepts(or questions)
        graph: the static graph is graph type is in ['Dense', 'Transition', 'DKT', 'Identity', 'Zero'], otherwise graph is None
        train_data_loader: data loader of the training dataset
        valid_data_loader: data loader of the validation dataset
        test_data_loader: data loader of the test dataset
    NOTE: stole some code from https://github.com/lccasagrande/Deep-Knowledge-Tracing/blob/master/deepkt/data_util.py
    """
    df = pd.read_csv(file_path)
    if "skill_id" not in df.columns:
        raise KeyError(f"The column 'skill_id' was not found on {file_path}")
    if "correct" not in df.columns:
        raise KeyError(f"The column 'correct' was not found on {file_path}")
    if "user_id" not in df.columns:
        raise KeyError(f"The column 'user_id' was not found on {file_path}")

    # if not (df['correct'].isin([0, 1])).all():
    #     raise KeyError(f"The values of the column 'correct' must be 0 or 1.")

    # Step 1 - Drop rows without skill, restore chronological order, merge multi-skill
    # duplicates and remove users with too few answers (see prepare_interactions)
    df, prep_stats = prepare_interactions(df, min_seq_len=min_seq_len)

    # Step 2 - Enumerate skill id
    df["skill"], _ = pd.factorize(
        df["skill_id"], sort=True
    )  # we can also use problem_id to represent exercises

    # Step 3 - Cross skill id with answer to form a synthetic feature
    # use_binary: (0,1); !use_binary: (1,2,3,4,5,6,7,8,9,10,11,12). Either way, the correct result index is guaranteed to be 1
    if use_binary:
        df["skill_with_answer"] = df["skill"] * 2 + df["correct"]
    else:
        df["skill_with_answer"] = df["skill"] * res_len + df["correct"] - 1

    # Step 4 - Convert to a sequence per user id and shift features 1 timestep
    feature_list = []
    question_list = []
    answer_list = []
    seq_len_list = []
    user_id_list = []
    skill_name_list = []
    order_id_list = []
    has_order = "order_id" in df.columns

    def get_data(user_id, series):
        feats = series["skill_with_answer"].tolist()
        qs = series["skill"].tolist()
        ans = series["correct"].eq(1).astype("int").tolist()
        orders = series["order_id"].tolist() if has_order else []
        if max_seq_len and len(feats) > max_seq_len:
            feats = feats[:max_seq_len]
            qs = qs[:max_seq_len]
            ans = ans[:max_seq_len]
            orders = orders[:max_seq_len]
        feature_list.append(feats)
        question_list.append(qs)
        answer_list.append(ans)
        order_id_list.append(orders)
        seq_len_list.append(len(ans))
        try:
            user_id_list.append(int(user_id))
        except (TypeError, ValueError):
            user_id_list.append(user_id)
        if "skill_name" in series.columns:
            skill_name_list.append(series["skill_name"].tolist()[: len(ans)])

    for user_id, series in df.groupby("user_id", sort=False):
        get_data(user_id, series)
    print("max seq_len: ", int(np.max(seq_len_list)))
    student_num = len(seq_len_list)
    print("student num: ", student_num)
    feature_dim = int(df["skill_with_answer"].max() + 1)
    print("feature_dim: ", feature_dim)
    question_dim = int(df["skill"].max() + 1)
    print("question_dim: ", question_dim)
    concept_num = question_dim

    # print('feature_dim:', feature_dim, 'res_len*question_dim:', res_len*question_dim)
    # assert feature_dim == res_len * question_dim

    kt_dataset = KTDataset(feature_list, question_list, answer_list)
    kt_dataset.user_ids = user_id_list
    kt_dataset.skill_lookup = _skill_lookup(df)
    kt_dataset.order_ids = order_id_list
    kt_dataset.prep_stats = prep_stats
    train_size = int(train_ratio * student_num)
    val_size = int(val_ratio * student_num)
    test_size = student_num - train_size - val_size
    split_gen = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        kt_dataset, [train_size, val_size, test_size], generator=split_gen
    )
    print("train_size: ", train_size, "val_size: ", val_size, "test_size: ", test_size)

    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_collate
    )
    valid_data_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate
    )

    graph = None
    if model_type == "GKT":
        if graph_type == "Dense":
            graph = build_dense_graph(concept_num)
        elif graph_type == "Transition":
            graph = build_transition_graph(
                question_list,
                seq_len_list,
                train_dataset.indices,
                student_num,
                concept_num,
            )
        elif graph_type == "Identity":
            graph = build_identity_graph(concept_num)
        elif graph_type == "Zero":
            graph = build_zero_graph(concept_num)
        elif graph_type == "Shuffled":
            # Edge-shuffle ablation: same non-zero pattern as Transition (from train only) but destinations shuffled
            base = build_transition_graph(
                question_list,
                seq_len_list,
                train_dataset.indices,
                student_num,
                concept_num,
            )
            graph = build_shuffled_graph(base.numpy(), seed=seed)
        elif graph_type == "DKT":
            graph = build_dkt_graph(dkt_graph_path, concept_num)
        static_types = ["Dense", "Transition", "DKT", "Identity", "Zero", "Shuffled"]
        if use_cuda and graph_type in static_types and graph is not None:
            graph = graph.cuda()
    return concept_num, graph, train_data_loader, valid_data_loader, test_data_loader


# Bump whenever prepare_interactions changes the sequences: notebooks lock it in run_config /
# resume fingerprints so checkpoints trained on an older preprocessing are never reused.
PREP_VERSION = "order-dedup-v1"


def prepare_interactions(df, min_seq_len=3, verbose=True):
    r"""Turn a raw log into one chronological row per attempt per learner.

    ASSISTments 2009 (skill_builder_data.csv) has two traps that leak labels:
      * the file is grouped by skill, not by time, so ``groupby('user_id')`` in file order
        puts later attempts into the history of earlier ones;
      * a problem tagged with k skills is stored as k rows sharing one ``order_id`` (same
        attempt, same label), so the next-step target is often a copy of the previous step.
    When ``order_id`` exists, rows are sorted by it inside each learner (learners keep their
    first-appearance order) and only one row per ``(user_id, order_id)`` is kept: the one with
    the smallest ``skill_id``. Logs without ``order_id`` are left in file order.
    Shared by Level 1 (``load_dataset``) and Level 2 (``level2.grkt_data.build_grkt_data``).
    """
    stats = {"rows_raw": int(len(df))}
    df = df.dropna(subset=["skill_id"])
    stats["rows_with_skill"] = int(len(df))
    stats["has_order_id"] = "order_id" in df.columns
    stats["users_out_of_order"] = 0
    stats["duplicate_rows_removed"] = 0
    if stats["has_order_id"]:
        backwards = df.groupby("user_id", sort=False)["order_id"].diff() < 0
        stats["users_out_of_order"] = int(backwards.groupby(df["user_id"]).any().sum())
        user_rank, _ = pd.factorize(df["user_id"])
        df = (
            df.assign(_user_rank=user_rank)
            .sort_values(["_user_rank", "order_id", "skill_id"], kind="mergesort")
            .drop(columns="_user_rank")
        )
        dup = df.duplicated(subset=["user_id", "order_id"], keep="first")
        stats["duplicate_rows_removed"] = int(dup.sum())
        df = df[~dup]
    df = df.groupby("user_id", sort=False).filter(lambda q: len(q) >= min_seq_len).copy()
    stats["rows_kept"] = int(len(df))
    stats["users_kept"] = int(df["user_id"].nunique())
    if verbose:
        print(
            "prepare_interactions: rows {rows_raw} -> {rows_kept}, users kept {users_kept}, "
            "re-sorted users {users_out_of_order}, rows sharing (user_id, order_id) removed "
            "{duplicate_rows_removed} (exact duplicates + extra skills of multi-skill items)".format(**stats)
        )
    return df, stats


def sequence_order_check(dataset):
    r"""Count leakage symptoms in the sequences actually fed to the models.

    ``dataset`` is the ``KTDataset`` returned inside the loaders (``loader.dataset.dataset``).
    Both counters must be 0: a step whose ``order_id`` is not larger than every earlier step
    means future attempts sit in the history; an equal ``order_id`` means a duplicated attempt.
    """
    orders = getattr(dataset, "order_ids", None) or []
    out = {"users": len(orders), "targets": 0, "non_increasing": 0, "duplicate_attempts": 0}
    for seq in orders:
        for prev, cur in zip(seq[:-1], seq[1:]):
            out["targets"] += 1
            out["non_increasing"] += int(cur <= prev)
            out["duplicate_attempts"] += int(cur == prev)
    out["checked"] = bool(orders) and any(len(s) for s in orders)
    return out


def _skill_lookup(df):
    lookup = {}
    if "skill_name" not in df.columns:
        return lookup
    for _, row in (
        df[["skill", "skill_id", "skill_name"]].drop_duplicates("skill").iterrows()
    ):
        lookup[int(row["skill"])] = str(row["skill_name"])
    return lookup


def build_transition_graph(
    question_list, seq_len_list, indices, student_num, concept_num
):
    graph = np.zeros((concept_num, concept_num))
    student_dict = dict(zip(indices, np.arange(student_num)))
    for i in range(student_num):
        if i not in student_dict:
            continue
        questions = question_list[i]
        seq_len = seq_len_list[i]
        for j in range(seq_len - 1):
            pre = questions[j]
            next = questions[j + 1]
            graph[pre, next] += 1
    np.fill_diagonal(graph, 0)
    # row normalization
    rowsum = np.array(graph.sum(1))

    def inv(x):
        if x == 0:
            return x
        return 1.0 / x

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph
