import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from kt_common.ids import as_id
from kt_common.interactions import (
    DATA_VERSION,
    interaction_stats,
    iter_users,
    load_interactions,
    skill_lookup,
    truncate_len,
)
from .utils import (
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
    # Step 1-2 - drop rows without skill, drop short learners, enumerate skill id.
    # Một nguồn duy nhất, dùng chung với level2/grkt_data.build_grkt_data.
    df = load_interactions(file_path, min_seq_len=min_seq_len)

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
    order_id_list = []
    has_order = "order_id" in df.columns

    for user_id, series in iter_users(df):
        take = truncate_len(len(series), max_seq_len)
        feature_list.append(series["skill_with_answer"].tolist()[:take])
        question_list.append(series["skill"].tolist()[:take])
        answer_list.append(series["correct"].eq(1).astype("int").tolist()[:take])
        seq_len_list.append(take)
        user_id_list.append(as_id(user_id))
        order_id_list.append(series["order_id"].tolist()[:take] if has_order else [])
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
    kt_dataset.skill_lookup = skill_lookup(df)
    kt_dataset.data_version = DATA_VERSION
    # order_id của ĐÚNG các bước đưa vào mô hình -> kiểm thứ tự/lượt trùng trên dữ liệu thật.
    kt_dataset.order_ids = order_id_list
    kt_dataset.data_stats = interaction_stats(df, max_seq_len)
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
    # Chuẩn hoá theo hàng: chia trực tiếp thay cho diag(1/rowsum) @ graph (nhân ma trận K×K
    # thừa). Hàng toàn 0 (khái niệm không có chuyển tiếp) giữ nguyên 0.
    # Nhân với nghịch đảo (không chia) để trùng từng bit với bản cũ.
    rowsum = graph.sum(axis=1, keepdims=True)
    r_inv = np.zeros_like(rowsum)
    np.divide(1.0, rowsum, out=r_inv, where=rowsum != 0)
    graph = graph * r_inv
    # covert to tensor
    graph = torch.from_numpy(graph).float()
    return graph


def build_dkt_graph(file_path, concept_num):
    graph = np.loadtxt(file_path)
    assert graph.shape[0] == concept_num and graph.shape[1] == concept_num
    graph = torch.from_numpy(graph).float()
    return graph
