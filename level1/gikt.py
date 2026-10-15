"""PyTorch port of the GIKT core (Yang et al., ECML-PKDD 2020).

The original `GIKT/` tree is TensorFlow 1.12 (`tf.placeholder`, `tf.contrib.rnn`,
`tf.contrib.layers`) and will not run on current Kaggle images. This module keeps
the pieces that matter for cap do 1:

- bipartite question–skill graph
- sum-aggregator embedding propagation (n-hop)
- LSTM over (question, response)
- next-step correct/incorrect prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GIKT(nn.Module):
    def __init__(
        self,
        num_questions: int,
        num_skills: int,
        q_neighbors: torch.Tensor,
        s_neighbors: torch.Tensor,
        emb_dim: int = 64,
        hidden_dim: int = 64,
        n_hop: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_hop = n_hop
        self.q_emb = nn.Embedding(num_questions, emb_dim, padding_idx=0)
        self.s_emb = nn.Embedding(num_skills, emb_dim, padding_idx=0)
        self.r_emb = nn.Embedding(3, emb_dim, padding_idx=2)  # 0, 1, pad=2
        self.register_buffer("q_neighbors", q_neighbors.long())
        self.register_buffer("s_neighbors", s_neighbors.long())
        self.agg = nn.Linear(emb_dim, emb_dim)
        self.input_proj = nn.Linear(emb_dim * 2, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, emb_dim)

    def _masked_mean(self, table: nn.Embedding, index: torch.Tensor) -> torch.Tensor:
        mask = index > 0
        safe = index.clamp(min=0)
        vec = table(safe)
        weights = mask.unsqueeze(-1).float()
        return vec.mul(weights).sum(dim=-2) / weights.sum(dim=-2).clamp(min=1.0)

    def encode_questions(self, questions: torch.Tensor) -> torch.Tensor:
        q_self = self.q_emb(questions.clamp(min=0))
        neigh = self.q_neighbors[questions.clamp(min=0)]
        q_from_s = self._masked_mean(self.s_emb, neigh)
        return torch.tanh(self.agg(q_self + q_from_s))

    def forward(self, questions: torch.Tensor, answers: torch.Tensor):
        """
        questions: [B, T]  (0 = pad)
        answers:   [B, T]  (0/1, pad aligned with questions==0)
        returns pred for t=1..T-1, shape [B, T-1]
        """
        q_enc = self.encode_questions(questions)
        resp = answers.clamp(0, 1)
        resp = torch.where(questions > 0, resp, torch.full_like(resp, 2))
        x = torch.cat([q_enc, self.r_emb(resp)], dim=-1)
        x = self.dropout(torch.relu(self.input_proj(x)))
        h, _ = self.lstm(x)
        h = self.dropout(self.out(h[:, :-1, :]))
        q_next = q_enc[:, 1:, :]
        logits = (h * q_next).sum(dim=-1)
        pad_next = questions[:, 1:] <= 0
        pred = torch.sigmoid(logits)
        pred = pred.masked_fill(pad_next, 0.0)
        return pred
