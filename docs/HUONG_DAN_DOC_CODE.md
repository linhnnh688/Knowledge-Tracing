# Hướng dẫn đọc code — từng hàm làm gì, ứng với chỗ nào trong bài báo

> Mục tiêu: khi giảng viên chỉ vào một hàm/dòng bất kỳ, bạn trả lời được **nó làm gì, nhận gì,
> trả gì, ứng với phương trình nào của bài báo, và khác bài báo ở đâu**.
>
> Hai bài báo nền:
> - **[GKT]** Nakagawa, Iwasawa, Matsuo — *Graph-based Knowledge Tracing: Modeling Student
>   Proficiency Using Graph Neural Network* (bản PDF ghi ICLR 2019; bản chính thức IEEE/WIC/ACM WI 2019).
>   Mã: `GKT/` (port PyTorch của jhljx).
> - **[GRKT]** Cui, Qian, Jiang, Zhang — *Leveraging Pedagogical Theories to Understand Student
>   Learning Process with Graph-based Reasonable Knowledge Tracing*, KDD 2024. Mã: `GRKT/`
>   (của tác giả, JJCui96/GRKT).
>
> Đọc kèm [BAO_CAO_KIEN_TRUC.md](BAO_CAO_KIEN_TRUC.md) (kiến trúc tổng thể, khác biệt so với
> repo gốc) và [ON_TAP_BAO_VE.md](ON_TAP_BAO_VE.md) (một người học thật đi qua từng hàm, kết
> quả, rò rỉ, điểm yếu). Tài liệu này đi **xuống mức hàm**. Mọi số dòng tính trên branch
> `fix/assist2009-leak-batchnorm`, commit `868838c`.

---

## Mục lục

1. [Một batch đi qua hệ thống như thế nào (trace đầu–cuối)](#1-một-batch-đi-qua-hệ-thống-như-thế-nào)
2. [Bảng đối chiếu: phương trình bài báo ↔ dòng code](#2-bảng-đối-chiếu-phương-trình-bài-báo--dòng-code)
3. [Code khác bài báo ở đâu (phải nắm để không bị bắt bẻ)](#3-code-khác-bài-báo-ở-đâu)
4. [`GKT/` — dữ liệu, đồ thị, mô hình GKT/DKT](#4-gkt--dữ-liệu-đồ-thị-mô-hình-gktdkt)
5. [`level1/` — huấn luyện, đánh giá, giải thích, dashboard](#5-level1--huấn-luyện-đánh-giá-giải-thích-dashboard)
6. [`GRKT/` — mô hình GRKT](#6-grkt--mô-hình-grkt)
7. [`level2/` — dữ liệu GRKT, đồ thị quan hệ, huấn luyện, quỹ đạo](#7-level2--dữ-liệu-grkt-đồ-thị-quan-hệ-huấn-luyện-quỹ-đạo)
8. [Notebook: mỗi ô gọi hàm nào](#8-notebook-mỗi-ô-gọi-hàm-nào)
9. [Những dòng "khó" hay bị chỉ vào](#9-những-dòng-khó-hay-bị-chỉ-vào)

---

## 1. Một batch đi qua hệ thống như thế nào

Ký hiệu: `B` = số người học trong batch (32), `T` = độ dài chuỗi dài nhất trong batch (≤ 200),
`C` = số khái niệm (112 với ASSIST2009 sau làm sạch), `H` = hidden 32, `E` = embedding 32.

```
skill_builder_data.csv
  │ prepare_interactions()        bỏ skill rỗng, sắp order_id, gộp lượt trùng, lọc < 3 lượt
  ▼
DataFrame (282.619 dòng)
  │ load_dataset()                factorize skill → 0..C-1; x = 2·skill + correct;
  │                               gom theo người học; cắt 200 lượt đầu
  ▼
KTDataset (3.852 chuỗi) ──random_split(seed 42)──► train / val / test (2.311 / 770 / 771)
  │                                   │
  │ build_transition_graph(train)     │ DataLoader + pad_collate (padding = -1)
  ▼                                   ▼
A [C×C]                     features [B,T]  questions [B,T]  answers [B,T]
  │                                   │
  └──────────────► GKT.forward() ◄────┘   vòng for i in range(T):
                         │                  _aggregate → _agg_neighbors → _update → _predict
                         ▼                  → _get_next_pred(q_{i+1})
                   pred [B, T-1]   (xác suất đúng cho bước 1..T-1)
                         │
                         ▼ KTLoss(pred, answers)  ← so với answers[:, 1:]
                   loss → backward → Adam
```

Level 2 dùng lại đúng `KTDataset` đó: `build_grkt_data()` dựng `UserSequence` (1-based,
có `problem_id`, thời gian) và **assert từng người học khớp Level 1**, rồi GRKT chạy với
`feed_dict = {probs, knows, corrs, times}` dạng `[B, S]` / `[B, S, K]`.

---

## 2. Bảng đối chiếu: phương trình bài báo ↔ dòng code

### 2.1. GKT ([GKT] §2.1–2.2) — `GKT/models.py`

| Bài báo | Ý nghĩa | Code |
|---|---|---|
| Aggregate: $h'^t_k=[h^t_k, x^tE_x]$ nếu $k=i$, $[h^t_k, E_c(k)]$ nếu $k\neq i$ | ghép trạng thái + embedding (khái niệm vừa làm dùng embedding "khái niệm + đúng/sai") | `GKT._aggregate` (dòng 158–197): `res_embedding = one_hot(x_t) @ emb_x`, `concept_embedding = emb_c(...)`, `index_put` đặt `res_embedding` vào vị trí $q_t$, `cat(ht, concept_embedding)` |
| Eq.1: $m^{t+1}_k = f_{self}(h'_k)$ ($k=i$), $f_{neighbor}(h'_i,h'_k)$ ($k\neq i$) | thông điệp | `GKT._agg_neighbors` (200–299) |
| Eq.2: $f_{neighbor}=A_{ij}f_{outgo}([h'_i,h'_j]) + A_{ji}f_{income}([h'_i,h'_j])$ | đồ thị thống kê, 2 chiều cạnh | dòng `adj = self.graph[masked_qt, :]`, `reverse_adj = self.graph[:, masked_qt]`, `neigh_features = adj*f_neighbor_list[0](...) + reverse_adj*f_neighbor_list[1](...)` |
| MHA: $\frac1K\sum_k \alpha^k_{ij} f_k(h'_i,h'_j)$ | đồ thị học bằng attention | nhánh `else` trong `_agg_neighbors` + `MultiHeadAttention.forward` + `neigh_features = 1/edge_type_num * ...` |
| $\tilde m = \mathcal G_{ea}(m)$ | erase–add gate | `EraseAddGate` (`layers.py` 61–104), gọi trong `GKT._update` (300–338) |
| $h^{t+1}_k = \mathcal G_{gru}(\tilde m_k, h^t_k)$ | GRU | `self.gru = nn.GRUCell(H, H)`, gọi trong `_update` |
| Predict: $y^t_k=\sigma(W_{out}h^{t+1}_k + b_k)$ | xác suất đúng mỗi khái niệm | `GKT._predict` (340–355): `self.predict = nn.Linear(H, 1)` + sigmoid |
| chọn khái niệm kế tiếp | lấy $y_{q_{t+1}}$ | `GKT._get_next_pred` (357–380): nhân với one-hot của $q_{t+1}$ rồi `sum` |
| NLL | mất mát | `KTLoss.forward` (`metrics.py`) |
| Dense graph $A_{ij}=1/(|V|-1)$ | | `build_dense_graph` (`utils.py`) |
| Transition graph $A_{ij}=n_{ij}/\sum_k n_{ik}$ | | `build_transition_graph` (`processing.py` 288–319) |

### 2.2. GRKT ([GRKT] §4) — `GRKT/model/GRKT.py`

| Bài báo | Ý nghĩa | Code (`GRKT.forward` trừ khi ghi khác) |
|---|---|---|
| Eq.3: $\beta^{\mathcal G}_{c_i c_j}=\sigma(k_{c_i}^\top W^{\mathcal G}_{cor} k_{c_j})$ | độ tương quan giữa hai khái niệm | `beta_matrix = req_matrix(E_K) @ E_Kᵀ` rồi `.sigmoid()` — **dùng chung một ma trận cho cả 3 đồ thị** (xem §3) |
| Eq.1–2: $\tilde f^{(l)}=\frac{1}{|\mathcal G(c)|}\sum \beta \tilde f^{(l-1)}W$, ReLU, $O$ | GNN nguyên mẫu (mean có trọng số β) | `beta_rel_tilde = beta * rel_map / rel_map.sum(-1)` (chia số láng giềng = mean); `X_matrix_*[k]` = $W$, `X_output_*[k]` = $O$ |
| Eq.4: cộng qua $\mathcal P,\mathcal S,\mathcal R$, residual khi cùng chiều | gộp 3 đồ thị | các dòng `h_tilde = h_tilde + h_tilde_3_rel/_pre/_sub`; `if k == 0: lk_tilde = ...` (không residual vì đổi chiều $d_e\to d_k$) |
| $L$ = số tầng GNN | số hop | `args.k_hop` (vòng `for k in range(self.args.k_hop)`) |
| Eq.7: $\hat m_{c,t} = h_{c,t}\cdot w_h$, $w_h\ge0$ | mastery | `self.know_master_proj = Positive_Linear(k_hidden, 1, "softmax")` |
| Eq.9: $\alpha_{q,c}=\sigma(e_q^\top W_{req}k_c)$ | câu hỏi cần khái niệm nào | `alpha_matrix = req_matrix(prob_embedding) @ E_Kᵀ` → `.sigmoid()` |
| Eq.8,10: retrieval GNN, không ReLU, $W_{rtv}\ge0$ | Stage I: truy xuất | khối `for k in range(k_hop): h_tilde_1_* = agg_*_matrix[k](h_tilde)` (`Positive_Linear` softmax), `* alpha_1`, `beta_*_tilde.matmul(...)` |
| Eq.11: $\hat a_t=\sigma(\tilde h\, w_h - d_{q_t})$ | dự đoán | `master = know_master_proj(h_tilde).gather(know)` → trung bình các KC → `score = (master - diff).sigmoid()` |
| Eq.12: $d_q=\text{MLP}_{diff}([k_c\oplus e_q])$ | độ khó câu hỏi | `self.prob_diff_mlp` (Linear–ReLU–Linear) trên `know_prob_emb` |
| Eq.13: $g=\text{MLP}_{gain}([h_{c,t^-}, \hat e_q])$ | Stage II: lượng tăng | `gain = self.gain_ffn(cat([know_prob_emb, target_h]))` |
| Eq.14: $\tilde G=\text{ReLU}(\text{GNN}_{gain}(\tilde G^{(0)}))$ | lan truyền tăng | `total_gain = scatter(...)`, vòng `gain_matrix_*`/`gain_output_*`, `total_gain.relu()` |
| Eq.15: $H_t = H_{t^-} + a_tG - (1-a_t)L$ | cập nhật theo đúng/sai | `h = h + corr_1 * total_gain - (~corr_1 * total_loss)` |
| Eq.16: $\pi=\text{softmax}(\text{MLP}_{dcs}(\cdot))$, argmax quyết định có học | Stage III: quyết định học | `decision = self.decision_mlp(learn_input)`; `F.gumbel_softmax(decision, tau, hard=True)` |
| Eq.17–18: $p=\text{MLP}_{prg}$, GNN$_{prg}$ | lượng học | `learn = self.learn_mlp(...)`, vòng `learn_matrix_*`, `total_learn.relu()` |
| Eq.19–20: $h \mathrel{+}= p\odot(1-e^{-(n+1)\Delta T\gamma})$ | đường cong học | `learn_exp = exp(-(learn_count+1) * delta_time * learn_kernel_para)`; `h = h + (1 - learn_exp) * total_learn` |
| Eq.21: $\gamma=\text{softplus}(\text{GNN}_{lrn}(k))$ | nhân học riêng mỗi KC | khối `lk_tilde` trước vòng thời gian → `learn_kernel_para = softplus(lk_tilde) * alpha` |
| Eq.22–23: $h \mathrel{-}= \Delta h\odot(1-e^{-(n+1)\Delta T\theta})$, $\Delta h = h_t-h_0$ | đường cong quên | `history_gain = (h - h_initial).clamp(0)`; `forget_exp = exp(...)`; `h = h - (history_gain - gain_after_forget) * (total_learn == 0)` |
| Eq.24: $\theta=\text{softplus}(\text{GNN}_{fgt}(k))$ | nhân quên | khối `fk_tilde` → `forget_kernel_para` |
| Eq.25: BCE | mất mát | `BaseModel.loss` (`F.binary_cross_entropy`) |
| App. B.1: $sim$, $pre$, ngưỡng $\eta$ | dựng đồ thị KC | `level2/grkt_data.build_relevance_maps` + `graph_variant(thresh=0.6)` |

---

## 3. Code khác bài báo ở đâu

Phải nắm rõ — giảng viên đọc bài báo rồi hỏi "sao code không giống?".

### 3.1. GKT

| Bài báo [GKT] | Code repo | Ghi chú khi trả lời |
|---|---|---|
| Hình 1 nói "chỉ cập nhật khái niệm liên quan" | erase–add + GRU chạy trên **mọi** khái niệm; khái niệm không có cạnh nhận $m_j=0$ nhưng gate vẫn cộng $w_j\tanh(b)$ và GRU vẫn đổi $h_j$ | đúng theo phương trình của chính bài báo (Eq.1 áp dụng cho mọi $k$); "chỉ liên quan" chỉ đúng gần đúng. Giải thích vì sao Zero graph vẫn hoạt động. |
| $b_k$: bias **riêng mỗi khái niệm** | `nn.Linear(H, 1)` — **một bias chung** | khác biệt của bản port jhljx, không phải của nhóm |
| Appendix A: "Combine simultaneous answer logs into one" | trước branch này **không** làm; nay `prepare_interactions` gộp lượt trùng `order_id` | bản sửa của nhóm **khớp lại** quy trình của bài báo |
| Appendix A: chỉ giữ skill được làm ≥ 10 lần; 1.000 người học | giữ mọi skill, 3.852 người học | dữ liệu lớn hơn, nhiều skill hiếm |
| Appendix B: split 8:1:1, dropout 0.5, batch 16, lr 0.01, BN ở output | 6:2:2, dropout 0, batch 32, lr 1e-3, BN trong MLP | 6:2:2 theo đề bài; lr nhỏ hơn để ổn định |
| Bảng 1 (ASSIST): DKT 0,709; GKT Transition 0,721 | DKT 0,761; GKT Transition 0,736 | khác tiền xử lý/tập con nên không so trực tiếp được; ở repo DKT > GKT |

### 3.2. GRKT

| Bài báo [GRKT] | Code (của tác giả, repo giữ nguyên) | Ghi chú |
|---|---|---|
| **3 giai đoạn**: retrieval (gồm dự đoán), strengthening, learning/forgetting | cùng cấu trúc | trong BAO_CAO_KIEN_TRUC tách "dự đoán" thành pha riêng chỉ để dễ đọc |
| Eq.3: $W^{\mathcal G}_{cor}$ **riêng cho mỗi đồ thị** | một ma trận `req_matrix` dùng chung cho cả $\alpha$ (Eq.9) và $\beta$ (Eq.3), và một $\beta$ chung cho 3 đồ thị | khác biệt nằm ở mã tác giả |
| Eq.16: argmax của softmax | `F.gumbel_softmax(hard=True)` — **lấy mẫu ngẫu nhiên, kể cả lúc eval** | nên kết quả eval có nhiễu nhỏ; repo lưu/khôi phục RNG |
| Eq.22: $\Delta h = h_t - h_0$ | `(h - h_initial).clamp(0)` — chỉ quên phần tăng | |
| Eq.20–23 không có hệ số $\alpha$ | nhân thêm `args.alpha = 0.01` | tham số "time interval factor" trong code tác giả |
| Bảng 6: $L=2$, lr 5e-3, $d_k=16$, $\eta=0.6$, l2 1e-6 | repo chạy `k_hop=1`, lr 1e-3, $d_k=16$, $\eta=0.6$, l2 0 | **$L=1$ là lựa chọn của nhóm** (tiết kiệm VRAM/thời gian trên Kaggle) — nói rõ khi bị hỏi |
| §5.1.1: ASSIST09 bản *combined* (đa KC, 1,22 KC/câu), cắt **cửa sổ 100**, 5-fold | `skill_builder_data.csv`, 1 KC/lượt, mỗi người học một chuỗi (≤200), split theo người học | giao thức của bài báo để cùng người học ở train và test (vi phạm §6.4 của đề) |
| §5.1.2: CONS, GAUCM, RPT | repo đo precedence, stability, forget_slope (+ 3 kiểm chứng cài đặt) | CONS của bài báo = 1,0000 ở mọi tập — chính là loại "đúng theo cấu trúc" mà repo xếp vào "kiểm chứng cài đặt" |
| App. B.1: "dùng $pre_{c_i,c_j}$ để xấp xỉ xác suất **$c_i$ là tiên quyết của $c_j$**" | công thức đếm cặp ($c_i$ **sai**, $c_j$ **đúng**) | **lời và công thức của bài báo ngược nhau**: chính lập luận của bài báo ("nếu $c_i$ tiên quyết thì $c_i$ đúng, $c_j$ sai hay gặp hơn") cho thấy công thức đo điều ngược lại. Kiểm chứng trên dữ liệu: ở 86% cạnh, khái niệm cột dễ hơn hàng ⇒ `pre_map[x, y] = 1` nghĩa là **y là tiên quyết của x**. Tiêu chí precedence trong repo đã sửa theo chiều này (commit `10812c0`). Mô hình GRKT không bị ảnh hưởng vì nó dùng cả `pre` lẫn `sub = preᵀ`. |

---

## 4. `GKT/` — dữ liệu, đồ thị, mô hình GKT/DKT

### 4.1. `GKT/processing.py`

| Hàm | Làm gì | Vào → Ra | Câu hỏi có thể gặp |
|---|---|---|---|
| `KTDataset` | Dataset PyTorch, mỗi phần tử = **một người học** `(features, questions, answers)` (list số nguyên). Gắn thêm `user_ids`, `skill_lookup`, `order_ids`, `prep_stats`. | | *Vì sao một phần tử là một người?* → để `random_split` chia theo người học, không người nào bị chia đôi. |
| `pad_collate(batch)` | Gộp các chuỗi dài khác nhau thành tensor `[B, T]`, đệm **-1**. | list → 3 LongTensor `[B,T]` | *-1 dùng để làm gì?* → đánh dấu padding; mọi chỗ sau đều lọc bằng `!= -1`. |
| `prepare_interactions(df, min_seq_len)` (216–256) | Làm sạch log: bỏ dòng thiếu skill → đếm người có `order_id` đi lùi (vector hoá: `groupby().diff() < 0`) → **sắp theo `order_id` trong từng người học** → **giữ 1 dòng mỗi `(user_id, order_id)`** (skill_id nhỏ nhất) → bỏ người < 3 lượt. Trả kèm thống kê. | DataFrame → (DataFrame, dict) | *Vì sao cần?* → file xếp theo skill, câu nhiều skill bị nhân dòng → rò rỉ 70,6% / 8,9%. Đây cũng là bước "combine simultaneous logs" của bài báo GKT. 176.103 dòng bị bỏ = 121.207 dòng lặp y hệt + 54.896 kỹ năng phụ. |
| `PREP_VERSION` (213) | hằng `"order-dedup-v1"`; tăng mỗi khi `prepare_interactions` đổi chuỗi. Được ghi vào `run_config.json`, fingerprint resume Cấp 1, và (từ `868838c`) fingerprint checkpoint GRKT + baseline Cấp 2. | | *Để làm gì?* → không bao giờ nạp nhầm checkpoint train trên tiền xử lý cũ (bị rò rỉ). |
| `sequence_order_check(dataset)` | Đếm trên chuỗi thật: số bước `order_id` không tăng và số lượt trùng. Phải = 0. | KTDataset → dict | dùng trong leak report. |
| `load_dataset(...)` | Hàm trung tâm của Level 1: đọc CSV → `prepare_interactions` → `factorize` skill → `x = 2·skill + correct` → gom theo người, cắt `max_seq_len` → `random_split(seed)` → DataLoader → dựng đồ thị theo `graph_type`. | đường dẫn + cấu hình → `(concept_num, graph, train/val/test loader)` | *Đồ thị dựng từ đâu?* → `train_dataset.indices`, tức chỉ người học train. |
| `build_transition_graph(question_list, seq_len_list, indices, ...)` | Đếm $n(i\to j)$ trên các cặp liên tiếp của người học **train**, bỏ đường chéo, chuẩn hoá hàng. | → Tensor `[C, C]` | `inv` dùng để tránh chia 0 cho hàng rỗng. |
| `_skill_lookup(df)` | Ánh xạ chỉ số khái niệm → tên skill (để in case study). | | |
| `build_dkt_graph` | Đọc đồ thị DKT từ file (không dùng trong thí nghiệm). | | |

### 4.2. `GKT/utils.py`

| Hàm | Làm gì |
|---|---|
| `build_dense_graph(n)` | $1/(n-1)$ ngoài đường chéo. |
| `build_identity_graph(n)` | ma trận đơn vị (ablation). |
| `build_zero_graph(n)` | ma trận 0 (ablation bỏ cạnh). |
| `build_shuffled_graph(A, seed)` | giữ vị trí hàng + giá trị khác 0 của Transition, **hoán vị toàn cục chỉ số cột**, chuẩn hoá lại hàng. Trùng vị trí bị gộp nên số cạnh giảm (2.665 → 2.146). |
| `gumbel_softmax`, `kl_*`, `nll_gaussian`, `encode_onehot` | phục vụ biến thể VAE của bản gốc — **không dùng**. |
| `accuracy` | độ chính xác từ xác suất 2 lớp (dùng trong `KTLoss`). |

### 4.3. `GKT/layers.py`

| Lớp/hàm | Làm gì | Ghi chú |
|---|---|---|
| `MLP` | Linear → ReLU → (dropout) → Linear → ReLU → **BatchNorm1d**. Dùng cho $f_{self}, f_{outgo}, f_{income}$. | `init_weights`: Xavier normal, bias 0,1. |
| `MLP.batch_norm` | Áp BN; với tensor 3 chiều thì gộp 2 chiều đầu rồi tách lại. **Chỉ bỏ BN khi đang train mà chỉ có 1 hàng.** | bản gốc bỏ cả lúc eval → dự đoán phụ thuộc batch (đã sửa, commit `09d9431`). |
| `EraseAddGate` | $e=\sigma(W_e x)$, $a=\tanh(W_a x)$, $x' = x - w_c\, e\odot x + w_c\, a$; `weight` dài C = một hệ số cho mỗi khái niệm. | "erase–add" từ DKVMN nhưng dùng chính $x$ làm tín hiệu (không có vector ghi riêng). |
| `ScaledDotProductAttention` | $QK^\top/\sqrt{d_k}$, che chính khái niệm vừa làm (`-1e9`), **softmax theo khái niệm** (`dim=-1`), `nan_to_num`, dropout. | bản gốc softmax theo head (`dim=0`) — nhóm đổi có chủ đích. |
| `MLPEncoder`, `MLPDecoder`, `VAE` | biến thể VAE — không dùng. | |

### 4.4. `GKT/models.py`

| Hàm | Làm gì | Shape |
|---|---|---|
| `GKT.__init__` | Tạo `graph` (Parameter đóng băng `requires_grad=False`), `emb_x` (2C × E), `emb_c` (C+1 × E, hàng cuối là padding), `f_self`, `f_neighbor_list` (2 MLP), `erase_add_gate`, `gru`, `predict`. | |
| `GKT._one_hot_tables(device)` | Tạo & cache bảng one-hot theo thiết bị của input (bản gốc tạo một lần → lệch thiết bị khi nạp checkpoint). | `[2C,2C]`, `[C+1,C]` |
| `GKT._aggregate(xt, qt, ht, B)` | Eq. Aggregate. Khái niệm vừa làm lấy embedding "khái niệm + đúng/sai", khái niệm khác lấy embedding khái niệm. | → `tmp_ht [B, C, H+E]` |
| `GKT._agg_neighbors(tmp_ht, qt)` | Eq.1–2. Tính $f_{self}$ cho khái niệm vừa làm, $f_{neighbor}$ cho mọi khái niệm khác theo hàng/cột của $A$ (hoặc attention), rồi **ghi đè vị trí tự thân** bằng $f_{self}$. | → `m_next [B, C, H]` |
| `GKT._update(tmp_ht, ht, qt)` | Gọi `_agg_neighbors` → erase–add gate → GRUCell trên `B·C` vector. Chỉ người học có `qt != -1` được cập nhật. | → `h_next [B, C, H]` |
| `GKT._predict(h_next, qt)` | `Linear(H,1)` cho mọi khái niệm, **sigmoid chỉ ở hàng có `qt != -1`**. | → `[B, C]` |
| `GKT._get_next_pred(yt, q_next)` | Lấy cột $q_{t+1}$ (padding → hàng 0 của bảng one-hot). | → `[B]` |
| `GKT._get_edges` | dựng cạnh cho VAE — không dùng. | |
| `GKT.forward(features, questions, return_state)` | Vòng **`for i in range(T)`**: aggregate → update → ghi `ht` → predict → lấy dự đoán cho $q_{i+1}$. `return_state=True` trả thêm $H_t$ mọi bước (dashboard, xAI). | → `pred [B, T-1]` |
| `MultiHeadAttention.forward` | query = embedding khái niệm vừa làm, key = embedding mọi khái niệm; trả `graphs [n_head, C, C]` với hàng $q_t$ là trọng số attention. `self.graphs` lưu bản detach để trực quan. | |
| `DKT.__init__` | `LSTM(2C, H)` + `Linear(H, C)`. | |
| `DKT.forward` | one-hot $x_t$ → `pack_padded_sequence` (độ dài thật) → LSTM → sigmoid → bỏ bước cuối → `_get_next_pred`. | → `[B, T-1]` |
| `DKT._get_next_pred` | như GKT nhưng vector hoá trên cả chuỗi. | |

### 4.5. `GKT/metrics.py`

| Hàm | Làm gì |
|---|---|
| `KTLoss.forward(pred, real)` | So `pred [B,T-1]` với `real[:, 1:]`; kẹp $p\in[10^{-6},1-10^{-6}]$; tính AUC/ACC theo batch (chỉ để log); **NLL tự chia cho số vị trí hợp lệ** (batch toàn padding → loss 0 thay vì NaN). |

---

## 5. `level1/` — huấn luyện, đánh giá, giải thích, dashboard

### 5.1. `level1/paths.py`

| Hàm | Làm gì |
|---|---|
| `repo_root()` | thư mục gốc repo (cha của `level1/`). |
| `ensure_gkt_on_path()` | thêm `GKT/` vào `sys.path` để `from processing import ...` chạy được. |
| `materialize_lfs(path)` | Nếu file là **con trỏ Git LFS** (Kaggle không có git-lfs), tải nội dung thật từ GitHub theo commit đang chạy, **kiểm tra sha256 + kích thước**, ghi đè. File thường trả nguyên. Repo/commit lấy từ `git remote origin` + `git rev-parse HEAD`, hoặc biến môi trường `LFS_REPO` / `LFS_REF`; không có git hoặc file nằm ngoài repo thì báo lỗi nói rõ là do LFS. |
| `read_json(path)` | `materialize_lfs` rồi `json.loads` — vì `*.json` cũng nằm trong LFS. Dùng cho các file `.json` đi kèm checkpoint baseline và `.meta.json` của Junyi. |

### 5.2. `level1/pipeline.py`

| Hàm | Làm gì | Câu hỏi có thể gặp |
|---|---|---|
| `set_seed(seed)` | seed `random`, NumPy, Torch, CUDA; bật cuDNN deterministic. | *Sao vẫn không tất định tuyệt đối?* → một số phép `index_put`/scatter trên GPU vẫn không tất định. |
| `resolve_data_file(name)` | tìm CSV (đường dẫn trực tiếp hoặc `GKT/data/`) + `materialize_lfs`. | |
| `load_kt_data(...)` | bọc `load_dataset`, trả `bundle = {concept_num, graph, train/val/test_loader, data_path}`. | *Bundle là gì?* → mọi thứ một mô hình cần, dùng chung giữa các mô hình để cùng split. |
| `order_leak_lines(bundle)` | dòng báo cáo "làm sạch log" + "thứ tự trong chuỗi" cho leak report cả 2 cấp. | |
| `save_split(bundle, path)` | ghi `split.json` = danh sách `user_id` của train/val/test. | |
| `build_model(type, C, graph, ...)` | DKT: `DKT(2C, 32, C)`; GKT: tạo `MultiHeadAttention(2, C, 32, 32)` nếu `MHA`, rồi `GKT(...)`. | |
| `_forward(model, ...)` | đưa batch lên thiết bị, gọi model, lấy phần tử đầu của tuple. | |
| `_epoch_loop(model, loader, ..., optimizer)` | Một epoch train (có optimizer) hoặc eval. Bỏ bước cập nhật nếu loss/gradient không hữu hạn; `clip_grad_norm_(10)`; **gom dự đoán toàn loader rồi tính AUC/ACC một lần**. | *AUC tính thế nào?* → toàn cục, không trung bình theo batch. |
| `train_one(...)` | Adam, tối đa `epochs`, **early stopping patience 5 theo val AUC**, nạp lại trọng số tốt nhất. | |
| `evaluate(...)` | `_epoch_loop` ở chế độ eval. | |
| `run_experiment(...)` | `set_seed` → (nạp bundle nếu chưa có) → `build_model` → `train_one` → `evaluate` test → lưu checkpoint → trả dict kết quả. | |
| `run_level1(...)` | chạy nhanh DKT + 4 biến thể GKT (dùng khi chạy local, notebook không gọi). | |
| `run_gikt_experiment` | đã bỏ (raise). | |

### 5.3. `level1/explain.py` (Nhiệm vụ 1: 5 đúng + 5 sai)

| Hàm | Làm gì |
|---|---|
| `graph_off(model)` | context manager: tạm đặt $A:=0$ (đồ thị tĩnh) hoặc hook trả 0 cho đầu ra attention (MHA) — để đo "đồ thị đóng góp bao nhiêu". |
| `skill_lookup_from_loader`, `user_ids_from_loader` | lấy tên skill / `user_id` theo vị trí trong loader **không shuffle**. |
| `_neighbors(graph, skill)` | top-5 láng giềng theo hàng của $A$. |
| `_dataset_rows(loader)` | các chuỗi gốc (features, questions, answers) theo đúng thứ tự loader. |
| `_flip(feats, qs, h)` | phản chứng: đổi lượt $h$ từ đúng↔sai ($x = 2s + r$ → $2s + 1 - r$). |
| `_edge_evidence(graph, skill, history)` | top-3 cạnh vào/ra của khái niệm đích + lần gần nhất người học làm khái niệm láng giềng đó. |
| `annotate_evidence(model, loader, buckets, ...)` | Với mỗi ca: **chạy lại** chuỗi riêng lẻ, kiểm tra $|p_{lại}-p_{log}|\le0{,}02$; nếu khớp thì lật từng lượt trong 8 lượt gần nhất (Δp), tắt đồ thị (Δp), ghi cạnh liên quan. |
| `_forward_pred(model, f, q, idx)` | chạy model trên một chuỗi, lấy dự đoán ở vị trí `idx`. |
| `collect_cases(...)` | chạy toàn test, chia đúng/sai, sắp theo độ tự tin $|p-0{,}5|$, chọn 5 mỗi loại (`_select`, có thể ép khác skill), gọi `annotate_evidence`. |
| `_select(rows, n, distinct_skills)` | chọn ca, ưu tiên mỗi ca một skill. |
| `format_cases`, `_evidence_lines`, `_explain_one` | in `cases.md`: lịch sử 8 lượt, tỉ lệ đúng cùng skill, láng giềng, các Δp, câu giải thích sinh từ số liệu. |
| `summarize_skill_evidence` | đếm số ca có lịch sử cùng skill. |

> Notebook Cấp 1 (cell §7, hàm `enrich_cases`) dùng phiên bản chọn ca **phân tầng theo 5 ngăn
> độ tự tin** và thêm **occlusion** (che một lượt). Occlusion có lỗi ở lượt ngay trước bước dự
> đoán (che về `-1` khiến `_predict` không sigmoid → trả logit → Δp > 1). Lỗi này **đang hiện
> trong kết quả**: cột `max_occ_abs_dp` của `results/level1_out/report_summary.md` có 2,21; 2,30.
> Sửa bằng cách xoá hẳn lượt đó; chỉ cần chạy lại phần case study.

### 5.4. `level1/dashboard.py`

| Hàm | Làm gì |
|---|---|
| `export_learner_view(model, loader, ...)` | chạy model (GKT: `return_state=True`) trên test, mỗi người học một bản ghi: từng bước `{t, skill, y_true, y_prob, y_pred, state_norm (‖h_c‖ mọi khái niệm)}` → `learner_view.json`. |
| `render_learner_view(views, i)` | in văn bản một người học (dự phòng khi không có widget). |
| `interactive_learner_view(views)` | dropdown `ipywidgets` chọn người học; không có thì in người đầu tiên. |

---

## 6. `GRKT/` — mô hình GRKT

### 6.1. `GRKT/model/GRKT.py`

| Thành phần | Làm gì |
|---|---|
| `positive_activate(mode, w)` | biến trọng số thành dương: sigmoid / softplus / relu / **softmax theo cột** / none. |
| `Positive_Linear(d_in, d_out, mode)` | $y = x\cdot\text{act}(W)$ — ma trận trọng số **không âm**. Với `softmax`: mỗi cột là trọng số dương tổng 1 → phép trộn lồi. Đây là ràng buộc $W\ge0$ của Eq.7, Eq.10. |
| `GRKT.parse_args` | tham số mặc định của tác giả (`d_hidden 128`, `k_hidden 16`, `k_hop 1`, `thresh 0.6`, `tau 0.2`, `alpha 0.01`). |
| `GRKT.__init__` | tạo `init_hidden [NK+1, 16]` (= $H_0$), `know_master_proj` (= $w_h$), embedding khái niệm/câu hỏi, `req_matrix`, các bộ ma trận `agg_*`, `gain_*`, `loss_*`, `learn_*`, `learn_kernel_*`, `forget_kernel_*` (mỗi bộ có bản `_rel`, `_pre`, `_sub`, mỗi bản là `ModuleList` dài `k_hop`), `prob_diff_mlp`, `gain_ffn`, `loss_ffn`, `decision_mlp`, `learn_mlp`; nhị phân hoá `rel_map`, `pre_map` (> `thresh`), `sub_map = pre_mapᵀ`. |
| `GRKT.forward(feed_dict)` | (1) $\alpha$, $\beta$, $\tilde\beta$ ba đồ thị; (2) tiền tính nhân học/quên (Eq.21, 24); (3) **vòng `for i in range(S)`**: retrieval → dự đoán → strengthening → learning/forgetting (dùng `knows[i+1]`, `times[i+1]`, **không dùng nhãn bước sau**). Ghi `feed_dict["scores"] = scores[:, 1:][filt]`. Ba hook `rec["m_pre"/"m_tilde"/"m_strong"]` do repo thêm. |

### 6.2. `GRKT/BaseModel.py`, `GRKT/utils.py`, `GRKT/metric.py`, `GRKT/Runner.py`

| Hàm | Làm gì | Repo có dùng? |
|---|---|---|
| `BaseModel.get_feed_dict(batch, mode)` | đổi list chuỗi → tensor `probs [B,S]`, `knows [B,S,K]`, `corrs`, `times`, `lags`; `filt = probs[:,1:] > 0`; `labels = corrs[:,1:][filt]`. | **có** (qua `grkt_api._to_device_feed`) |
| `BaseModel.loss` | BCE giữa `scores` và `labels`. | **có** |
| `BaseModel.get_optimizer` | Adam(lr, weight_decay=l2). | **có** |
| `BaseModel.train_an_epoch / eval_an_epoch / fit / test` | vòng huấn luyện của tác giả. | không (thay bằng `grkt_api.train_grkt`) |
| `feature2tensor(batch, pos, type, dim)` | đệm và ghép một trường của mọi lượt thành tensor (dim 2: `[B,S]`, dim 3: `[B,S,K]`). | **có** |
| `Runner` | k-fold trên cửa sổ 100 lượt của tác giả. | không (vi phạm tách theo người học) |

---

## 7. `level2/` — dữ liệu GRKT, đồ thị quan hệ, huấn luyện, quỹ đạo

### 7.1. `level2/dataset_spec.py`

| Hàm | Làm gì |
|---|---|
| `DatasetSpec` | mô tả một tập: file chuẩn hoá, file gốc, ánh xạ cột, `time_mode`, `min_interactions`, `max_users`, `top_concepts`. Ba bản đăng ký: `assist2009`, `assist2009_smoke`, `junyi`. |
| `get_spec`, `register_spec` | tra/đăng ký spec. |
| `resolve_canonical_path(spec)` | tìm CSV chuẩn hoá + `materialize_lfs`. |
| `ensure_canonical(spec, out_dir, raw_path, ...)` | trả CSV chuẩn hoá; với Junyi thì tạo một lần từ file gốc và ghi `.meta.json` để lần sau kiểm cấu hình. |
| `_canonicalize_junyi(raw_path, ...)` | đọc `Log_Problem.csv` **theo mảnh 2 triệu dòng**, mã hoá cột thành số nguyên (`_CodeMap`), chọn `top_concepts` khái niệm phổ biến (`_cap_concepts`), lấy mẫu `max_users` người học (seed 42), sắp theo `(user_id, timestamp_ms)`, gán lại `order_id`. |
| `_resolve_columns`, `_match_column`, `_norm_name` | khớp tên cột linh hoạt (chính xác trước, rồi bỏ hoa/thường/gạch dưới). |
| `_to_duration_ms`, `_to_epoch_ms`, `_correct_flags`, `_join_key` | đổi đơn vị thời gian, nhãn đúng/sai, khoá nối. |
| `_load_content_map`, `_find_content_file` | đọc `Info_Content.csv` để lấy tên khái niệm. |
| `describe_schema`, `probe_columns` | in ra cột sẽ dùng mà không nạp cả file. |
| `find_kaggle_input`, `list_kaggle_inputs` | dò file trong `/kaggle/input`. |

### 7.2. `level2/grkt_data.py`

| Hàm | Làm gì | Câu hỏi có thể gặp |
|---|---|---|
| `UserSequence` | chuỗi một người học ở chỉ số GRKT (1-based, 0 = padding): `probs`, `knows`, `corrs`, `times` (phút), `lags`, `skills0` (chỉ số Level 1). `as_grkt_items()` → định dạng tác giả `(prob, [knows], corr, time, lag)`. | |
| `GRKTData` | mọi chuỗi + `concept_num`, `n_probs`, `skill_lookup`, thống kê. `by_users(ids)` lấy chuỗi theo danh sách người học. | |
| `_user_delta_minutes(series, mode)` | Δt giữa hai lượt **trong cùng người học**: `duration_ms` (ASSIST: thời gian làm lượt trước), `timestamp_ms` (Junyi: hiệu timestamp), `step` (mỗi bước 1). | *Sao không diff trên cả bảng?* → sẽ lẫn khoảng cách giữa hai người học. |
| `_cumulative_times(deltas, cap)` | kẹp Δt ≤ 240 phút, cộng dồn, **làm tròn sau khi cộng**. | *Sao làm tròn sau?* → làm tròn từng Δt < 1 phút thành 0 thì pha học/quên chết. |
| `resolve_time_mode` | đổi chế độ thời gian nếu thiếu cột. | |
| `build_grkt_data(bundle, csv, ...)` | đọc CSV, **cùng `prepare_interactions`** như Level 1, factorize skill & problem, dựng `UserSequence`, `_assert_align` từng người. Từ `868838c` còn đếm `order_id` không tăng / lượt trùng **ngay trên chuỗi GRKT** (`stats["order_check"]`) và ghi `stats["prep_version"]`. | *Làm sao chắc cùng dữ liệu với Level 1?* → assert từng người, từng khái niệm, từng nhãn. *Thứ tự thì sao?* → `_assert_align` không so thứ tự, nên có thêm `order_check` riêng. |
| `_assert_align(...)` | so `user_id`, chuỗi khái niệm, chuỗi nhãn với Level 1; lệch là raise. | |
| `build_relevance_maps(sequences, n_knows, cold_thresh)` | đếm `tt, tf, ft, ff` (App. B.1 [GRKT]) **chỉ trên train**; `rel = (tt+ff)/tổng`, `pre = ft/(tf+ft)`; lọc cặp quá ít (`cold_thresh=5`). | xem §3.2 về chiều của `pre`. |
| `binarize(mat, thresh)` | > 0,6 → 1; xoá hàng/cột padding và đường chéo. | |
| `graph_variant(maps, kind)` | `train` / `dense` / `zero` / `shuffled` → `rel_map`, `pre_map` 0/1 + số cạnh. | |
| `_shuffle_binary` | giữ số cạnh ra mỗi đỉnh, hoán vị đích. | |
| `adjacency_from_graphs` | láng giềng 1-hop (rel ∪ pre ∪ preᵀ) cho tiêu chí locality. | |
| `split_sequences(data, split)` | ánh xạ `split.json` của Level 1 sang chuỗi GRKT. | |
| `leak_report(...)` | báo cáo chống rò rỉ Level 2 (giao người học, số cạnh train vs val, thứ tự/lượt trùng trên chuỗi Cấp 1, và dòng "Thứ tự trong chuỗi GRKT" đếm trên chuỗi GRKT). Đọc từng dòng: ON_TAP_BAO_VE §4.1. | |

### 7.3. `level2/grkt_api.py`

| Hàm | Làm gì |
|---|---|
| `ensure_grkt_importable`, `import_grkt` | import `GRKT` như một package (tránh trùng tên `utils.py` với GKT). |
| `make_args(n_knows, n_probs, rel_map, pre_map, ...)` | tạo `Namespace` đủ khoá mà GRKT đọc (`thresh=0.5` vì đồ thị đã 0/1; **không bao giờ 0** vì nhánh `thresh == 0` của tác giả biến mọi thứ thành 1). |
| `build_grkt(args)` | khởi tạo GRKT, đưa lên thiết bị. |
| `batch_items(items, bs, shuffle)` | chia danh sách chuỗi thành batch. |
| `_to_device_feed(model, batch, mode)` | `get_feed_dict` + chuyển thiết bị. |
| `train_grkt(...)` | vòng epoch của Level 2: Adam, BCE, **early stopping theo val AUC toàn cục**, checkpoint cuộn mỗi 10 phút (trọng số + Adam + thứ tự batch + RNG), dừng chủ động khi gần hết giờ (`TrainingPaused`). |
| `_eval_items`, `evaluate_grkt`, `predict_grkt` | đánh giá toàn cục (AUC/ACC/BCE) / trả xác suất cho calibration. |
| `_rng_state`, `_restore_rng` | lưu/khôi phục RNG (vì Gumbel ngẫu nhiên cả lúc eval). |
| `_check_fingerprint` | không resume nếu split/đồ thị/siêu tham số/`prep_version` đổi. Checkpoint cũ thiếu `prep_version` chỉ bị cảnh báo. |
| `run_grkt_experiment(...)` | dựng args → model → `train_grkt` → test → lưu checkpoint; trả dict cùng khoá với `run_experiment` của Level 1. Tham số `ckpt_name` (từ `868838c`): tên file checkpoint do `experiment.py` quyết định; không truyền thì seed 42 → `GRKT-<biến thể>.pt`, seed khác → `GRKT-<biến thể>-seed<s>.pt`. |

### 7.4. `level2/baselines.py`

| Hàm | Làm gì |
|---|---|
| `level1_baseline(shared, specs, ...)` | đánh giá DKT / GKT-Transition / GKT-Zero **trên đúng bundle của Level 2**; không có checkpoint hợp lệ thì train lại (`train_baseline_resumable`). |
| `graph_bundle(shared, graph_type)` | nạp bundle với đồ thị khác nhưng assert giữ nguyên split. |
| `collect_level1_probs` | xác suất + nhãn toàn test (cùng mặt nạ với GRKT) cho bảng calibration. |
| `baseline_fingerprint`, `load_retrained_baseline`, `load_level1_state`, `checkpoint_seed` | quản lý checkpoint baseline, chặn dùng nhầm. Fingerprint gồm đường dẫn dữ liệu, `concept_num`, số người mỗi tập, siêu tham số, seed và `prep_version`. Các file `.json` đi kèm được đọc qua `read_json` (tự tải từ LFS). |

### 7.5. `level2/experiment.py`

| Hàm | Làm gì |
|---|---|
| `run_dataset(dataset, ...)` | Toàn bộ Level 2 cho một tập, 8 bước in ra notebook: **B1** chuẩn hoá + split; **B2** dựng chuỗi GRKT + assert khớp Level 1; **B3** đếm rel/pre từ train, sinh biến thể, leak report; **B4** train GRKT cho từng biến thể; **B5** baseline DKT/GKT cùng split; **B6** ghi quỹ đạo + tiêu chí hợp lý; **B7** calibration; **B8** đa seed cho **mọi** biến thể (train trước, seed trùng trong `SEEDS` bị lọc) + kiểm định ghép cặp với zero. |
| `_paired_vs_zero(ms)` | paired t-test GRKT-<biến thể> − GRKT-zero theo seed, CI 95%, chênh từng seed → `multiseed_paired.csv`. Bỏ dòng trùng (mô hình, seed) trước khi ghép cặp; nếu chênh lệch giống hệt nhau ở mọi seed (độ lệch chuẩn 0) thì p và CI ghi **NaN** thay vì một p cực nhỏ giả tạo. Kết quả hiện có: +0,0143 AUC, p = 0,0009. |
| `run_level2(datasets, ...)` | lặp `run_dataset` qua các tập, nối bảng `level2_all_datasets.csv`. |
| `grkt_activation_gib` | ước lượng VRAM để chọn batch/khái niệm. |
| `_table_row`, `_write_partial_table`, `_safe_plot` | ghi bảng (kể cả khi dừng sớm), vẽ hình không làm hỏng run. |

### 7.6. `level2/trajectory.py` (Nhiệm vụ 2)

| Hàm | Làm gì |
|---|---|
| `record_trajectories(model, sequences, ...)` | chạy GRKT với `feed_dict["rec"]`, thu `m_pre`, `m_tilde`, `m_strong` `[S, NK]` cho từng người học. `perturb="flip_penultimate"` lật bước áp chót. |
| `directional_consistency` | **kiểm chứng cài đặt**: đúng → tăng, sai → giảm tại khái niệm vừa làm và láng giềng (≡ 1 theo cấu trúc). |
| `locality` | **kiểm chứng cài đặt**: tỉ lệ thay đổi nằm trong khái niệm vừa làm ∪ láng giềng 1-hop (≡ 1 với `k_hop=1`). |
| `precedence_consistency(traj, pre_map)` | **tiêu chí thật**: tại bước làm đúng khái niệm phụ thuộc $d$, tỉ lệ $m(\text{tiên quyết}) \ge m(d)$; trừ đi giá trị khi hoán vị cạnh → `precedence_gap` (> 0 là tốt). Chiều cạnh: cột của `pre_map` là tiên quyết. |
| `stability_under_perturbation` | **tiêu chí thật**: lật bước áp chót, so vector mastery cuối (Spearman, tỉ lệ khái niệm thay đổi). |
| `forgetting_monotonicity` | **tiêu chí thật**: trên khái niệm không được chạm, Δm theo Δt phải có độ dốc ≤ 0. |
| `stage_attribution` | độ lớn thay đổi trung bình của từng giai đoạn (củng cố / học-quên / truy xuất). |
| `trajectory_metrics_table` | gộp mọi tiêu chí thành bảng theo biến thể đồ thị. |
| `save_trajectories`, `load_trajectories`, `export_mastery_csv` | lưu/đọc quỹ đạo (`.npz`, `.csv`). |
| `plot_mastery_heatmap`, `plot_mastery_curves` | hình khái niệm × thời gian, đường mastery theo bước. |

### 7.7. `level2/calibration.py`, `level2/archive.py`

| Hàm | Làm gì |
|---|---|
| `brier_score` | $\overline{(p-y)^2}$. |
| `expected_calibration_error` | ECE (10 ngăn, trọng số theo số mẫu) + ACE + bảng từng ngăn. |
| `calibration_table`, `reliability_plot`, `save_reliability_rows` | bảng/hình cho nhiều mô hình. |
| `package_output`, `PeriodicPackager` | nén thư mục kết quả thành zip định kỳ ở luồng nền (để không mất kết quả khi Kaggle cắt phiên). |

---

## 8. Notebook: mỗi ô gọi hàm nào

### `kaggle_level1_full.ipynb`

| Mục | Làm gì | Hàm chính |
|---|---|---|
| §0 | clone đúng `BRANCH`, thêm `sys.path`, ghi commit | `git clone --branch` |
| §0.1 | kiểm tra gói (không cài lại pandas/numpy của Kaggle) | |
| §0.2 | cấu hình (`QUICK_RUN`, `SEEDS`, `TRAIN_MODELS`, ngân sách giờ), tải dữ liệu LFS | `materialize_lfs` (bản trong notebook) |
| §0.3 | khôi phục phiên trước, khoá `run_config.json` (có `prep_version`) | |
| §0.4 | đính output cũ để chỉ đánh giá (`EVAL_ONLY`) | |
| §1 | EDA phân phối độ dài chuỗi | |
| §2, §2.1 | nạp bundle cho từng đồ thị, assert cùng split, thống kê đồ thị | `load_kt_data`, `save_split` |
| §2.2 | độ phủ đồ thị + **leak report** (assert thứ tự/lượt trùng = 0) | `order_leak_lines`, `sequence_order_check` |
| §3 | train mọi (mô hình × seed), resume theo `runs_progress.json` | `build_model`, `_epoch_loop` |
| §4 | bảng chính, đa seed, paired t-test, learning curves | `scipy.stats.ttest_rel` |
| §5 | calibration (Brier, ECE, ACE) | |
| §6 | tắt đồ thị lúc suy luận | `graph_off` |
| §7 | 5 đúng + 5 sai, bằng chứng | `annotate_evidence`, `enrich_cases` (trong notebook) |
| §8 | dashboard người học | `export_learner_view`, `interactive_learner_view` |
| §9 | manifest + `report_summary.md` | |
| §10, §10.1 | đóng gói zip, hướng dẫn resume | |
| §11 | checklist đối chiếu yêu cầu | |

### `kaggle_level2_full.ipynb`

| Mục | Làm gì | Hàm chính |
|---|---|---|
| §0 | clone `BRANCH`, chuẩn bị môi trường | |
| §0.2 | cấu hình (`DATASETS`, Junyi 3.000 người, `VARIANTS_BY_DATASET`, `SEEDS`) | |
| §0.3 | khôi phục phiên trước, khoá `RUN_CONFIG_LOCKED` (có `prep_version`) | |
| §1 | giải thích GRKT và các sửa đổi so với bản gốc (markdown) | |
| §2 | EDA, dò dữ liệu Junyi trong `/kaggle/input` | `find_kaggle_input`, `describe_schema`, `ensure_canonical` |
| §3, §3.1 | chạy toàn bộ Level 2; kiểm tra sản phẩm bắt buộc | `run_level2` → `run_dataset` (B1–B8) |
| §4 | provenance: cấu hình, môi trường, lịch sử train | |
| §5, §5.2 | so sánh DKT / GKT / GRKT; chẩn đoán checkpoint Level 1 cũ | đọc `level2_metrics.csv` |
| §6 | quỹ đạo mastery + tiêu chí hợp lý | đọc `trajectory_metrics.csv` |
| §7 | calibration | |
| §8 | đa seed + kiểm định train − zero | đọc `multiseed_*.csv` |
| §9 | dashboard người học Level 2 | |
| §10–12 | kiểm kê, `report_summary.md`, zip | `package_output` |
| §13–14 | sự cố thường gặp, việc cần làm sau khi chạy | |

---

## 9. Những dòng "khó" hay bị chỉ vào

| Dòng | Giải thích ngắn |
|---|---|
| `df["skill_with_answer"] = df["skill"] * 2 + df["correct"]` (`processing.py`) | mã hoá (khái niệm, đúng/sai) thành một số: khái niệm `s` làm sai → `2s`, đúng → `2s+1`. Đầu vào one-hot 2C chiều của DKT/GKT. |
| `torch.utils.data.random_split(kt_dataset, [...], generator=split_gen)` | chia **theo người học** vì mỗi phần tử dataset là một người; generator có seed → tái lập. |
| `qt_mask = torch.ne(qt, -1)` (GKT) | người học nào còn dữ liệu ở bước này; người đã hết chuỗi (padding) không được cập nhật. |
| `masked_feat = F.embedding(xt[qt_mask], one_hot_feat)` | tra one-hot bằng `F.embedding` trên ma trận đơn vị — cách nhanh để tạo one-hot. |
| `m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)` | ghi đè vị trí khái niệm vừa làm bằng $f_{self}$ → lý do Identity ≡ Zero. |
| `adj = self.graph[masked_qt.long(), :]` / `reverse_adj = self.graph[:, masked_qt]` | hàng $q_t$ (cạnh ra, $A_{ij}$) và cột $q_t$ (cạnh vào, $A_{ji}$) của Eq.2. |
| `res = self.gru(m_next[qt_mask].reshape(-1, H), ht[qt_mask].reshape(-1, H))` | GRUCell xử lý `mask_num × C` vector cùng lúc — mỗi khái niệm của mỗi người là một "mẫu". |
| `y[qt_mask] = torch.sigmoid(y[qt_mask])` | chỉ sigmoid hàng hợp lệ; hàng padding giữ logit (vô hại vì bị mặt nạ, trừ khi occlusion che bước ngay trước dự đoán). |
| `pred_one = pred_answers.clamp(1e-6, 1 - 1e-6)` (`KTLoss`) | tránh `log(0)`. |
| `pack_padded_sequence(..., enforce_sorted=False)` (DKT) | LSTM bỏ qua phần padding. |
| `self.graph.requires_grad = False` | đồ thị tĩnh không bị học — ablation có nghĩa. |
| `h = h.clamp(min=-10, max=10)` (GRKT) | giữ trạng thái hữu hạn qua chuỗi dài. |
| `beta_rel_tilde = beta_matrix * rel_map / rel_map.sum(-1, True).clamp(1)` | Eq.1: trung bình (chia số láng giềng) có trọng số $\beta$, chỉ trên cạnh có trong đồ thị. |
| `F.gumbel_softmax(decision, tau=0.2, hard=True)` | quyết định rời rạc "học / không học" nhưng vẫn lan gradient (straight-through). |
| `delta_time = (new_time - time).clamp(0)` | Δt tới câu kế tiếp (phút) cho đường cong học/quên. |
| `feed_dict["filt"] = probs[:, 1:] > 0` | chỉ tính loss/AUC ở bước có câu hỏi thật, bỏ bước đầu (chưa có lịch sử). |
| `dst, src = np.nonzero(pre)` (`precedence_consistency`) | hàng của `pre_map` = khái niệm phụ thuộc, cột = tiên quyết. |
