# Giải thích GKT & GRKT — codebase, dữ liệu, đồ thị và luồng huấn luyện

Tài liệu này giải thích **hai mô hình đang nằm trong repo**:

| Thư mục | Nội dung | Nguồn |
|---|---|---|
| `GKT/` | **DKT** (baseline không đồ thị) và **GKT** (Graph-based Knowledge Tracing, Nakagawa et al., 2019) | `jhljx/Graph-based-Knowledge-Tracing` |
| `GRKT/` | **GRKT** (Cui et al., KDD 2024): graph + retrieval + memory strengthening/forgetting | `JJCui96/GRKT`, đã package-hoá và sửa có kiểm soát |

Đọc kèm: `GKT/README.md`, `README.md` (gốc repo, phần "Cấp độ 1"/"Cấp độ 2"), `KE_HOACH.md`.

---

## 0. Bản đồ file

### GKT

| File | Vai trò |
|---|---|
| `GKT/train.py` | CLI + vòng lặp train/test + lưu checkpoint |
| `GKT/processing.py` | CSV → chuỗi theo người học, split train/val/test, dựng đồ thị tĩnh |
| `GKT/models.py` | `GKT`, `MultiHeadAttention`, `VAE`, `DKT` |
| `GKT/layers.py` | `MLP`, `EraseAddGate`, `ScaledDotProductAttention`, `MLPEncoder`, `MLPDecoder` |
| `GKT/metrics.py` | `KTLoss` (NLL + AUC/ACC), `VAELoss` |
| `GKT/utils.py` | dựng các loại đồ thị, gumbel-softmax, KL, NLL Gaussian |
| `GKT/data/` | `skill_builder_data.csv` (ASSISTments 2009, ~83 MB), `assistment_test15.csv` (15 học sinh, smoke test) |

### GRKT

| File | Vai trò |
|---|---|
| `GRKT/model/GRKT.py` | toàn bộ mô hình GRKT (forward 4 pha) |
| `GRKT/BaseModel.py` | khung train/eval: `train_an_epoch`, `eval_an_epoch`, `fit`, `test` |
| `GRKT/Runner.py` | k-fold trên các cửa sổ chuỗi (⚠️ có rò rỉ người học — xem §3.7) |
| `GRKT/Training.py` | entry point CLI (đã sửa: registry thay `eval()`, chặn dataset ngoài `assist09`) |
| `GRKT/utils.py` | `feature2tensor` (padding batch), `get_performance`, `reset_seed` |
| `GRKT/metric.py` | `AUC`, `ACC` (gom toàn bộ prediction) |
| `GRKT/data/assist09/assist09_preprocess.py` | CSV → `assist09.bin` (chuỗi + `rel_map`/`pre_map`) |
| `GRKT/data/assist09/assist09.bin` | dữ liệu đã đóng gói (pickle) |

Ngoài ra **pipeline thật của repo** không dùng `GRKT/Runner.py` mà dùng `level2/grkt_api.py`
(`train_grkt`) + `level2/grkt_data.py` (`build_grkt_data`, `build_relevance_maps`) để tránh rò rỉ.

---

## 1. Nền chung: bài toán và ký hiệu

**Knowledge Tracing (KT)**: cho lịch sử làm bài của một người học

```
(q_1, r_1), (q_2, r_2), ..., (q_S, r_S)     q_i = khái niệm (KC) của câu hỏi, r_i ∈ {0,1}
```

dự đoán xác suất làm đúng câu tiếp theo: `p(r_{i+1} = 1 | q_{≤i}, r_{≤i})`.

Ký hiệu dùng trong tài liệu:

| Ký hiệu | Nghĩa | GKT | GRKT |
|---|---|---|---|
| `C` / `n_knows` | số khái niệm (skill) | 123 với ASSIST09 | 123 |
| `NK` | `n_knows + 1` (chừa index 0 làm padding) | — | 124 |
| `B`, `S` | batch size, độ dài chuỗi (theo batch, đã padding) | 128, ≤200 | 128, ≤100/200 |
| `H` (GKT) | `hid_dim` — chiều trạng thái mỗi khái niệm | 32 | — |
| `E` (GKT) | `emb_dim` — chiều embedding | 32 | — |
| `DH` (GRKT) | `d_hidden` — chiều embedding câu hỏi/khái niệm | — | 128 |
| `KH` (GRKT) | `k_hidden` — chiều "trạng thái mastery" mỗi khái niệm | — | 16 |

**Khác biệt cốt lõi về ký ức**:

* **GKT**: mỗi khái niệm có một vector trạng thái `h_t[c] ∈ R^H`; mô hình là **mạng hồi quy trên đồ thị**, cập nhật *tất cả* khái niệm ở mỗi bước bằng GNN + GRU.
* **GRKT**: mỗi khái niệm có vector mastery `h[c] ∈ R^KH`; không có RNN — mỗi bước gồm **truy hồi (retrieval)** trên đồ thị, **củng cố** (đúng → gain, sai → loss), rồi **học/quên theo thời gian** (Δtime).

---

## 2. GKT

### 2.1 Câu chuyện mô hình

GKT coi "trạng thái tri thức" là một **ma trận** `h_t ∈ R^{C×H}` — mỗi hàng là mức hiểu một khái niệm. Sau khi học sinh trả lời câu `q_t` đúng/sai, mô hình:

1. **Aggregate**: gắn embedding `(khái niệm, đáp án)` vào đúng hàng `q_t`, các hàng khác chỉ có embedding khái niệm thuần.
2. **GNN aggregate**: lan truyền thông tin từ các khái niệm lân cận trên đồ thị `A`.
3. **Erase–Add gate** + **GRU**: trộn trạng thái cũ và thông tin mới → `h_{t+1}`.
4. **Predict**: chiếu `h_{t+1}` thành xác suất đúng cho **mọi** khái niệm, rồi lấy đúng khái niệm của câu kế tiếp.

### 2.2 Dữ liệu vào và tiền xử lý

`GKT/processing.py::load_dataset` (dòng 41) nhận CSV có 3 cột bắt buộc: `user_id`, `skill_id`, `correct` (có `skill_name` thì dùng cho giải thích/dashboard).

```
CSV
 ├─ dropna(skill_id)                                  # bỏ dòng không có KC
 ├─ groupby(user_id).filter(len >= min_seq_len=3)     # bỏ người học quá ngắn
 ├─ skill = pd.factorize(skill_id, sort=True)         # KC → 0..C-1
 ├─ skill_with_answer = skill*2 + correct             # 0..2C-1  ← "feature"
 ├─ groupby(user_id, sort=False) → mỗi người 1 chuỗi, cắt max_seq_len=200
 └─ random_split theo NGƯỜI HỌC: 0.6 / 0.2 / 0.2 (seed=42)
```

Kết quả mỗi mẫu (một người học) là **ba list song song, cùng độ dài S**:

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| `features` | `[B, S]` | chỉ số `skill*2 + correct`, padding **−1** |
| `questions` | `[B, S]` | chỉ số khái niệm 0-based, padding **−1** |
| `answers` | `[B, S]` | nhãn 0/1, padding **−1** |

`pad_collate` (dòng 30) pad bằng −1 trong mỗi batch.

⚠️ Lưu ý quan trọng: **feature tại bước `i` đã chứa đáp án `r_i`**. Mô hình cập nhật trạng thái sau khi biết `r_i` rồi mới dự đoán `r_{i+1}` (loss lấy `answers[:, 1:]`). Vì vậy `pred_res` có độ dài `S−1`.

Đầu ra `load_dataset`: `concept_num`, `graph` (hoặc `None`), 3 `DataLoader`.

### 2.3 Xây dựng đồ thị — 9 biến thể

Đồ thị là ma trận `A ∈ R^{C×C}`, `A[i,j]` = mức liên hệ từ khái niệm `i` sang `j`.

| `--graph-type` | Công thức | Học được? | Ý nghĩa |
|---|---|---|---|
| `Dense` | `A[i,j] = 1/(C−1)`, chéo 0 | không | đồ thị đầy đủ, mỗi láng giềng trọng số ~1/C |
| `Transition` | đếm `i→j` liên tiếp trong **tập train**, chuẩn hoá theo hàng | không | "học xong i thì hay làm j" |
| `DKT` | đọc ma trận từ `dkt_graph.txt` | không | đồ thị ngoài (nếu có) |
| `Identity` | `A = I` | không | ablation |
| `Zero` | `A = 0` | không | ablation "bỏ hết cạnh" |
| `Shuffled` | `Transition` nhưng hoán vị đích | không | ablation phá cấu trúc, giữ ~số cạnh |
| `PAM` | `A = rand(C,C)` | **có** (`nn.Parameter`) | đồ thị tiềm ẩn học tự do |
| `MHA` | attention `emb_c(q_t)` → `emb_c(all)` | tham số attention | `n_head = edge_types`, mỗi head là một đồ thị |
| `VAE` | suy luận cạnh kiểu NRI + Gumbel-softmax | **có** | mỗi cạnh có `edge_types` loại |

**Transition** (`processing.py:176`): chỉ đếm trên `train_dataset.indices`, nên **không rò rỉ** từ val/test. Đếm `graph[pre, next] += 1` cho mọi cặp liên tiếp, xoá chéo, rồi chuẩn hoá hàng:

```
A[i,j] = P(khái niệm kế tiếp = j | hiện tại = i)   (hàng i tổng = 1)
```

**MHA** (`models.py:328`): query = embedding của `q_t`, key = embedding của *tất cả* khái niệm. Mask tự chú ý (không cho `c = q_t`) bằng `-1e9`, rồi softmax. Kết quả là `edge_types` ma trận attention `[E_types, C, C]`, các hàng ứng với `q_t` được ghi lại để dùng/visualize.

**VAE** (`models.py:396` + `layers.MLPEncoder/MLPDecoder`): với mỗi lần làm bài, dựng danh sách cạnh **liên quan tới `q_t`** (cả hai chiều, bỏ self-loop) → encoder dự đoán logits loại cạnh `[edge_num, edge_types]` → Gumbel-softmax → decoder tái tạo embedding khái niệm. Loss phụ:

```
L_vae = mean_t [ NLL_gaussian(rec_t, emb_c) + KL(q(z|x) || prior) ]
```

Đồ thị dùng để lan truyền lấy từ `edges` đã sample. `self.graphs` (requires_grad=False) chỉ để lưu/visualize.

### 2.4 Kiến trúc và tham số

`GKT.__init__` (dòng 20):

| Thành phần | Chi tiết |
|---|---|
| `emb_x` | `nn.Embedding(res_len*C, E)` — embedding của `(khái niệm, đáp án)` |
| `emb_c` | `nn.Embedding(C+1, E, padding_idx=-1)` — embedding khái niệm; hàng cuối = vector 0 cho padding |
| `one_hot_feat`, `one_hot_q` | ma trận one-hot dùng cho `F.embedding` (tra cứu thay cho one-hot thật) |
| `f_self` | MLP `H+E → H → H` (2 lớp + BatchNorm) |
| `f_neighbor_list` | với đồ thị tĩnh: **2** MLP (`in`, `out`); với `MHA`/`VAE`: **`edge_types`** MLP `2(H+E) → H` |
| `erase_add_gate` | `EraseAddGate(H, C)` — cổng xoá/thêm, trọng số riêng từng khái niệm |
| `gru` | `nn.GRUCell(H, H)` |
| `predict` | `nn.Linear(H, 1)` |

`MLP` (`layers.py:14`): `Linear → ReLU → Dropout → Linear → ReLU → BatchNorm1d`. BatchNorm có guard cho batch 1 hàng (`layers.py:36`).

`EraseAddGate` (`layers.py:54`): với `w_c` là scalar học được của từng khái niệm:

```
m' = m − w_c · σ(erase(m)) ⊙ m + w_c · tanh(add(m))
```

### 2.5 Luồng forward từng bước

`GKT.forward` (dòng 280) — với `h_0 = 0`:

```python
for i in range(S):
    xt, qt = features[:, i], questions[:, i]
    tmp_ht = self._aggregate(xt, qt, ht, B)          # [B, C, H+E]
    h_next = self._update(tmp_ht, ht, qt)            # [B, C, H]
    ht[mask] = h_next[mask]
    yt = self._predict(h_next, qt)                   # [B, C]  (sigmoid)
    if i < S-1:
        pred_list.append(self._get_next_pred(yt, questions[:, i+1]))  # [B]
```

**(1) Aggregate** (`_aggregate`, dòng 79)

```
e[b,c] = emb_c[c]                        với mọi khái niệm
e[b,q_i[b]] = emb_x[x_i[b]]              ghi đè hàng của câu đang làm
tmp_ht = concat(ht, e)                   [B, C, H+E]
```

**(2) GNN aggregate** (`_agg_neighbors`, dòng 111) — công thức (1) của bài báo

Với từng người học `b` có `q = q_i[b]`:

```
self_ht   = tmp_ht[b, q]
self_feat = f_self(self_ht)                                  # [H]
neigh_in  = concat(self_ht, tmp_ht[b, c])                    # [2(H+E)] cho từng c
```

* Đồ thị tĩnh / PAM:
  ```
  neigh[c] = A[q, c] · f_in(neigh_in) + A[c, q] · f_out(neigh_in)
  ```
  (`A[q,c]`: cạnh đi ra từ `q`; `A[c,q]`: cạnh đi vào `q` — vì vậy có 2 MLP.)
* `MHA`/`VAE`:
  ```
  neigh[c] = Σ_k graphs[k][q, c] · f_k(neigh_in)      (MHA chia thêm cho edge_types)
  ```

Sau đó:

```
m_next[b, c] = neigh[c]  với c ≠ q
m_next[b, q] = self_feat
```

**(3) Update** (`_update`, dòng 172)

```
m_next = erase_add_gate(m_next)                  # cho mọi khái niệm của người học có q ≠ -1
h_next = GRUCell(input=m_next[·,c], hidden=ht[·,c])   với mọi c
```

**(4) Predict** (`_predict`, dòng 203)

```
y[b, c] = σ(W·h_next[b,c] + bias)      # xác suất đúng cho MỌI khái niệm
pred_i   = y[b, q_{i+1}[b]]            # chọn đúng câu kế tiếp (_get_next_pred)
```

Output của `forward`: `pred_res [B, S−1]`, cùng `ec_list/rec_list/z_prob_list` (chỉ VAE dùng) và `states` nếu `return_state=True` (mỗi `h_t` — dùng cho dashboard Cấp độ 1).

### 2.6 Loss và metric

`KTLoss` (`metrics.py:12`):

1. `real = answers[:, 1:]` (nhãn dịch 1 bước), mask `real != -1`.
2. Clamp prediction về `[1e-6, 1−1e-6]`.
3. Dựng `[B, 2, S−1]` = `[log(1−p), log p]` → `NLLLoss(ignore_index=-1)`.
4. Trả thêm AUC (`roc_auc_score`) và ACC (argmax so với nhãn) **trên từng batch**.

> Vì tính AUC **theo batch rồi lấy trung bình** (`train.py:262-268`), con số AUC in ra trong `GKT/train.py` không phải AUC toàn cục. `level1/pipeline.py` đã sửa bằng cách gom toàn bộ prediction rồi mới tính một lần — xem README gốc repo.

VAE thêm `VAELoss` (`metrics.py:56`) = trung bình theo thời gian của `(NLL Gaussian tái tạo + KL)`.

### 2.7 Huấn luyện

`GKT/train.py`:

| Bước | Dòng | Ghi chú |
|---|---|---|
| Parse args | 23–59 | `--model {GKT,DKT}`, `--graph-type`, `--hid-dim 32`, `--emb-dim 32`, `--epochs 50`, `--batch-size 128`, `--lr 1e-3` |
| Tạo thư mục log | 80–104 | `logs/expGKT-<Graph><timestamp>/` chứa `metadata.pkl`, `model.pt`, `optimizer.pt`, `scheduler.pt`, `log.txt` |
| Nạp dữ liệu | 106–113 | `load_dataset(...)` |
| Dựng model | 115–133 | `GKT` hoặc `DKT`; `graph_model` (`MHA`/`VAE`) tách riêng |
| Optimizer | 137–138 | `Adam(lr)`, `StepLR(step_size=200, gamma=0.5)` |
| Vòng epoch | 176–321 | mỗi batch: forward → `KTLoss` → (VAE: + `VAELoss`) → `backward` → `step`; đánh giá valid; lưu checkpoint nếu `val_loss` tốt nhất |
| Test | 324–395 | nạp lại checkpoint tốt nhất, tính loss/AUC/ACC trên test |

Cấu hình mặc định: **không** gradient clipping, dropout 0, BatchNorm trong MLP, seed 42, `cudnn.deterministic=True`.

### 2.8 Review GKT — các điểm cần biết

| # | Vị trí | Vấn đề | Mức | Hệ quả |
|---|---|---|---|---|
| 1 | `GKT/train.py:128-131` | `GKT(...)` **không truyền `binary=args.binary`** → bên trong luôn `res_len=12` | thấp (tiêu tốn) | `emb_x` và `one_hot_feat` là `12C` thay vì `2C` → thừa ~6× bộ nhớ/tham số và matmul; index vẫn hợp lệ nên **không sai kết quả**. Cờ `--binary`/`--result-type` vô tác dụng |
| 2 | `GKT/train.py:137-138` và `156-158` | Tạo optimizer/scheduler **hai lần**; nhánh `--load-dir` nạp state từ `save_dir` (thư mục mới) chứ không phải `load_dir` | trung bình | Chế độ finetune/resume bị lỗi; train từ đầu không ảnh hưởng |
| 3 | `GKT/train.py:217` | `scheduler.step()` **mỗi batch** nhưng `step_size=200` (ý định là epoch) | trung bình | LR giảm 0.5 mỗi 200 batch, khác thiết kế ban đầu |
| 4 | `GKT/metrics.py:34-44` | AUC/ACC tính **theo batch** rồi lấy trung bình | trung bình | Số liệu lệch so với AUC toàn cục; batch một lớp bị bỏ (`-1`) |
| 5 | `GKT/train.py:161` + `utils.py:121` | `--prior` hard-code prior 4 phần tử `[0.91,0.03,0.03,0.03]` nhưng `edge-types` mặc định = 2 | trung bình (nhánh không mặc định) | Bật `--prior` với `edge-types=2` sẽ lỗi broadcast |
| 6 | `GKT/train.py:46` | `--hard` được parse nhưng `VAE.forward` gọi `gumbel_softmax(..., tau)` không truyền `hard` | thấp | Cờ chết |
| 7 | `GKT/utils.py:29-55` | `build_shuffled_graph` hoán vị **toàn bộ cột**; có thể trùng đích (ghi đè) và đẩy cạnh vào đường chéo rồi bị xoá | thấp | Số cạnh thực tế ≠ `Transition`; khi báo cáo nên ghi "xấp xỉ cùng mật độ" |
| 8 | `GKT/models.py:153-155` (gọi trong vòng `forward`) | VAE/`_get_edges` được tính lại **mỗi timestep** dù input `emb_c` không đổi | thấp (hiệu năng) | Chậm; loss VAE lặp cùng một giá trị `S` lần |
| 9 | `GKT/models.py:166-168` | `m_next` là **view** của `tmp_ht` rồi ghi in-place | thấp (dễ vỡ) | Nếu sau này tái sử dụng `tmp_ht` sẽ bị sửa nhầm |
| 10 | `GKT` với `A=I` vs `A=0` | Hai cấu hình cho kết quả giống hệt | — | Đã chứng minh trong README gốc repo: cột `c≠q_t` đều bằng 0, ô `c=q_t` bị `self_features` ghi đè |
| 11 | `GKT/outputs/*.csv` | Sinh từ `assistment_test15.csv` (15 học sinh, AUC ~0.52) | — | **Không dùng cho báo cáo**; số liệu báo cáo nằm ở `outputs/level1/` |

---

## 3. GRKT

### 3.1 Câu chuyện mô hình

GRKT trả lời ba câu hỏi mà GKT gộp chung trong GRU:

1. **Retrieval** — "Câu hỏi này cần những khái niệm nào, và các khái niệm liên quan đang ở mức nào?" → kích hoạt mastery theo đồ thị.
2. **Memory strengthening** — "Vừa làm đúng/sai, khái niệm (và các khái niệm liên quan) mạnh lên/yếu đi bao nhiêu?" → gain/loss lan truyền trên đồ thị.
3. **Learning & forgetting** — "Từ lần làm trước đến giờ đã bao lâu? Kiến thức vừa học có 'ngấm' chưa? Cái gì bị quên?" → hàm mũ theo Δtime.

Đồ thị của GRKT gồm **hai quan hệ** (khác GKT chỉ một ma trận):

* `rel_map` — quan hệ **đồng xuất hiện** (relevance).
* `pre_map` — quan hệ **tiên quyết** (prerequisite), và `sub_map = pre_mapᵀ` (chiều ngược lại).

### 3.2 Dữ liệu vào

#### a) Định dạng gốc của GRKT (upstream)

`assist09_preprocess.py:35-45` tạo mỗi tương tác là một tuple:

```python
(prob, knows, corr, timestamp, timelag)
# prob      : id câu hỏi, 1-based
# knows     : list id khái niệm, 1-based  (ASSIST09: list 1 phần tử)
# corr      : bool
# timestamp : order_id (bản gốc dùng mốc thời gian thô)
# timelag   : ms_first_response_time
```

Cả `prob` và `knows` đều **1-based** vì index `0` được giữ cho padding. `assist09.bin` chứa:

```python
{
  'n_probs': ..., 'n_knows': ..., 'n_feats': 5,
  'data': [ [tuple, tuple, ...], ... ],   # mỗi phần tử = 1 cửa sổ ≤100 tương tác
  'rel_map': np.ndarray (n_knows+1, n_knows+1),
  'pre_map': np.ndarray (n_knows+1, n_knows+1),
}
```

Cửa sổ (`assist09_preprocess.py:94-99`):

```python
while len(seq) >= 10:
    data.append(seq[:100])      # cắt 100
    seq = seq[100:]             # phần còn lại
```

→ người học < 10 tương tác bị bỏ; người dài bị chia thành nhiều mẫu **độc lập** (mẫu sau mất lịch sử trước đó).

#### b) Cách repo này dùng (Level 2)

`level2/grkt_data.py::build_grkt_data` (dòng 170) dựng lại chuỗi từ CSV chuẩn hoá **giống hệt** `GKT/processing.load_dataset` (cùng `factorize`, cùng `min_seq_len=3`, `max_seq_len=200`, cùng thứ tự người học) rồi `_assert_align` từng người học với bundle Cấp độ 1. Khác biệt:

* `knows = [[skill+1]]` — mỗi bước đúng **một** khái niệm (`max_concepts=1`).
* `times` = **phút cộng dồn** trong từng người học, chặn mỗi bước ≤ 240 phút (`_cumulative_times`).
  Lý do: `exp(-(n+1)·Δt·k)` với Δt thô (giây/mili-giây) sẽ underflow → "quên sạch" sau 1 bước.
* Không cắt cửa sổ 100; mỗi người học là **một chuỗi** (tối đa `MAX_SEQ_LEN=200`), split theo người học.

#### c) Batch và padding

`BaseModel.get_feed_dict` (dòng 46) + `utils.feature2tensor`:

| Key | Shape | Nguồn | Padding |
|---|---|---|---|
| `probs` | `[B, S]` | pos 0 | **0** |
| `knows` | `[B, S, K]` | pos 1 | 0 (và pad list kỹ năng) |
| `corrs` | `[B, S]` | pos 2 | `False` |
| `times` | `[B, S]` | pos 3 | 0 |
| `lags` | `[B, S]` | pos 4 | 0 — **không dùng** trong `GRKT.forward` |
| `filt` | `[B, S−1]` | `probs[:,1:] > 0` | mặt nạ vị trí hợp lệ |
| `labels` | `[N_valid]` | `corrs[:,1:][filt]` | — |

### 3.3 Xây dựng đồ thị `rel_map` / `pre_map`

#### a) Bộ đếm (upstream và `level2/grkt_data.py::build_relevance_maps` giống nhau)

Với **mỗi người học**, duy trì hai vector đếm theo khái niệm: `t[k]` = số lần đúng, `f[k]` = số lần sai *tính đến trước bước hiện tại*. Tại một tương tác `(k, corr)`:

```python
if corr:
    tt[:, k] += t ;  tt[k, :] += t      # i đúng trước, k đúng bây giờ  (cùng đúng)
    ft[:, k] += f ;  tf[k, :] += f      # i sai trước,  k đúng bây giờ  (flip sai→đúng)
else:
    ff[:, k] += f ;  ff[k, :] += f      # cùng sai
    tf[:, k] += t ;  ft[k, :] += t      # i đúng trước, k sai bây giờ  (flip đúng→sai)
# cập nhật sau cùng
t[k] += 1 hoặc f[k] += 1
```

Bốn bộ đếm có nghĩa: `tt` = cùng đúng, `ff` = cùng sai, `tf` = đúng→sai, `ft` = sai→đúng.

#### b) Công thức

```python
rel_filt = (tt + ff + tf + ft) >= 4 * cold_thresh      # >= 20 lần đồng xuất hiện
pre_filt = (tf + ft)           >= 2 * cold_thresh      # >= 10 lần "flip"

rel = (tt + ff) / (tt + ff + tf + ft)                  # tỉ lệ CÙNG kết quả
pre = ft       / (tf + ft)                             # tỉ lệ flip sai→đúng
rel *= rel_filt ;  pre *= pre_filt
rel[i,i] = pre[i,i] = 0
```

* `rel_map[i,j]` ∈ [0,1] = **mức đồng xuất hiện cùng kết quả** của hai khái niệm (càng gần 1 càng "hay đi cùng nhau và cùng đúng/sai").
* `pre_map[i,j]` = trong các cặp có kết quả *lật*, tỉ lệ lật theo hướng `i sai → j đúng`; tác giả dùng như **độ mạnh quan hệ tiên quyết theo hướng i→j**. Đây là heuristic đếm, không phải quan hệ nhân quả.
* `sub_map = pre_mapᵀ` = cùng quan hệ nhưng chiều ngược.

`level2/grkt_data.py` bổ sung 2 điều so với upstream:

1. Chỉ đếm **từ tập train** (`split_sequences` + `graphs_train_only.npz`) — chống rò rỉ.
2. Xoá hàng/cột `0` (padding) và hỗ trợ đa khái niệm (duyệt mọi `know` trong lượt, cập nhật bộ đếm sau cùng).

#### c) Từ ma trận đếm → ma trận lan truyền

GRKT **không** dùng trực tiếp `rel_map`; mô hình nhân với độ tương đồng học được `beta_matrix` rồi chuẩn hoá hàng (`GRKT.py:226-235`):

```python
beta  = σ( W_req · W_knowᵀ )                            # [NK, NK], W_know = know_embedding.weight
β_rel = beta ⊙ rel_map / rowsum(rel_map).clamp(1)       # trung bình có trọng số trên láng giềng
β_pre = beta ⊙ pre_map / rowsum(pre_map).clamp(1)
β_sub = beta ⊙ sub_map / rowsum(sub_map).clamp(1)
```

`α` (mức câu hỏi "cần" khái niệm) tính riêng (`GRKT.py:223`):

```python
alpha[b,s] = σ( W_req · prob_embedding[p_{b,s}] · W_knowᵀ )   # [B, S, NK]
```

#### d) Binarize và biến thể đồ thị

`args.rel_map/pre_map` được đưa vào dạng **0/1**: `(map > args.thresh)` (`GRKT.py:133-138`).
`level2/grkt_data.py::graph_variant` tạo 4 biến thể: `train` (đếm từ train, ngưỡng 0.6), `dense` (đầy đủ), `zero` (bỏ cạnh), `shuffled` (giữ bậc, đổi đích).

⚠️ Không được đặt `thresh=0`: nhánh `if args.thresh == 0` trong `GRKT.py:140-142` gán **toàn bộ** ma trận = 1 (kể cả hàng/cột padding). Level 2 dùng `thresh=0.5` trên ma trận đã 0/1.

### 3.4 Kiến trúc và tham số

| Thành phần | Kích thước | Vai trò |
|---|---|---|
| `init_hidden` | `[NK, KH]` | trạng thái mastery khởi tạo (dùng chung cho mọi người học) |
| `know_embedding` | `[NK, DH]`, `padding_idx=0` | embedding khái niệm |
| `prob_embedding` | `[n_probs+1, DH]`, `padding_idx=0` | embedding câu hỏi (GRKT dùng **problem id**, GKT thì không) |
| `know_master_proj` | `Positive_Linear(KH→1, softmax)` | gộp `KH` kênh → 1 scalar mastery/khái niệm |
| `req_matrix` | `Linear(DH→DH, no bias)` | chiếu dùng cho cả `alpha` và `beta` |
| `agg_{rel,pre,sub}_matrix` | `Positive_Linear(KH→KH, softmax)` × `k_hop` | biến đổi trong pha retrieval |
| `gain_ffn`, `loss_ffn` | `Linear(2DH+KH → DH) → ReLU → Linear(DH → KH)` | tính gain/loss từng khái niệm |
| `gain_/loss_matrix_*`, `gain_/loss_output_*` | `Linear(KH→KH)` × 3 quan hệ × `k_hop` | lan truyền gain/loss trên đồ thị |
| `learn_kernel_*`, `forget_kernel_*` | `Linear(DH→KH)` rồi `Linear(KH→KH)` × 3 quan hệ × `k_hop` | học tốc độ học/quên **từ đồ thị** |
| `decision_mlp` | `Linear(4DH+KH → 2DH) → ReLU → Linear(→2)` | quyết định "có áp dụng learn vector hay không" (Gumbel) |
| `learn_mlp` + `learn_matrix_*`, `learn_output_*` | như gain | vector học bị trễ theo thời gian |
| `rel_matrix` | `Linear(DH→DH)` | **không dùng** (xem §3.8) |

`Positive_Linear` (`GRKT.py:29`) = `x @ softmax(W)`: trọng số không âm, mỗi cột tổng = 1 → mọi phép biến đổi tuyến tính trong đồ thị đều "dương". `pos_mode` trong `parse_args` **không có tác dụng** (mọi chỗ đều hard-code `'softmax'`).

### 3.5 Luồng forward — 4 pha

`GRKT.forward` (dòng 206). Giả sử `B` chuỗi, `S` bước, `K` khái niệm/bước (ở repo này `K=1`).

#### Pha A — Tính trước (không phụ thuộc bước)

```python
W          = know_embedding.weight                        # [NK, DH]
alpha      = σ(req(W_prob)) · Wᵀ                          # [B, S, NK]  "câu hỏi cần KC nào"
beta       = σ(req(W)) · Wᵀ                               # [NK, NK]
β_rel, β_pre, β_sub                                       # [NK, NK]  (đã chuẩn hoá hàng)

k_learn  = softplus( message_passing(W; β_rel, β_pre, β_sub) ) · alpha_const   # [NK, KH]
k_forget = softplus( message_passing(W; β_rel, β_pre, β_sub) ) · alpha_const   # [NK, KH]
```

`message_passing` = `k_hop` vòng, mỗi vòng:

```python
v_rel = β_rel @ Linear_kh(W_prev)     →  ReLU  →  Linear(KH→KH)
# tương tự cho pre, sub; cộng dồn vào W_prev
```

→ `k_learn[c,:]` / `k_forget[c,:]` là **tốc độ học/quên riêng của khái niệm `c` và từng kênh mastery**, suy ra từ cấu trúc đồ thị.

> ⚠️ Đừng lẫn hai chữ "alpha": `alpha[b,s,c]` là **ma trận nhu cầu** (câu hỏi `s` cần khái niệm `c` bao nhiêu, học được),
> còn `args.alpha = 0.01` là **hệ số thang thời gian** nhân vào `k_learn`/`k_forget`. Chúng độc lập với nhau.

#### Pha B — Truy hồi và dự đoán (mỗi bước `i`)

```python
h = h.clamp(-10, 10)                                     # [B, NK, KH]

know_emb  = mean(emb(knows[:, i]))                       # [B, DH]   (K=1 → chính nó)
z_i       = concat([know_emb, prob_emb[:, i]])           # [B, 2DH]

h_tilde = h
for hop in range(k_hop):
    for rel in (rel, pre, sub):
        v = PositiveLinear_rel(h_tilde)                  # [B, NK, KH]
        v = alpha[:, i, :, None] * v                     # nhân "nhu cầu" của câu hỏi
        h_tilde = h_tilde + β_rel @ v                    # lan truyền trên đồ thị

master_full = know_master_proj(h_tilde)                  # [B, NK]  mastery mọi KC
master      = mean_{c ∈ q_i} master_full[c]              # [B]      (mask KC padding)
diff        = MLP_diff(z_i)                              # [B]      độ lệch học được
score_i     = σ(master − diff)                           # [B]      ← dự đoán cho câu i
```

Điểm mấu chốt: `score_i` được tính **trước** khi dùng `r_i` → không rò rỉ nhãn. Đây là khác biệt then chốt so với GKT (GKT đưa `r_i` vào input rồi dự đoán `i+1`); kết quả cuối cùng vẫn là "dự đoán lần làm bài kế tiếp" nhưng cách bố trí của GRKT sạch hơn về mặt thời điểm.

#### Pha C — Củng cố (strengthening) theo đúng/sai

```python
for c in q_i:
    gain_c = gain_ffn([z_i, h[c]])                       # [KH]  (cần "học" bao nhiêu)
    loss_c = loss_ffn([z_i, h[c]])                       # [KH]  (mất bao nhiêu nếu sai)

Δ = scatter(gain_c)  →  lan truyền (β_rel, β_pre, β_sub, có nhân alpha)  →  ReLU   # [B, NK, KH]
Λ = scatter(loss_c)  →  lan truyền tương tự                              →  ReLU

h = h + r_i · Δ − (1 − r_i) · Λ                          # [B, NK, KH]
learn_count[c] += 1 nếu (r_i · Δ[c] > 0)
```

Gain/loss của khái niệm vừa làm **lan sang các khái niệm lân cận** (rel/pre/sub) — đây là phần "graph" của việc học, không chỉ phần truy hồi.

#### Pha D — Chuyển sang bước `i+1`: học trễ và quên

Chỉ chạy khi `i < S−1` (dùng `knows[:, i+1]`, `times[:, i+1]`):

```python
feat = [z_i, z_{i+1}, h[q_i], h[q_{i+1}]]                 # [B, 2K, 4DH+KH]

decision = GumbelSoftmax(decision_mlp(feat), tau, hard=True)[..., 0]   # 1 = áp dụng, 0 = bỏ
learn    = learn_mlp(feat) * decision                    # [B, 2K, KH]
ΔL       = scatter(learn) → lan truyền (rel/pre/sub) → ReLU            # [B, NK, KH]

Δt          = clamp(t_{i+1} − t_i, ≥0)                   # [B]
e_learn     = exp( −(count[c]+1) · Δt · k_learn[c] )     # [B, NK, KH]
e_forget    = exp( −(count[c]+1) · Δt · k_forget[c] )    # [B, NK, KH]

h = h + (1 − e_learn) · ΔL                               # "học ngấm" theo thời gian
hist_gain = clamp(h − h_init, ≥0)
h = h − hist_gain · (1 − e_forget)   tại các KC có ΔL == 0   # quên phần đã tích luỹ

learn_count[c] += 1 nếu ΔL[c] > 0
```

Diễn giải:

* `Δt` càng lớn → `1 − e_learn` càng gần 1 → **học trễ ngấm nhiều hơn**; `e_forget` càng nhỏ → **quên nhiều hơn**.
* `(count+1)` khiến khái niệm đã luyện nhiều lần có tốc độ học/quên khác khái niệm mới.
* Quên chỉ tác động lên **phần gain tích luỹ** (`h − h_init`), không đụng tới tri thức khởi tạo.
* `decision` là **Gumbel-Softmax hard** → mô hình học *có/không* áp dụng vector học cho từng khái niệm (rời rạc nhưng vẫn lan gradient).

#### Đầu ra

```python
scores = stack(scores, -1)                    # [B, S]
feed_dict['scores'] = scores[:, 1:][filt]     # [N_valid]  (bỏ bước 0 vì chưa có lịch sử)
# labels = corrs[:, 1:][filt]
```

### 3.6 Loss và metric

* `BaseModel.loss` (dòng 62): `BCE(scores, labels)` trên các vị trí hợp lệ.
* `metric.AUC` / `metric.ACC`: gom toàn bộ `scores` của cả split rồi tính **một lần** (AUC toàn cục, ACC ngưỡng 0.5).
* `get_performance` (dùng cho early stopping ở upstream) = trung bình nhân của AUC và ACC.

### 3.7 Huấn luyện

#### a) Đường upstream (`GRKT/Runner.py` + `BaseModel.fit`)

```
Runner.run
 ├─ shuffle toàn bộ data (seed 0)
 ├─ chia k_exps=5 chunk bằng nhau  → mỗi lần lấy 1 chunk làm test, 4 chunk còn lại làm train
 ├─ split_ratio=0.9: train_set → 90% train + 10% valid (valid = các cửa sổ CUỐI)
 ├─ model.fit: tối đa 500 epoch, batch 128, Adam 1e-3, early_stop=20
 │    mỗi epoch: train_an_epoch (shuffle batch) → eval valid → giữ bản tốt nhất theo geometric mean(AUC, ACC)
 └─ model.test(test_set) → in AUC/ACC từng fold + trung bình
```

⚠️ Ba điểm rò rỉ/không nhất quán của đường này:

1. **k-fold trên *cửa sổ*, không trên người học** → cùng một người học xuất hiện ở train và test (các cửa sổ khác nhau).
2. **Valid là 10% cửa sổ cuối của train_set** → cùng người học với train, dùng để chọn model.
3. `rel_map`/`pre_map` trong `assist09.bin` được đếm trên **toàn bộ** người học, gồm cả test.

`GRKT/Training.py` đã ghi cảnh báo này ngay đầu file: số liệu từ CLI này **không dùng cho Cấp độ 2**.

#### b) Đường chính thức của repo (`level2/grkt_api.py`)

| Khác biệt | Upstream | Level 2 |
|---|---|---|
| Split | k-fold cửa sổ | split theo **người học** (dùng lại `split.json` của Cấp độ 1, seed 42) |
| Valid | 10% cửa sổ cuối | 20% người học của Cấp độ 1 |
| Đồ thị | đếm trên toàn bộ dữ liệu | chỉ train (`graphs_train_only.npz`) |
| Thời gian | `order_id` thô | phút cộng dồn, cap 240 phút/bước |
| Chọn model | geometric mean(AUC, ACC) | **val AUC toàn cục** |
| Epoch / patience | 500 / 20 | 5 / 3 (`EPOCHS`/`PATIENCE` trong notebook; `run_dataset` mặc định 8 epoch) |
| Batch | 128 | 32 (`BATCH_SIZE` trong notebook; `make_args` mặc định 64) |
| Lưu | chỉ khi xong | checkpoint giữa epoch + resume (`ckpt_GRKT-*.pt`), fingerprint cấu hình |
| `thresh` | 0.6 | 0.5 trên ma trận đã 0/1 |

`train_grkt` (dòng 252) tự viết vòng epoch: shuffle bằng `RandomState` có seed, lưu `order` + `batch_pos` để **resume đúng batch**, ghi checkpoint mỗi `ckpt_every_s`, dừng chủ động theo `deadline` và raise `TrainingPaused`.

### 3.8 Review GRKT — các điểm cần biết

| # | Vị trí | Vấn đề | Mức | Hệ quả |
|---|---|---|---|---|
| 1 | `GRKT/Runner.py:36-52` | k-fold trên cửa sổ, valid lấy từ train | **cao (nếu báo cáo)** | Rò rỉ người học → AUC bị thổi phồng; repo đã thay bằng `level2` |
| 2 | `GRKT/data/assist09/assist09_preprocess.py:58-92` | `rel_map`/`pre_map` đếm trên toàn bộ dữ liệu | **cao (nếu báo cáo)** | Rò rỉ cấu trúc đồ thị từ test; `level2` khắc phục |
| 3 | `assist09_preprocess.py:40` + `GRKT.py:502-505` | `times = order_id` (chênh lệch ~1 giữa 2 dòng) trong khi `alpha=0.01` | trung bình | `1 − exp(−(n+1)·1·0.01) ≈ 0.01` → pha "học trễ" gần như **không hoạt động**; phải dùng Δt theo phút (Level 2) |
| 4 | `GRKT.py:508-509` | Cổng quên `(total_learn == 0).all(-1, True)` — so sánh **chính xác bằng 0** sau ReLU | trung bình | Rất hiếm khi đúng → quên gần như không kích hoạt; đổi sang ngưỡng ε sẽ đổi hành vi mạnh, cần thận trọng khi "sửa" |
| 5 | `GRKT.py:472` | `scatter(-2, total_know_index, learn_3)` với `2K` chỉ số có thể **trùng** (câu hiện tại và câu kế tiếp cùng KC, K=1 → xảy ra thường xuyên) | trung bình | Ghi đè chứ không cộng dồn (last-wins, thứ tự ghi không được đảm bảo trên CUDA); vector học của câu trước bị mất khi trùng KC |
| 6 | `GRKT.py:217-513` | Không dùng `pack_padded_sequence`; mọi bước padding vẫn chạy đủ 4 pha | thấp (hiệu năng) | Với chuỗi ngắn trong batch dài, chi phí gần như không giảm |
| 7 | `GRKT.py:223-224` | `prob_embedding[0] = 0` → tại bước padding `alpha = σ(0) = 0.5`, không phải 0 | thấp | Ảnh hưởng lên `h_tilde` ở bước padding (bị `filt` loại khi chấm), nhưng vẫn tốn tính toán |
| 8 | `GRKT.py:77` | `rel_matrix` được khai báo nhưng **không dùng** trong `forward` | thấp | ~`DH²` tham số chết (16k với DH=128) |
| 9 | `GRKT.py:53,69` + `Positive_Linear` | `pos_mode` không được dùng; mọi `Positive_Linear` hard-code `'softmax'` | thấp | Cờ CLI vô tác dụng |
| 10 | `GRKT/BaseModel.py:53` | `lags` được dựng nhưng `GRKT.forward` không đọc | thấp | Tốn bộ nhớ; `ms_first_response` hiện chưa dùng |
| 11 | `GRKT/BaseModel.py:151-152` | `best_model = deepcopy(self)` **trước khi train** | trung bình | Nếu không epoch nào cải thiện, "best" là model **chưa train** → test bằng khởi tạo ngẫu nhiên |
| 12 | `GRKT/BaseModel.py:175` | `stop_counter == early_stop` (so sánh bằng) | thấp | Chỉ đúng khi tăng đúng 1 mỗi epoch (hiện đúng), nhưng dễ vỡ nếu sửa luồng |
| 13 | `GRKT/BaseModel.py:49-53` | Tensor tạo theo `self.args.device` lúc gọi | thấp | Lệch device nếu model đã chuyển; `level2` bọc bằng `_to_device_feed` |
| 14 | `assist09_preprocess.py:97-99` | Cắt cửa sổ 100, bỏ đuôi <10 | trung bình (giao thức) | Mẫu thứ 2 trở đi **mất lịch sử**; không tương đương "một người = một chuỗi" |
| 15 | `GRKT.py:497` | `history_gain = (h − h_init).clamp(0)` — chỉ quên phần gain dương | — | Chủ ý của mô hình: không làm suy giảm "tri thức nền" |

---

## 4. So sánh GKT vs GRKT

| Khía cạnh | GKT (2019) | GRKT (2024) |
|---|---|---|
| Đơn vị ký ức | `h_t[c] ∈ R^H` cho *mọi* khái niệm | `h[c] ∈ R^KH` mastery, có "đếm số lần luyện" |
| Cơ chế cập nhật | **GNN + GRU** (học cách trộn) | **Retrieval → gain/loss → học trễ/quên theo Δt** (công thức tường minh) |
| Input mỗi bước | `(skill, đáp án)` + skill id | **problem id**, skill id, **thời gian** |
| Đồ thị | 1 quan hệ (`A`), có thể học (PAM/MHA/VAE) | 2 quan hệ (`rel`, `pre`) + `sub = preᵀ`, **cố định** (mask 0/1) |
| Trọng số cạnh | `A[i,j]` trực tiếp | `beta ⊙ mask / bậc` — học độ tương đồng rồi mới áp mask |
| Thời gian | không dùng | trung tâm: `exp(−(n+1)·Δt·k)` cho cả học và quên |
| Dự đoán | `y = σ(W h_{t+1})`, chọn câu kế tiếp; input đã chứa `r_t` | `σ(master − diff)` cho **câu hiện tại**, chưa thấy `r_i` |
| Loss | NLL 2 lớp (tương đương BCE) + VAE (tuỳ chọn) | BCE |
| Chuẩn hoá | BatchNorm trong MLP, dropout/GRU | **không** dropout/norm; mọi ma trận dương |
| Chi phí | nhẹ | nặng hơn nhiều (nhiều ma trận `NK×NK`, `B×NK×KH`) |
| Điểm mạnh | đơn giản, ổn định, dễ mở rộng đồ thị học được | mô hình hoá "quên", "học trễ", "quan hệ tiên quyết" — diễn giải được |
| Điểm yếu | không có khái niệm thời gian; trạng thái bị GRU "trộn" khó giải thích | nhiều siêu tham số; phụ thuộc mạnh vào chất lượng `rel/pre`; nhạy với thang thời gian |

**Khi nào dùng cái nào (theo số liệu trong repo)**: trên ASSIST09 (mỗi câu 1 KC), DKT vẫn mạnh nhất (AUC 0.7866), GKT-Transition 0.7443 — nghĩa là **đồ thị chưa chứng minh được giá trị** ở mức trung bình (xem README, mục "Ba kết luận trung thực"). GRKT được chọn cho Cấp độ 2 vì nó cho **quỹ đạo mastery** và hai loại quan hệ, phục vụ phân tích nâng cao — không phải vì nó chắc chắn thắng AUC.

---

## 5. Ví dụ chạy tay (một người học, C = 4)

Giả sử chuỗi: `q = [0, 2, 1]`, `r = [1, 0, 1]`.

### GKT

```
features = [0*2+1, 2*2+0, 1*2+1] = [1, 4, 3]
questions = [0, 2, 1]
answers   = [1, 0, 1]

h0 = 0  (4 × 32)
i=0: input (q=0, r=1) → h1 → y1 ∈ R^4 → pred = y1[2]        (dự đoán câu 2)
i=1: input (q=2, r=0) → h2 → y2 → pred = y2[1]              (dự đoán câu 1)
i=2: input (q=1, r=1) → h3 → không dự đoán (hết chuỗi)

loss so sánh pred[0] với r[1]=0, pred[1] với r[2]=1
```

### GRKT

```
knows = [[1], [3], [2]]   (1-based)
probs, times do dữ liệu cung cấp

bước 0: α cần KC {1}; retrieval trên h0 → score0 (dự đoán câu 0, =1) → dùng r=1
        gain cho KC 1 + lan sang láng giềng theo rel/pre/sub → h cập nhật
        transition: Δt = t1 - t0; quyết định Gumbel; h += (1-e_learn)·ΔL; quên phần gain
bước 1: score1 (dự đoán câu 1, =0) → nếu sai: h -= Λ (loss lan truyền)
bước 2: score2 (dự đoán câu 2, =1); không có transition vì i = S-1
scores[1:] so với corrs[1:]  → 2 điểm dự đoán được chấm
```

---

## 6. Lệnh chạy và cấu hình

### GKT (upstream)

```bash
cd GKT
python train.py --data-file=skill_builder_data.csv --model=GKT --graph-type=Dense
python train.py --data-file=skill_builder_data.csv --model=DKT
```

### GKT qua pipeline của repo (khuyến nghị — AUC toàn cục, split cố định)

```bash
python -c "from level1.pipeline import run_level1; run_level1(data_file='assistment_test15.csv', epochs=2)"
```

### GRKT (upstream, chỉ để tái lập — có rò rỉ)

```bash
cd GRKT
python Training.py --model=GRKT --dataset=assist09
```

### GRKT (Level 2, giao thức dùng cho báo cáo)

```bash
python -c "from level2.experiment import run_dataset; run_dataset('assist2009_smoke', save_root='outputs/level2', epochs=1, batch_size=16, max_seq_len=50, variants=('train',), run_baselines=False, record_trajectory=True, run_calibration=True)"
```

Bảng cấu hình mặc định:

| Tham số | GKT (`train.py`) | GRKT upstream | GRKT Level 2 |
|---|---|---|---|
| `epochs` | 50 | 500 | 5 (`QUICK_RUN=False`), 2 khi smoke |
| `batch_size` | 128 | 128 | 32 (16 khi smoke) |
| `lr` | 1e-3 | 1e-3 | 1e-3 |
| `hidden` | 32 (H, E) | 128 (DH), 16 (KH) | 128, 16 |
| `k_hop` | — | 1 | 1 |
| `alpha` (cờ học/quên) | — | 0.01 | 0.01 |
| `tau` (Gumbel) | 0.5 (VAE) | 0.2 | 0.2 |
| `thresh` (đồ thị) | — | 0.6 | 0.5 trên ma trận 0/1 |
| `max_seq_len` | 200 | 100 (cửa sổ) | 200 (50 khi smoke) |
| split | 0.6/0.2/0.2 theo người | k-fold cửa sổ | theo người, seed 42 |
| chọn model | val loss | geometric mean(AUC, ACC) | val AUC toàn cục |

---

## 7. Checklist khi đọc/sửa code

1. **GKT**: mọi thứ đều 0-based, padding `-1`; đừng quên `features = skill*2 + correct` (không phải `skill`).
2. **GKT**: `pred_res` dài `S−1` — nhãn đã dịch 1 bước. Muốn "dự đoán câu t từ lịch sử < t" thì phải tự dời vị trí.
3. **GKT**: đồ thị tĩnh có `requires_grad=False`; chỉ `PAM`/`MHA`/`VAE` mới học.
4. **GRKT**: mọi id **1-based**, index 0 = padding; `knows` là *list* (đa KC), repo này `K=1`.
5. **GRKT**: `score_i` dự đoán câu `i`, nên `scores[:,1:]` mới khớp `corrs[:,1:]`.
6. **GRKT**: `rel_map`/`pre_map` phải là 0/1 khi đưa vào (hoặc `thresh` đúng); `thresh=0` biến thành đồ thị đầy đủ.
7. **GRKT**: đơn vị thời gian quyết định hành vi học/quên — dùng phút, không dùng `order_id`/ms thô.
8. **Khi báo cáo**: không dùng số từ `GRKT/Training.py` (rò rỉ) và `GKT/outputs/*.csv` (smoke 15 học sinh).

---

## 8. Thuật ngữ

| Thuật ngữ | Nghĩa trong tài liệu |
|---|---|
| KC / khái niệm | knowledge concept — `skill_id` |
| mastery | mức độ thành thạo một KC (scalar hoặc vector `KH`) |
| retrieval | truy hồi/khôi phục trạng thái liên quan tới câu hỏi hiện tại |
| gain / loss | lượng tri thức tăng khi đúng / giảm khi sai |
| strengthening | bước củng cố sau khi trả lời |
| transition | bước chuyển giữa hai lần làm bài (học trễ + quên) |
| `rel_map` / `pre_map` | đồ thị đồng xuất hiện / tiên quyết |
| Gumbel-Softmax | lấy mẫu rời rạc có đạo hàm (dùng cho quyết định học và loại cạnh VAE) |
