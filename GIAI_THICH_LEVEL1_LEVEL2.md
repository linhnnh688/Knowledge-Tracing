# Giải thích `level1/` và `level2/` — hai module đưa GKT/GRKT lên Kaggle

Tài liệu này là phần tiếp theo của [GIAI_THICH_GKT_GRKT.md](file:///D:/workspace/Knowledge-Tracing/GIAI_THICH_GKT_GRKT.md)
(giải thích thuật toán GKT/GRKT). Ở đây chỉ trả lời một câu: **dữ liệu chạy qua hai package
này theo đường nào, mỗi bước sinh ra cái gì, và chỗ nào có thể sai**.

Mục tiêu đọc xong:

* biết notebook Kaggle gọi hàm nào, hàm đó nằm ở file/dòng nào;
* biết shape và ý nghĩa của dữ liệu ở từng chặng (CSV → loader → chuỗi GRKT → batch);
* biết artifact nào được ghi ra, artifact nào chỉ là bằng chứng;
* biết cơ chế checkpoint/resume và các hàng rào chống trộn hai lần chạy.

---

## 0. Bản đồ file và đường gọi

### 0.1 Vai trò từng file

| File | Dòng | Vai trò |
|---|---:|---|
| [level1/paths.py](file:///D:/workspace/Knowledge-Tracing/level1/paths.py#L1) | 14 | `repo_root()` + `ensure_gkt_on_path()` — chèn `GKT/` vào `sys.path` để import `processing`, `models`, `metrics` của upstream |
| [level1/pipeline.py](file:///D:/workspace/Knowledge-Tracing/level1/pipeline.py#L1) | 426 | Toàn bộ API Cấp độ 1: nạp dữ liệu, chia split, dựng model, train, evaluate, lưu checkpoint |
| [level1/explain.py](file:///D:/workspace/Knowledge-Tracing/level1/explain.py#L1) | 426 | Chọn 5 case đúng + 5 case sai và sinh bằng chứng phản chứng (flip một lượt, bỏ toàn bộ cạnh) |
| [level1/dashboard.py](file:///D:/workspace/Knowledge-Tracing/level1/dashboard.py#L1) | 130 | Xuất `learner_view.json` + widget chọn người học; lưu `state_norm` theo từng khái niệm |
| `level1/gikt.py`, `level1/gikt_data.py` | 75 + 149 | Bản port GIKT cũ; **không nằm trên đường chạy** (`run_gikt_experiment` raise `NotImplementedError`) |
| [level2/dataset_spec.py](file:///D:/workspace/Knowledge-Tracing/level2/dataset_spec.py#L1) | 840 | Hợp đồng CSV chuẩn hoá, 3 spec dữ liệu, chuẩn hoá Junyi theo mảnh, dò file Kaggle |
| [level2/grkt_data.py](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L1) | 512 | Đổi bundle Cấp độ 1 thành chuỗi GRKT 1-based, đối chiếu từng người học, đếm `rel/pre` chỉ từ train |
| [level2/grkt_api.py](file:///D:/workspace/Knowledge-Tracing/level2/grkt_api.py#L1) | 684 | `make_args` / `train_grkt` / `evaluate_grkt` / `run_grkt_experiment` — thay `GRKT/Runner.py` |
| [level2/baselines.py](file:///D:/workspace/Knowledge-Tracing/level2/baselines.py#L1) | 512 | Nạp checkpoint DKT/GKT Cấp độ 1 (hoặc train lại có resume) trên **cùng split** |
| [level2/trajectory.py](file:///D:/workspace/Knowledge-Tracing/level2/trajectory.py#L1) | 564 | Ghi `m_pre/m_tilde/m_strong`, 4 tiêu chí hợp lý, xuất npz/CSV/PNG |
| [level2/calibration.py](file:///D:/workspace/Knowledge-Tracing/level2/calibration.py#L1) | 161 | ECE / ACE / Brier + reliability diagram |
| [level2/archive.py](file:///D:/workspace/Knowledge-Tracing/level2/archive.py#L1) | 88 | Đóng gói `level2_out/` thành zip để tải về và resume phiên sau |
| [level2/experiment.py](file:///D:/workspace/Knowledge-Tracing/level2/experiment.py#L1) | 557 | `run_dataset` (B1→B8) và `run_level2` (nhiều tập, một deadline chung) |
| [kaggle_level1.ipynb](file:///D:/workspace/Knowledge-Tracing/kaggle_level1.ipynb) | 27 cell | Config + gọi package; không chứa logic |
| [kaggle_level2.ipynb](file:///D:/workspace/Knowledge-Tracing/kaggle_level2.ipynb) | 32 cell | Config + resume zip + gọi `run_level2` + đóng gói |

### 0.2 Sơ đồ gọi tổng quát

```
kaggle_level1.ipynb
├─ level1.pipeline.load_kt_data ──► GKT/processing.load_dataset
│                                   ├─ KTDataset + random_split(seed=42)
│                                   ├─ DataLoader(pad_collate, pad=-1)
│                                   └─ build_transition_graph(train indices only)
├─ level1.pipeline.save_split
├─ level1.pipeline.run_experiment ──► build_model ──► GKT/models.{DKT,GKT}
│                                 └─ train_one ──► _epoch_loop ──► GKT/metrics.KTLoss
├─ level1.explain.collect_cases
└─ level1.dashboard.export_learner_view

kaggle_level2.ipynb
├─ level2.dataset_spec.ensure_canonical ──► <dataset>_canonical.csv + .meta.json
├─ level2.experiment.run_level2
│   └─ run_dataset × từng dataset
│       B1 ensure_canonical + level1.pipeline.load_kt_data + save_split
│       B2 grkt_data.build_grkt_data + split_sequences
│       B3 build_relevance_maps + graph_variant + leak_report
│       B4 grkt_api.run_grkt_experiment ──► GRKT/model/GRKT.py
│       B5 baselines.level1_baseline ──► level1.pipeline.build_model/_epoch_loop
│       B6 trajectory.record_trajectories + 4 tiêu chí
│       B7 calibration (tuỳ chọn)
│       B8 multi-seed (tuỳ chọn)
└─ level2.archive.package_output ──► level2_output.zip
```

### 0.3 Nguyên tắc thiết kế: notebook mỏng, package dày

Notebook chỉ giữ **config + thứ tự gọi + hiển thị**. Mọi phép tính nằm trong package để:

1. chạy lại được ngoài Kaggle (local/CPU) mà không cần notebook;
2. test được từng hàm;
3. notebook không phải là "nguồn sự thật" duy nhất của thực nghiệm.

Hệ quả cần nhớ: **giá trị mặc định trong package không phải giá trị notebook dùng**. Ví dụ
`run_dataset(..., max_seq_len=100)` nhưng notebook Cấp độ 2 truyền `MAX_SEQ_LEN=200`. Khi đọc
kết quả, phải nhìn config của notebook, không nhìn default của hàm.

---

## 1. Cách hai package bắc cầu sang upstream

### 1.1 `GKT/` được import bằng `sys.path`

`GKT/` là mã upstream dùng import tuyệt đối (`from processing import ...`,
`from models import DKT, GKT`, `from metrics import KTLoss`). Vì vậy
[level1/paths.py:9](file:///D:/workspace/Knowledge-Tracing/level1/paths.py#L9) chèn thẳng thư mục
`<repo>/GKT` vào đầu `sys.path`:

```python
sys.path.insert(0, str(repo_root() / "GKT"))
```

Sau đó `level1.pipeline` mới import được `processing`, `models`, `metrics`. Notebook Cấp độ 1
làm lại thao tác này bằng tay (cell 0) để các cell sau import trực tiếp cũng chạy.

**Rủi ro:** `models`, `processing`, `metrics` trở thành tên module **toàn cục**. Nếu về sau có
package khác cũng đặt tên `models.py` và được import trước, `import models` sẽ lấy nhầm file.
Đây là lý do `GRKT/` (được viết lại) dùng import tương đối, còn `GKT/` vẫn giữ import tuyệt đối.

### 1.2 `GRKT/` được import như package

`level2/grkt_api.py:36` thêm **repo root** (không phải `GRKT/`) vào `sys.path`, rồi
`import GRKT` bình thường:

```python
from GRKT.model.GRKT import GRKT
```

Vì `GRKT/` đã được sửa sang import tương đối, nó không đụng tên module với `GKT/`.

### 1.3 Hệ quả về không gian chỉ số

| | Cấp độ 1 (`GKT/`) | Cấp độ 2 (`GRKT/`) |
|---|---|---|
| Khái niệm | 0-based, `0..K-1` | 1-based, `0` = padding |
| Câu hỏi | không dùng embedding câu hỏi | 1-based, `0` = padding |
| Feature | `skill * 2 + correct` | tách `knows` + `corrs` |
| Đồ thị | ma trận `K×K`, hàng chuẩn hoá (Transition) | `rel_map`/`pre_map` 0/1, kích thước `(K+1)²` |

`level2/grkt_data.py` chịu trách nhiệm dịch giữa hai không gian này, và `_assert_align`
([grkt_data.py:297](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L297)) kiểm tra
từng người học để chắc chắn phép dịch không làm lệch chuỗi.

---

## 2. Cấp độ 1 — flow chi tiết

### 2.1 Đầu vào và đầu ra

| Loại | Nội dung |
|---|---|
| Dữ liệu | `GKT/data/skill_builder_data.csv` (525.534 dòng / 4.217 người / 123 khái niệm / 26.688 câu hỏi) hoặc `assistment_test15.csv` (1.888 dòng / 15 người) khi `QUICK_RUN=True` |
| Cấu hình | §0.2 của notebook: `EPOCHS`, `BATCH_SIZE`, `MAX_SEQ_LEN`, `SAVE_DIR` |
| Sản phẩm | Dưới `outputs/level1/` (local) hoặc `/kaggle/working/level1_out/` (Kaggle) |

Cấu hình thực tế của `kaggle_level1.ipynb`:

| Tham số | `QUICK_RUN=False` | `QUICK_RUN=True` |
|---|---:|---:|
| `DATA_FILE` | `skill_builder_data.csv` | `assistment_test15.csv` |
| `EPOCHS` | 20 | 8 |
| `BATCH_SIZE` | 32 | 16 |
| `MAX_SEQ_LEN` | 200 | 80 |
| `hid_dim` / `emb_dim` | 32 / 32 | 32 / 32 |
| `lr` | 1e-3 | 1e-3 |
| `seed` (split + khởi tạo) | 42 | 42 |

### 2.2 Bản đồ cell → hàm → sản phẩm

| Cell | Việc làm | Hàm chính | Sản phẩm |
|---|---|---|---|
| 0 | clone repo, `sys.path` | — | — |
| 3.1 | nạp dữ liệu + chia split | `load_kt_data`, `save_split` | `split.json` |
| 3.2 | baseline DKT | `run_experiment(model_type="DKT")` | `DKT.pt` |
| 3.3 | GKT-Transition | `run_experiment(graph_type="Transition")` | `GKT-Transition.pt` |
| 3.4 | ablation | `load_kt_data` + `run_experiment` cho `Dense`, `Identity`, `Zero`, `Shuffled` | `GKT-*.pt` |
| 4.1 | bảng tổng hợp | — | `metrics.csv`, `metrics.json` |
| 4.5 | đa seed | `run_experiment` × 5 cấu hình × 3 seed | `multiseed_metrics.csv` |
| 5 | 10 case + bằng chứng | `collect_cases`, `format_cases` | `cases.md` |
| 5b | dashboard | `export_learner_view` | `learner_view.json` |
| 6 | lưu bảng | — | `metrics.csv/json` |

### 2.3 Bước 3.1 — nạp dữ liệu và chia split

`load_kt_data` ([pipeline.py:40](file:///D:/workspace/Knowledge-Tracing/level1/pipeline.py#L40))
nhận đường dẫn tương đối (tự dò trong `GKT/data/`) rồi gọi upstream
`processing.load_dataset`. Thứ tự upstream làm:

1. bỏ dòng thiếu `skill_id`;
2. bỏ người học có ít hơn `min_seq_len=3` tương tác;
3. `pd.factorize(skill_id, sort=True)` → cột `skill` 0-based;
4. `feature = skill * 2 + correct` (nhị phân);
5. `groupby("user_id", sort=False)` → mỗi người một chuỗi, cắt còn `max_seq_len`;
6. `KTDataset` + `random_split(seed=42, 0.6/0.2/0.2)` → **chia theo người học**;
7. tạo 3 `DataLoader`; `pad_collate` đệm `-1` cho feature/question/answer;
8. **sau khi chia** mới dựng đồ thị; `Transition` và `Shuffled` chỉ dùng
   `train_dataset.indices`.

Điểm 8 quan trọng: đồ thị Cấp độ 1 **không nhìn thấy người học test**. Đây là điều README ghi
"Transition (train only)".

Bundle trả về ([pipeline.py:71](file:///D:/workspace/Knowledge-Tracing/level1/pipeline.py#L71)):

```python
{
  "concept_num": 123,
  "graph": tensor [K, K] hoặc None,
  "train_loader": DataLoader,
  "val_loader": DataLoader,
  "test_loader": DataLoader,
  "data_path": "/.../skill_builder_data.csv",
}
```

Shape một batch:

| Tensor | Shape | Ý nghĩa |
|---|---|---|
| `features` | `[B, S]` | `skill*2 + correct`, đệm `-1` |
| `questions` | `[B, S]` | chỉ số khái niệm, đệm `-1` |
| `answers` | `[B, S]` | nhãn 0/1, đệm `-1` |

Với model, `pred` có shape `[B, S-1]`: dự đoán cho vị trí `1..S-1` từ lịch sử `0..S-2`.

`save_split` ([pipeline.py:94](file:///D:/workspace/Knowledge-Tracing/level1/pipeline.py#L94))
ghi ba danh sách `train_users/val_users/test_users`. **File này chỉ là bằng chứng/tài liệu**:
Cấp độ 2 không đọc lại nó, mà chia lại bằng đúng seed và đúng tham số (xem §3.3).

### 2.4 Bước 3.2/3.3 — huấn luyện DKT và GKT

`build_model` ([pipeline.py:105](file:///D:/workspace/Knowledge-Tracing/level1/pipeline.py#L105))
có hai nhánh:

```python
DKT(2 * concept_num, hid_dim, concept_num, ...)
GKT(concept_num, hid_dim, emb_dim, edge_types=2, graph_type,
    graph=graph, ..., binary=True, has_cuda=...)
```

`binary=True` là **sửa so với upstream**. `GKT/models.py` khai báo `binary=False` mặc định, kéo
theo `res_len = 12` và one-hot `12·concept_num` chiều; trong khi dữ liệu lại được mã hoá
`skill*2 + correct` (chỉ 2 giá trị mỗi khái niệm). `GKT/train.py:128` gọi constructor mà không
truyền `binary`, nên đường train gốc dùng sai số chiều feature. Cấp độ 1 truyền `binary=True`
để khớp đúng `res_len = 2`.

`train_one` ([pipeline.py:184](file:///D:/workspace/Knowledge-Tracing/level1/pipeline.py#L184)):

| Thành phần | Giá trị |
|---|---|
| Optimizer | `Adam(lr=1e-3)` |
| Loss | `KTLoss` = NLL có `ignore_index=-1` (bỏ vị trí đệm) |
| Gradient clipping | `clip_grad_norm_(..., 10.0)` |
| Early stop | theo `val AUC`, `patience=5` |
| Lưu tốt nhất | snapshot `state_dict` về CPU, nạp lại cuối train |
| Metric | **AUC toàn cục** trên cả loader, không phải trung bình AUC từng batch |

Mask tính metric ([pipeline.py:160](file:///D:/workspace/Knowledge-Tracing/level1/pipeline.py#L160)):

```python
real = answers[:, 1:]
mask = (real != -1) & (questions[:, 1:] != -1)
```

Tức là chỉ tính trên vị trí có nhãn **và** có câu hỏi thật. Cùng mask này được dùng lại ở
Cấp độ 2 (baselines và calibration), nên ECE/Brier của hai cấp so sánh được.

`run_experiment` ([pipeline.py:249](file:///D:/workspace/Knowledge-Tracing/level1/pipeline.py#L249))
trả về dict:

```python
{
  "name": "DKT" | "GKT-Transition",
  "model_type": "DKT" | "GKT",
  "graph_type": "N/A" | "Transition" | ...,
  "concept_num": int,
  "best_val_auc": float,
  "test_loss": float, "test_auc": float, "test_acc": float,
  "history": {train_loss, val_loss, train_auc, val_auc, train_acc, val_acc},
  "seed": int,
  "ckpt": str,          # nếu save_dir được truyền
  "model": model.cpu(), # giữ lại cho explain/dashboard
  "bundle": bundle,
}
```

Bộ khoá này được Cấp độ 2 cố ý lặp lại trong `run_grkt_experiment`, để bảng so sánh chỉ cần
nối hai danh sách kết quả.

### 2.5 Bước 3.4 — ablation cấu trúc đồ thị

| `graph_type` | Cách dựng | Câu hỏi nó trả lời |
|---|---|---|
| `Transition` | đếm cạnh `(c_i → c_{i+1})` trên train, chuẩn hoá hàng | cấu trúc chuyển tiếp có giúp không |
| `Dense` | mọi cặp nối nhau | "có cạnh" nói chung có giúp không |
| `Identity` | chỉ self-loop | đồ thị có tác dụng ngoài việc giữ trạng thái không |
| `Zero` | không cạnh | mô hình có đồ thị nhưng không lan truyền |
| `Shuffled` | giữ số cạnh của Transition, xáo đích | cấu trúc **đúng** có quan trọng hay chỉ cần mật độ cạnh |

Notebook chạy cả bốn biến thể; `run_level1` trong package chỉ chạy `Transition`, `Dense`,
`Identity`, `Zero` — **không có `Shuffled`**. Nếu dùng `run_level1` để tái lập notebook, kết quả
sẽ thiếu một dòng.

### 2.6 Bước 4.5 — đa seed

Notebook đặt `RUN_MULTI_SEED = True` và chạy `seeds = [42, 123, 456]`. Cách làm:

* bundle được nạp **một lần** với `seed=42` và tái sử dụng → split không đổi;
* `run_experiment(..., seed=s, bundle=bundle_fixed)` chỉ đổi khởi tạo trọng số và thứ tự batch;
* 5 cấu hình (DKT, GKT-Transition, GKT-Identity, GKT-Zero, GKT-Shuffled) × 3 seed = **15 lần
  train**, cộng 6 lần train ở các bước trước → **21 lần train** trong một lần chạy đầy đủ.

`GKT-Dense` không nằm trong danh sách multi-seed dù đã train ở bước 3.4. Đây là lựa chọn tiết
kiệm thời gian, không phải lỗi, nhưng bảng đa seed sẽ thiếu Dense.

### 2.7 Bước 5 — 10 case và bằng chứng phản chứng

`collect_cases` ([explain.py:191](file:///D:/workspace/Knowledge-Tracing/level1/explain.py#L191)):

1. chạy model trên `test_loader` (không shuffle);
2. gom mọi dự đoán vào hai rổ `correct` / `incorrect`;
3. sắp theo `confidence = |p - 0.5|` giảm dần;
4. chọn `n_each=5` mỗi rổ, ưu tiên khác khái niệm;
5. gắn lịch sử 8 bước gần nhất và 5 láng giềng mạnh nhất trên đồ thị.

`annotate_evidence` ([explain.py:111](file:///D:/workspace/Knowledge-Tracing/level1/explain.py#L111))
chạy lại model để lấy ba loại bằng chứng:

| Bằng chứng | Cách đo | Ý nghĩa |
|---|---|---|
| `p_recheck` | chạy lại đúng batch, so với `p` đã log (dung sai 0.02) | xác nhận phép đo không lỗi |
| `flips` | đảo kết quả một lượt trong quá khứ: `feature = skill*2 + correct` → `skill*2 + (1-correct)` | lượt nào ảnh hưởng mạnh nhất tới dự đoán hiện tại |
| `delta_graph` | tạm gán `model.graph.data = 0` rồi chạy lại, sau đó khôi phục | đồ thị đóng góp bao nhiêu cho dự đoán này |

Ngoài ra `_edge_evidence` lấy cạnh mạnh nhất vào/ra khái niệm đích và ghi người học đã chạm
khái niệm đó lần cuối khi nào, đúng hay sai.

Kết quả ghi vào `cases.md`; lỗi trong bước này chỉ in cảnh báo, không làm chết train.

### 2.8 Bước 5b — dashboard người học

`export_learner_view` ([dashboard.py:22](file:///D:/workspace/Knowledge-Tracing/level1/dashboard.py#L22))
gọi `model(features, questions, return_state=True)`. Với GKT, đầu ra là
`(pred, ec_list, rec_list, z_prob_list, states)`, nên lấy `out[0]` và `out[4]`.

Mỗi bước lưu:

* `skill`, `skill_name`, `y_true`, `y_prob`, `y_pred`;
* `state_norm`: chuẩn L2 theo từng khái niệm của `h_t ∈ [B, C, H]`, tức vector độ lớn trạng
  thái ẩn `[C]` (code gọi là "Frobenius-style per-concept norm").

`learner_view.json` là mảng một dict/người học, mỗi dict có `steps`. Widget `ipywidgets` chỉ là
lớp hiển thị; không có widget thì in người đầu tiên.

### 2.9 Sản phẩm Cấp độ 1

| File | Nội dung |
|---|---|
| `split.json` | danh sách user của 3 tập — bằng chứng chia theo người |
| `metrics.csv` / `metrics.json` | bảng tổng hợp mọi mô hình |
| `DKT.pt`, `GKT-<graph>.pt` | `state_dict` của lần train tương ứng |
| `cases.md` | 10 case + lịch sử + phản chứng + lý giải |
| `learner_view.json` | chuỗi dự đoán + `state_norm` của mọi người test |
| `multiseed_metrics.csv` | mean ± std trên 3 seed (nếu bật) |

---

## 3. Cấp độ 2 — flow chi tiết

### 3.1 Hợp đồng dữ liệu: CSV chuẩn hoá

Mọi tập phải quy về đúng 8 cột
([dataset_spec.py:37](file:///D:/workspace/Knowledge-Tracing/level2/dataset_spec.py#L37)):

| Cột | Ý nghĩa | Ai dùng |
|---|---|---|
| `order_id` | số thứ tự dòng 1..N sau khi sort | không dùng để chia chuỗi; chỉ để đối chiếu |
| `user_id` | người học | chia split, groupby |
| `problem_id` | câu hỏi | GRKT `probs` (1-based) |
| `skill_id` | khái niệm | factorize → `skill` 0-based |
| `skill_name` | tên hiển thị | dashboard/quỹ đạo |
| `correct` | 0/1 | nhãn |
| `ms_first_response` | thời gian làm bài (ms) | `time_mode="duration_ms"` |
| `timestamp_ms` | epoch ms | `time_mode="timestamp_ms"` |

Ba spec:

| Spec | File chuẩn hoá | Thời gian | Lọc tối thiểu | Cap |
|---|---|---|---|---:|
| `assist2009` | `GKT/data/skill_builder_data.csv` (đã có sẵn) | `duration_ms` | 3 | — |
| `assist2009_smoke` | `GKT/data/assistment_test15.csv` | `duration_ms` | 3 | — |
| `junyi` | `junyi_canonical.csv` (sinh ra) | `timestamp_ms` | 10 | 200 khái niệm |

### 3.2 `ensure_canonical` — chuẩn hoá một lần, dùng nhiều lần

[dataset_spec.py:664](file:///D:/workspace/Knowledge-Tracing/level2/dataset_spec.py#L664).
Thuật toán:

1. Xác định `target = out_dir/<canonical_file>`; nếu đã tồn tại thì kiểm `.meta.json`.
2. Nếu chưa có file gốc, tự dò `/kaggle/input` (`find_kaggle_input`).
3. Đọc **header** trước, khớp tên cột: chính xác trước, rồi bỏ hoa/thường/khoảng trắng/gạch
   dưới (`_match_column`). Bốn vai trò bắt buộc: user, problem, timestamp, duration.
4. Quét file theo mảnh `chunksize=2_000_000`; mã hoá id thành số nguyên ngay trong mảnh
   (`_CodeMap`), nên RAM giữ int chứ không giữ chuỗi.
5. Nhãn: lấy cột `correct`/`is_correct`; nếu không có thì suy từ `answer_count` theo quy ước
   pyKT (1 lần → đúng, ≥3 lần → sai, 2 lần → bỏ).
6. Khái niệm: lấy cột `ucid`/`exercise`; nếu log không có cột khái niệm thì nối bảng tra
   (`_load_content_map`), và **bắt buộc phủ ≥ 50%** lượt làm bài, nếu không thì raise.
7. `top_concepts`: giữ N khái niệm hay gặp nhất **trước khi** lọc người học, để số tương tác
   của mỗi người phản ánh dữ liệu còn lại thật.
8. Ghép mảnh → sort `(user_id, timestamp_ms)` → bỏ người dưới `min_interactions` → subsample
   theo seed 42 nếu có `max_users` → cắt `max_keep_per_user=500` → giải mã id → ghi CSV.
9. Ghi `.meta.json` gồm `max_users`, `min_interactions`, `top_concepts`, `raw_file`, số dòng,
   số người, số khái niệm.

#### Hàng rào `.meta.json`

[dataset_spec.py:636](file:///D:/workspace/Knowledge-Tracing/level2/dataset_spec.py#L636).
Nếu lần trước chạy `QUICK_RUN=True` (cap 300 người) rồi lần này chạy full, `ensure_canonical`
sẽ thấy file đã tồn tại và **dùng lại** — kết quả sẽ lặng lẽ là của 300 người. `.meta.json`
biến việc đó thành `RuntimeError` kèm hướng dẫn xoá cả hai file. Nếu `.meta.json` không có
(file cũ), hàm chỉ in cảnh báo vì không thể chứng minh cấu hình.

#### Chi phí Junyi

`Log_Problem.csv` 16,2 triệu dòng, ~3 GB. Chuẩn hoá mất 15–30 phút, đỉnh RAM 2–3 GB; bước
`load_kt_data` đọc lại CSV có thể cần thêm 5–8 GB. `MAX_USERS` chỉ giảm phần train, **không**
giảm thời gian quét file.

### 3.3 B1 — chia split lại y hệt Cấp độ 1

[experiment.py:112](file:///D:/workspace/Knowledge-Tracing/level2/experiment.py#L112).

```python
shared = load_kt_data(canonical, batch_size=..., graph_type="Transition",
                      model_type="GKT", train_ratio=0.6, val_ratio=0.2,
                      seed=42, max_seq_len=200, min_seq_len=min_seq)
split = save_split(shared, save_dir / "split.json")
```

Điểm cần hiểu: Cấp độ 2 **không đọc** `split.json` của Cấp độ 1. Nó chia lại bằng cùng file
chuẩn hoá + cùng `seed=42` + cùng `train_ratio/val_ratio` + cùng `max_seq_len/min_seq_len`.
Vì `random_split` dùng `torch.Generator().manual_seed(seed)`, cùng đầu vào cho cùng split.

Điều kiện để "cùng split" đúng:

* `min_interactions` phải khớp `min_seq_len` của lần nạp;
* `max_seq_len` phải khớp (200 ở cả hai cấp);
* `train_ratio=0.6`, `val_ratio=0.2`;
* cùng file CSV (Junyi phải dùng đúng file đã sinh, không sinh lại với cap khác).

Đây là **hợp đồng ngầm**: nếu ai đổi mặc định `train_ratio` ở notebook Cấp độ 1 mà không đổi
Cấp độ 2, hai bên sẽ chia khác nhau. `_assert_align` chỉ bắt được lệch **chuỗi**, không bắt
được lệch **split**, vì nó so từng người theo đúng thứ tự dữ liệu. Chỗ dễ phát hiện nhất là
đối chiếu DKT: AUC phải tái hiện 0.7866 ± 0.005.

### 3.4 B2 — dựng chuỗi GRKT và đối chiếu

`build_grkt_data`
([grkt_data.py:170](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L170)) đọc lại
CSV chuẩn hoá và lặp lại đúng pipeline upstream:

```python
df = df.dropna(subset=["skill_id"])
df = df.groupby("user_id").filter(lambda q: len(q) >= min_seq_len)
df["skill"], _ = pd.factorize(df["skill_id"], sort=True)
df["prob"], _ = pd.factorize(df["problem_id"], sort=True)
```

Sau đó với từng người:

| Trường | Công thức | Ghi chú |
|---|---|---|
| `knows[i]` | `[skill + 1]` | 1-based; mỗi lượt đúng một khái niệm (`max_concepts=1`) |
| `probs[i]` | `prob + 1` | 1-based |
| `corrs[i]` | 0/1 | nhãn |
| `times[i]` | cộng dồn Δt phút, chặn `gap_cap_min=240` | `times[0]=0` |
| `skills0[i]` | `skill` 0-based | để đối chiếu Cấp độ 1 |
| `lags[i]` | thời gian làm bài (giây) | GRKT hiện không dùng |

`_cumulative_times` ([grkt_data.py:100](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L100))
chặn từng Δt ở 240 phút rồi mới cộng dồn và làm tròn. Nếu dùng timestamp thô, số mũ
`exp(-(n+1)·Δt·k)` underflow và mô hình "quên sạch" sau một bước. Nếu làm tròn từng Δt, mọi
lượt dưới 1 phút thành 0 và giai đoạn học/quên không học được gì.

`_assert_align`
([grkt_data.py:297](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L297)) so ba thứ
với bundle Cấp độ 1 tại cùng vị trí:

* `user_id`;
* chuỗi `skills0` với `questions` của Cấp độ 1;
* chuỗi `corrs` với `answers` của Cấp độ 1.

Cộng thêm hai assert cuối hàm: `derived_num == concept_num` và `positions == len(base_users)`.
Đây là lý do Cấp độ 2 không thể "lặng lẽ" dùng một cách tiền xử lý khác.

`split_sequences`
([grkt_data.py:461](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L461)) đổi danh
sách user của split thành ba bộ `{user_ids, sequences, items}`.

Cấu trúc dữ liệu:

```
GRKTData
├─ dataset, concept_num, n_probs, skill_lookup, time_mode, gap_cap_min, stats
└─ sequences: List[UserSequence]
             ├─ user_id
             ├─ probs   List[int]   [S]
             ├─ knows   List[[int]] [S][1]
             ├─ corrs   List[int]   [S]
             ├─ times   List[int]   [S]
             ├─ lags    List[int]   [S]
             └─ skills0 List[int]   [S]
```

`as_grkt_items()` đổi mỗi người thành `[(prob, [knows], corr, time, lag), ...]` đúng định dạng
upstream, nên phần forward của GRKT không phải sửa.

### 3.5 B3 — dựng đồ thị chỉ từ train

`build_relevance_maps`
([grkt_data.py:328](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L328)) đếm bốn
bộ đếm trên **một người học** rồi cộng dồn:

| Bộ đếm | Nghĩa |
|---|---|
| `tt[c, d]` | làm `c` đúng, sau đó làm `d` đúng |
| `tf[c, d]` | `c` đúng → `d` sai |
| `ft[c, d]` | `c` sai → `d` đúng |
| `ff[c, d]` | `c` sai → `d` sai |

Công thức:

```text
rel = (tt + ff) / (tt + ff + tf + ft)      # đồng xuất hiện cùng kết quả
pre = ft / (tf + ft)                       # tiên quyết: sai c rồi đúng d
```

Lọc thưa: `rel` chỉ giữ cặp có tổng ≥ `4·cold_thresh`, `pre` chỉ giữ cặp có `tf+ft` ≥
`2·cold_thresh`. Với `cold_thresh=5`, ngưỡng là 20 và 10 lượt. Hàng/cột 0 (padding) bị đặt 0.

`graph_variant` ([grkt_data.py:407](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L407)):

| Biến thể | Cách sinh | Số cạnh |
|---|---|---|
| `train` | `binarize(rel/pre, graph_thresh=0.6)` | phụ thuộc dữ liệu |
| `dense` | mọi cặp, trừ padding và self | `(K)(K-1)` |
| `zero` | toàn 0 | 0 |
| `shuffled` | giữ số cạnh, hoán vị đích | bằng `train` |

Hai ngưỡng khác nhau, đừng lẫn:

* `graph_thresh=0.6` — ngưỡng **nhị phân hoá** tỉ lệ `rel`/`pre` khi dựng đồ thị;
* `bin_thresh=0.5` → `args.thresh` — chỉ để GRKT giữ nguyên ma trận 0/1 đã có. Không đặt 0 vì
  nhánh `thresh=0` của upstream gán toàn bộ ma trận bằng 1, kể cả padding.

`adjacency_from_graphs`
([grkt_data.py:449](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L449)) tạo mặt
nạ láng giềng 1-hop dùng cho tiêu chí quỹ đạo:

```text
adj = (rel > 0) | (pre > 0) | (preᵀ > 0)
```

`preᵀ` là vì `pre[c, d]` nghĩa là `c` là tiên quyết của `d`; khi đo "láng giềng của khái niệm
đang làm", cần cả hai chiều.

`leak_report` ([grkt_data.py:471](file:///D:/workspace/Knowledge-Tracing/level2/grkt_data.py#L471))
in và ghi `leak_report.txt`:

* số người train/val/test;
* giao của ba tập (phải bằng 0);
* số tương tác từng tập;
* số cạnh dựng từ train và số cạnh **nếu** dựng từ val (khác nhau chứng tỏ đồ thị phụ thuộc
  tập huấn luyện);
* số cạnh từng biến thể;
* khẳng định không dùng nhãn tương lai: chỉ `knows[i+1]` và `times[i+1]`.

Báo cáo này **in số, không assert giao bằng 0**. Giao khác 0 sẽ lộ ra ngay trên màn hình,
nhưng không làm dừng chương trình. Nguồn bảo đảm thật sự là cách chia: mỗi người chỉ nằm trong
một tập.

### 3.6 B4 — huấn luyện GRKT

`run_grkt_experiment`
([grkt_api.py:474](file:///D:/workspace/Knowledge-Tracing/level2/grkt_api.py#L474)) làm theo
thứ tự:

1. `make_args` — đóng gói `n_knows`, `n_probs`, `rel_map`, `pre_map`, siêu tham số;
2. `build_grkt` — khởi tạo model, chuyển lên device;
3. in số cạnh và số tham số;
4. xử lý resume (ba định dạng, xem dưới);
5. `train_grkt` nếu chưa có checkpoint cuối;
6. `evaluate_grkt` trên test;
7. ghi `GRKT-<graph>.pt` + fingerprint, xoá `ckpt_GRKT-<graph>.pt`.

#### Ước lượng bộ nhớ

`grkt_activation_gib`
([experiment.py:63](file:///D:/workspace/Knowledge-Tracing/level2/experiment.py#L63)):

```text
bytes ≈ 40 · S · B · (K+1) · k_hidden · 4
```

| Cấu hình | K | B | S | k_hidden | Ước lượng |
|---|---:|---:|---:|---:|---:|
| ASSISTments | 123 | 32 | 200 | 16 | ~1,9 GiB |
| Junyi sau cap | 200 | 32 | 200 | 16 | ~3,1 GiB |
| Junyi không cap | 1.326 | 32 | 200 | 16 | ~20,3 GiB |

Dòng cuối là lý do `top_concepts=200` tồn tại: T4 có 15 GiB. Số người học không ảnh hưởng
công thức này; nó chỉ ảnh hưởng thời gian.

#### Vòng lặp train

`train_grkt` ([grkt_api.py:252](file:///D:/workspace/Knowledge-Tracing/level2/grkt_api.py#L252)):

| Thành phần | Chi tiết |
|---|---|
| Optimizer | `model.get_optimizer()` = Adam(lr, weight_decay=0) |
| Shuffle | `RandomState(seed*1000 + epoch).permutation(n_items)` |
| Loss | `F.binary_cross_entropy(scores, labels)` |
| Đánh giá | AUC/ACC **toàn cục** trên val |
| Early stop | `patience` theo val AUC |
| Checkpoint | mỗi `ckpt_every_s=600` giây và cuối mỗi epoch |
| Dừng chủ động | `deadline` → ghi checkpoint → `TrainingPaused` |

Điểm khác upstream: lưu `order` và `batch_pos` để resume **đúng batch đang dở**, kể cả
`loss_sum`/`loss_count` của epoch dở. Upstream `Runner.fit` chỉ biết epoch.

#### Ba định dạng checkpoint

| Định dạng | File | Nội dung | Khi nạp |
|---|---|---|---|
| `grkt-resume-v1` | `ckpt_GRKT-<graph>.pt` | weights + Adam + epoch + batch_pos + order + loss dở + best_state + RNG + fingerprint | train tiếp đúng batch |
| `grkt-final-v1` | `GRKT-<graph>.pt` | weights tốt nhất + fingerprint + best_val_auc | bỏ qua train, chỉ đánh giá |
| đời cũ | bất kỳ | chỉ weights | nạp kèm cảnh báo "không kiểm chứng được split" |

Thứ tự ưu tiên trong `experiment.py`: `GRKT-<graph>.pt` trước, rồi `ckpt_GRKT-<graph>.pt`.
Nếu file resume bị hỏng, `run_grkt_experiment` thử fallback sang `ckpt_...` trước khi raise.

#### Fingerprint

Fingerprint ([grkt_api.py:548](file:///D:/workspace/Knowledge-Tracing/level2/grkt_api.py#L548))
gồm: dataset, graph_kind, `n_knows`, `n_probs`, số người/số bước của 3 split, batch_size,
d_hidden, k_hidden, k_hop, alpha, tau, thresh, seed, lr, số cạnh rel/pre.

Vì sao cần: `load_state_dict` chỉ bắt lệch **kiến trúc**. Nó không bắt được việc chia split
khác, đồ thị khác, hay `batch_size` khác — tất cả đều cho ra trọng số cùng shape nhưng ý nghĩa
khác. `_check_fingerprint` raise thay vì lặng lẽ trộn hai lần chạy.

RNG cũng được lưu vì `gumbel_softmax` chạy cả ở eval. Không lưu RNG thì lần resume bắt đầu
một dòng ngẫu nhiên khác, val/test AUC lệch dù trọng số đúng.

#### Nhánh "đã train xong"

Khi nạp `grkt-final-v1`, `run_grkt_experiment` **vẫn chạy lại val** để lấy `best_val_auc`.
Vì eval có `gumbel_softmax`, con số này có thể lệch nhẹ so với phiên trước. Đây là hành vi đã
biết, được ghi trong log, không phải lỗi.

### 3.7 B5 — đường nền DKT/GKT trên cùng split

`level1_baseline` ([baselines.py:373](file:///D:/workspace/Knowledge-Tracing/level2/baselines.py#L373))
chạy ba cấu hình: DKT, GKT-Transition, GKT-Zero.

#### Đổi đồ thị mà không đổi split

`graph_bundle` ([baselines.py:45](file:///D:/workspace/Knowledge-Tracing/level2/baselines.py#L45))
gọi lại `load_kt_data` với `graph_type` khác, rồi **assert** danh sách user của ba split giống
bundle gốc từng phần tử. Nhờ vậy nếu upstream đổi thứ tự thao tác, chương trình fail thay vì
âm thầm so hai split khác nhau.

> Ghi chú review: docstring của `graph_bundle` nói "dựng đồ thị trước rồi mới random_split".
> Đọc `GKT/processing.py` thì thứ tự thực tế là **split trước, đồ thị sau** (split ở dòng
> 134–141, graph ở dòng 146+). Kết luận "đổi graph không đổi split" vẫn đúng vì generator của
> split cố định; chỉ câu mô tả thứ tự là sai. Không ảnh hưởng kết quả.

#### Nạp checkpoint hay train lại

| Nguồn | Điều kiện | Hành vi |
|---|---|---|
| `ckpt_dir/<name>.pt` (Cấp độ 1) | chỉ `assist2009` | nạp `state_dict` với `weights_only=True` |
| `save_dir/baselines/<name>.pt` | lần chạy trước đã train lại | kiểm `.json` fingerprint rồi nạp |
| `save_dir/baselines/ckpt_<name>.pt` | train dở | resume từ epoch đã xong |
| không có | — | train mới `baseline_epochs` (20) |

`load_level1_state` trả `False` thay vì raise khi shape không khớp, và in lý do. Nhờ vậy đổi
`hid_dim` ở Cấp độ 1 không làm chết Cấp độ 2; nó chỉ khiến đường nền bị train lại.

`train_baseline_resumable`
([baselines.py:251](file:///D:/workspace/Knowledge-Tracing/level2/baselines.py#L251)) khác
`train_one` ở ba điểm:

* ghi checkpoint **cuối mỗi epoch** (một epoch Junyi đã hàng chục phút);
* nếu `time.time() + 1.1·epoch_trước > deadline` thì dừng **trước** epoch kế, ghi checkpoint,
  trả `paused=True`;
* người gọi raise `TrainingPaused` để notebook đóng gói zip.

Trọng số hoàn chỉnh ghi vào `save_dir/baselines/`, **không bao giờ ghi đè `ckpt_dir`** — tên
file trùng với sản phẩm Cấp độ 1 (`DKT.pt`).

`collect_level1_probs` dùng cùng mask với `_epoch_loop`, nên `(probs, labels)` của DKT/GKT
dùng thẳng được cho ECE/Brier ở B7.

### 3.8 B6 — quỹ đạo mastery và tiêu chí hợp lý

Đây là nhiệm vụ bắt buộc của Cấp độ 2. `record_trajectories`
([trajectory.py:59](file:///D:/workspace/Knowledge-Tracing/level2/trajectory.py#L59)) chạy
forward trên 150 người test đầu tiên, batch 16, và ghi ba mốc:

| Mốc | Thời điểm | Dùng để tách cơ chế |
|---|---|---|
| `m_pre[i]` | trước khi cập nhật bước `i` | trạng thái đã dùng để dự đoán `score[i]` |
| `m_tilde[i]` | sau truy xuất trên đồ thị | phần lan truyền ảnh hưởng bao nhiêu |
| `m_strong[i]` | sau củng cố đúng/sai | việc làm đúng/sai ảnh hưởng bao nhiêu |

Hai hiệu:

```text
Δm_củng_cố = m_strong[i] - m_pre[i]
Δm_học_quên = m_pre[i+1] - m_strong[i]
```

Lưu ý kỹ thuật:

* cột 0 (padding) bị ép về 0 vì upstream `scatter` có thể ghi đè lên hàng padding;
* giá trị thô lưu ở `*_raw`; bản không đuôi là `sigmoid(raw)` để vẽ/so sánh;
* GRKT không chặn mastery trong [0,1]; sigmoid là biến đổi đơn điệu, không đổi dấu của hiệu.

Bốn tiêu chí:

| Tiêu chí | Công thức | Cách đọc |
|---|---|---|
| `strengthen` | tỉ lệ bước mà `sign(m_strong - m_pre)` đúng chiều với kết quả | gần 1 là đúng theo cấu trúc; nếu ≠ 1 thì hook ghi hoặc tiền xử lý sai |
| `spread` | như trên nhưng tại láng giềng 1-hop, trừ khái niệm đang làm | số đo **thật sự có thể sai**; so `train`/`shuffled`/`dense` |
| `locality` | `Σ|Δm|` trong tập đang làm ∪ láng giềng / `Σ|Δm|` toàn bộ | đồ thị zero cho 1.0 một cách tầm thường; đọc cùng `spread` |
| `forget_slope` | hồi quy trung bình `Δm_học_quên` theo Δt (5 ngăn) | kỳ vọng ≤ 0; dương nghĩa là "không làm gì cũng giỏi lên" |

`stage_attribution` đo độ lớn trung bình của ba giai đoạn (`strengthen`, `learn_forget`,
`retrieval`), phục vụ phần biện luận mô hình học từ đâu.

`trajectory_metrics_table` **dùng chung `adj` của đồ thị train cho mọi biến thể**. Có chủ đích:
"láng giềng" phải là một tập cố định thì `spread`/`locality` mới so sánh được. Với biến thể
`zero`, `spread=None` (không có thay đổi trên láng giềng) và `locality=1`.

Sản phẩm:

| File | Nội dung |
|---|---|
| `trajectories/traj_<kind>.npz` | mảng `float16` cho mastery, int cho chỉ số; mỗi người một bộ `u{i}__*` |
| `trajectories/mastery_<kind>.csv` | bảng dài 25 người đầu: `(user, step, concept, m_pre, m_strong, m_tilde, correct)` |
| `trajectories/heatmap_<kind>.png` | khái niệm × bước của người đầu tiên |
| `trajectories/curves_<kind>.png` | 4 khái niệm của 2 người, nền xanh/đỏ theo đúng/sai |
| `trajectory_metrics.csv` | bảng tiêu chí |

### 3.9 B7 — hiệu chỉnh dự đoán (tuỳ chọn)

`calibration_table` + `expected_calibration_error`
([calibration.py:32](file:///D:/workspace/Knowledge-Tracing/level2/calibration.py#L32)):

* `Brier = mean((p - y)²)`;
* `ECE = Σ_b (n_b/N) · |avg(p_b) - avg(y_b)|` trên 10 ngăn đều theo `p`;
* `ACE` = trung bình không trọng số, nhạy với ngăn thưa;
* `over_prediction = mean(p) - mean(y)`.

Đầu vào là `base_probs` (DKT/GKT đã thu ở B5) cộng probs GRKT lấy từ `predict_grkt`. Vì
`base_probs` luôn được thu khi `run_baselines=True`, bật B7 không cần chạy lại đường nền.

### 3.10 B8 — đa seed (tuỳ chọn)

Chọn `train` nếu có, nếu không lấy biến thể đầu tiên. Chạy `run_grkt_experiment` với
`save_dir=None` để **không ghi đè** checkpoint chính. Kết quả gộp bằng `groupby("model")`,
`std` mẫu (`ddof=1`) với `n_seed` ghi kèm.

Mỗi seed là một lần train đầy đủ; đây là lý do mặc định tắt.

### 3.11 `run_level2` — nhiều tập, một deadline

[experiment.py:527](file:///D:/workspace/Knowledge-Tracing/level2/experiment.py#L527):

```python
deadline = time.time() + time_budget_min * 60   # tính MỘT lần
for name in datasets:
    run = run_dataset(name, deadline=deadline, ...)
combined.to_csv(save_root / "level2_all_datasets.csv")
```

Ngân sách **không** được cấp lại cho tập thứ hai. Nếu Junyi ăn hết thời gian, `run_dataset`
của tập sau raise `TrainingPaused` ngay trước batch đầu, kèm thông báo tăng ngân sách hoặc
chạy riêng tập đó. Đây là hành vi có chủ đích: thà dừng sớm còn hơn bị Kaggle kill giữa lúc
ghi checkpoint.

### 3.12 Sản phẩm Cấp độ 2 (mỗi dataset)

```
/kaggle/working/level2_out/<dataset>/
├─ split.json
├─ leak_report.txt
├─ graphs_train_only.npz        # rel, pre (thô) + rel_bin, pre_bin (train)
├─ level2_metrics.csv
├─ summary.json
├─ GRKT-<graph>.pt
├─ ckpt_GRKT-<graph>.pt         # chỉ tồn tại khi train dở
├─ baselines/                   # chỉ khi phải train lại đường nền
│  ├─ <name>.pt
│  ├─ <name>.json               # fingerprint
│  └─ ckpt_<name>.pt
├─ trajectories/…
├─ trajectory_metrics.csv
├─ calibration.csv              # nếu RUN_CALIBRATION
├─ reliability.png
├─ reliability_rows.csv
└─ multiseed_metrics.csv        # nếu RUN_MULTI_SEED
```

Ngoài ra `level2_all_datasets.csv` nằm ở `save_root`.

### 3.13 Resume qua nhiều phiên Kaggle

```
PHIÊN A
  §0.3  không có zip → "train từ đầu"
  §3    run_level2:
          B1..B3  chia split, dựng chuỗi, dựng đồ thị
          B4      train GRKT → ckpt_GRKT-train.pt mỗi 10 phút
                  hết TIME_BUDGET_MIN → ghi ckpt → TrainingPaused
  catch TrainingPaused → package_output → level2_output.zip
  người dùng TẢI ZIP VỀ

PHIÊN B
  upload zip thành private Kaggle Dataset → Add Input
  RESUME_ZIP = "/kaggle/input/<slug>/level2_output.zip"   (hoặc để None nếu chỉ có 1)
  §0.3  giải nén, bỏ qua file đích mới hơn hoặc bằng file trong zip
  §3    B1..B3 chạy lại (nhanh, không train)
          B4  thấy GRKT-train.pt → chỉ đánh giá
              hoặc ckpt_GRKT-train.pt → train tiếp đúng batch
  §8.5  đóng gói lại, tải về
```

Quy tắc đóng gói (`package_output`,
[archive.py:23](file:///D:/workspace/Knowledge-Tracing/level2/archive.py#L23)):

| Quy tắc | Lý do |
|---|---|
| bỏ `junyi_canonical.csv` + `.meta.json` | tái sinh được, ~1 GB |
| bỏ `.tmp` | dấu vết phiên bị kill giữa lúc ghi |
| bỏ file > `max_mb=150`, **trừ** checkpoint resume | thiếu checkpoint là mất khả năng resume |
| `keep_ckpts=False` bỏ cả `.pt`/`.npz` | khi chỉ cần số liệu + ảnh (mất resume) |
| in số checkpoint resume trong zip | cảnh báo nếu bằng 0 |
| kiểm tra toàn vẹn zip | `testzip()` sau khi ghi |

Cell §0.3 phía notebook còn chặn **zip-slip**: tên file bắt đầu bằng `/` hoặc chứa `..` bị
raise, vì zip được tải lên từ máy khác.

### 3.14 Bảng lỗi và rủi ro đã kiểm tra

| # | Vấn đề | Bằng chứng | Ảnh hưởng | Xử lý hiện tại |
|---|---|---|---|---|
| 1 | Docstring `graph_bundle` mô tả sai thứ tự split/graph | `baselines.py:60` vs `GKT/processing.py:134-160` | chỉ gây hiểu nhầm | assert danh sách user vẫn bảo vệ đúng |
| 2 | `GKT/` import bằng tên module toàn cục | `level1/paths.py:9` | có thể import nhầm `models.py` khác | chưa xử lý; chỉ xảy ra khi thêm package trùng tên |
| 3 | Cấp độ 2 không đọc `split.json`, chia lại theo tham số | `experiment.py:207-225` | đổi ratio ở Cấp độ 1 mà quên Cấp độ 2 → khác split | `_assert_align` bắt lệch chuỗi; đối chiếu DKT 0.7866 là chốt cuối |
| 4 | `leak_report` in giao tập nhưng không assert | `grkt_data.py:493` | giao ≠ 0 không làm dừng | cách chia user-wise bảo đảm; báo cáo để kiểm tra bằng mắt |
| 5 | Eval GRKT có `gumbel_softmax` | `trajectory.py:76` | kết quả eval lệch nhẹ giữa các lần | seed cố định; đã ghi trong log nhánh "đã train xong" |
| 6 | `best_val_auc=None` khi nạp checkpoint đường nền | `baselines.py:451` | cột `val_auc` trống trong bảng | `_table_row` xử lý `None` |
| 7 | `.meta.json` thiếu thì chỉ cảnh báo | `dataset_spec.py:644` | không phát hiện được cấu hình cũ | xoá file khi nghi ngờ; chạy lại từ đầu |
| 8 | `MAX_USERS` không giảm thời gian quét Junyi | `dataset_spec.py` (lọc sau concat) | tưởng nhanh nhưng vẫn 15–30 phút | ghi rõ trong README |
| 9 | `run_dataset` default `max_seq_len=100`, notebook dùng 200 | `experiment.py:119` vs §0.2 của notebook Cấp độ 2 | gọi trực tiếp package sẽ khác cấu hình Cấp độ 1 | notebook là nguồn sự thật |
| 10 | `RUN_MULTI_SEED=True` mặc định ở notebook Cấp độ 1 | §4.5 của `kaggle_level1.ipynb` | 21 lần train trong một lần chạy | tắt khi không cần báo cáo đa seed |
| 11 | `run_level1` thiếu `Shuffled` so với notebook | `pipeline.py:386` | tái lập bằng package thiếu 1 dòng | dùng notebook hoặc bổ sung |
| 12 | Vẽ hình có thể lỗi sau khi đã train xong | `_safe_plot` `experiment.py:97` | mất hình, không mất số liệu | bọc try/except, in cảnh báo |

---

## 4. Cầu nối giữa hai cấp

### 4.1 Chuỗi bảo đảm "cùng split"

```text
Cấp độ 1                                     Cấp độ 2
---------                                    ---------
skill_builder_data.csv                       cùng file (hoặc bản chuẩn hoá tương đương)
min_seq_len=3                                min_interactions=3
max_seq_len=200                              max_seq_len=200
random_split(seed=42, 0.6/0.2/0.2)           load_kt_data(seed=42, 0.6/0.2/0.2)
        │                                            │
        └────────── cùng người, cùng thứ tự ─────────┘
                                                     │
                                    build_grkt_data + _assert_align từng người
```

### 4.2 Bảng đối chiếu hai cấp

| Khía cạnh | Cấp độ 1 | Cấp độ 2 |
|---|---|---|
| Câu hỏi nghiên cứu | GKT/GNN có hơn DKT không; đồ thị nào có tác dụng | GRKT + quỹ đạo mastery trên ≥ 2 tập |
| Dữ liệu | 1 tập ASSISTments (+ smoke) | ASSISTments + Junyi |
| Đồ thị | Transition/Dense/Identity/Zero/Shuffled | `rel_map`/`pre_map` + train/dense/zero/shuffled |
| Đơn vị chia | người học | người học (cùng seed 42) |
| Loss | NLL có mask | BCE |
| Metric | AUC/ACC toàn cục | AUC/ACC toàn cục (cùng mask) |
| Early stop | val AUC, patience 5 | val AUC, patience 3 (notebook) |
| Checkpoint | cuối train | cuối epoch + giữa epoch + fingerprint |
| Giải thích | phản chứng flip + bỏ cạnh | quỹ đạo mastery + 4 tiêu chí |
| Resume | không | có, qua zip nhiều phiên |
| Sản phẩm chính | `cases.md`, `learner_view.json` | `trajectory_metrics.csv`, `traj_*.npz` |

### 4.3 Vì sao Cấp độ 2 không dùng `GRKT/Runner.py`

| Vấn đề của Runner | Hệ quả | Cấp độ 2 thay bằng |
|---|---|---|
| k-fold trên **cửa sổ 100 tương tác** | cùng người xuất hiện ở train và test | split theo người của Cấp độ 1 |
| `deepcopy(self)` để giữ model tốt nhất | giữ cả tensor GPU → hết VRAM sau vài epoch | snapshot `state_dict` về CPU |
| không có checkpoint giữa epoch | phiên Kaggle bị cắt là mất toàn bộ | `ckpt_*` mỗi 10 phút |
| đồ thị dựng từ toàn bộ dữ liệu | rò rỉ qua đồ thị | đếm `rel/pre` chỉ trên train |

---

## 5. Cấu hình thực tế của hai notebook

### 5.1 Cấp độ 1

| Biến | `QUICK_RUN=False` | `QUICK_RUN=True` |
|---|---:|---:|
| `DATA_FILE` | `skill_builder_data.csv` | `assistment_test15.csv` |
| `EPOCHS` | 20 | 8 |
| `BATCH_SIZE` | 32 | 16 |
| `MAX_SEQ_LEN` | 200 | 80 |
| `RUN_MULTI_SEED` | `True` | `True` |
| `SAVE_DIR` | `outputs/level1` hoặc `/kaggle/working/level1_out` | như trái |

### 5.2 Cấp độ 2

| Biến | `QUICK_RUN=False` | `QUICK_RUN=True` |
|---|---:|---:|
| `DATASETS` | `["assist2009", "junyi"]` | `["assist2009_smoke"]` |
| `EPOCHS` | 5 | 2 |
| `BATCH_SIZE` | 32 | 16 |
| `MAX_SEQ_LEN` | 200 | 50 |
| `MAX_USERS` | `None` (toàn bộ) | 300 |
| `VARIANTS` | `("train",)` (ablation tắt) | `("train",)` |
| `RUN_BASELINES` | `True` | `True` |
| `RUN_CALIBRATION` | `False` | `False` |
| `RUN_MULTI_SEED` | `False` | `False` |
| `RESUME_GRKT_CKPT` | `True` | `True` |
| `TIME_BUDGET_MIN` | 600 (chặn thêm bởi đồng hồ phiên) | 30 |
| `CKPT_EVERY_MIN` | 10 | 10 |
| `D_HIDDEN` / `K_HIDDEN` / `K_HOP` | 128 / 16 / 1 | như trái |
| `LR` | 1e-3 | 1e-3 |
| `PATIENCE` | 3 | 2 |
| `GRAPH_THRESH` | 0.6 | 0.6 |
| `COLD_THRESH` | 5 | 1 |
| `ALPHA` | 0.01 | 0.01 |
| `GAP_CAP_MIN` | 240 | 240 |
| `BASELINE_EPOCHS` | 20 | 2 |
| `SPLIT_SEED` | 42 | 42 |
| `SESSION_LIMIT_H` / `SAFETY` | 12 h / 45 phút | như trái |

`TIME_BUDGET_MIN` bị chặn thêm bởi thời gian còn lại của phiên:
`min(TIME_BUDGET_MIN, SESSION_LIMIT_H·60 - đã_chạy - SESSION_SAFETY_MIN)`. Nếu đọc đồng hồ
phiên thất bại, notebook bỏ chặn và tính từ lúc chạy cell §3.

---

## 6. Cách đọc kết quả

### 6.1 `level2_metrics.csv`

| Cột | Nghĩa |
|---|---|
| `model` | `DKT`, `GKT-Transition`, `GKT-Zero`, `GRKT-train` |
| `model_type` / `graph` | loại mô hình và biến thể đồ thị |
| `auc` / `acc` / `loss` | test |
| `val_auc` | val tốt nhất (trống nếu nạp checkpoint đường nền) |
| `n_test` | số dự đoán test được tính |
| `edges_rel` / `edges_pre` | số cạnh (0 với DKT/GKT) |
| `seed` / `seconds` | seed và thời gian train |

### 6.2 `trajectory_metrics.csv`

| Cột | Đọc thế nào |
|---|---|
| `consistency_strengthen` | phải gần 1; ≠ 1 là lỗi hook/tiền xử lý |
| `consistency_spread` | so giữa `train` và `shuffled`; `None` với `zero` |
| `locality` | đọc cùng `spread`, không đọc một mình |
| `forget_slope` | kỳ vọng ≤ 0 |
| `abs_delta_*` | mô hình học chủ yếu từ củng cố hay từ thời gian |

### 6.3 Kiểm tra đối chiếu bắt buộc

1. **DKT trên ASSISTments phải tái hiện 0.7866 ± 0.005.** Lệch nghĩa là split hoặc
   `max_seq_len` khác Cấp độ 1.
2. **`train∩val = train∩test = val∩test = 0`** trong `leak_report.txt`.
3. **Số cạnh `train` > 0.** Bằng 0 thì GRKT-train chạy y hệt GRKT-zero và mọi kết luận về đồ
   thị vô nghĩa (đã có cảnh báo trong `experiment.py`).
4. **`Identity == Zero`** trên Cấp độ 1 là kết quả đã biết; nếu khác nhiều, nghi ngờ
   `graph_type` bị truyền sai.
5. **`consistency_strengthen ≈ 1`.** Nếu không, kiểm tra hook `rec` và thứ tự cập nhật.

---

## 7. Checklist chạy Kaggle

### 7.1 Cấp độ 1

- [ ] Repo clone đúng nhánh; `GKT/data/skill_builder_data.csv` tồn tại.
- [ ] `QUICK_RUN=False` cho số liệu báo cáo.
- [ ] Chạy lần lượt cell 3.1 → 3.4 để `results` có đủ mô hình.
- [ ] Cell 5 cần `results` có `GKT-Transition`; nếu đổi tên, sửa `next(...)`.
- [ ] Tải `outputs/level1/` hoặc `/kaggle/working/level1_out/` trước khi phiên kết thúc.
- [ ] Nếu định dùng checkpoint cho Cấp độ 2: phải giữ `DKT.pt` và `GKT-*.pt`.

### 7.2 Cấp độ 2

- [ ] Junyi: bấm **Agree/Subscribe** trên Kaggle trước khi `Add input`; nếu không, input rỗng.
- [ ] `RESUME_ZIP` đúng đường dẫn (hoặc `None` khi `/kaggle/input` chỉ có một zip/cây).
- [ ] Chạy §0.3 trước §3; kiểm tra dòng "đã phục hồi N file" và danh sách checkpoint.
- [ ] Không đổi `SPLIT_SEED=42`, `MAX_SEQ_LEN=200`, `min_interactions` giữa các phiên.
- [ ] Chạy §8.5 trước khi tắt phiên, kể cả khi train chưa xong.
- [ ] Phiên sau: giữ `RESUME_GRKT_CKPT=True`; muốn train lại từ đầu thì đặt `False` **và** xoá
      cả `ckpt_GRKT-*.pt` lẫn `GRKT-*.pt`.
- [ ] Nếu chỉ còn dưới 1 giờ, đừng bật thêm ablation/calibration/multi-seed.

---

## 8. Tóm tắt một câu cho từng package

* **`level1/`** là lớp API sửa lỗi và đóng gói upstream `GKT/`: chia split theo người một lần,
  train DKT/GKT, lưu checkpoint, xuất bằng chứng giải thích và dashboard.
* **`level2/`** là pipeline mới đặt trên `GRKT/`: chuẩn hoá mọi tập về một CSV, chia lại đúng
  split Cấp độ 1, dựng đồ thị chỉ từ train, train GRKT có checkpoint/resume, ghi quỹ đạo
  mastery và các tiêu chí hợp lý, đóng gói zip để chạy tiếp qua nhiều phiên Kaggle.

---

## 9. Thuật ngữ nhanh

| Thuật ngữ | Nghĩa trong tài liệu này |
|---|---|
| bundle | Dict `{concept_num, graph, train_loader, val_loader, test_loader, data_path}` |
| split | Danh sách `user_id` của train/val/test |
| chuẩn hoá | Đổi file gốc về 8 cột `CANONICAL_COLUMNS` |
| fingerprint | Bộ tham số nhận dạng cấu hình đã sinh checkpoint |
| deadline | Mốc thời gian tuyệt đối; hết mốc thì ghi checkpoint rồi dừng |
| `m_pre` / `m_tilde` / `m_strong` | Mastery trước bước, sau truy xuất, sau củng cố |
| `rel` / `pre` | Đồng xuất hiện / tiên quyết, đếm từ train |
| `spread` | Mức lan truyền đúng chiều sang khái niệm láng giềng |
