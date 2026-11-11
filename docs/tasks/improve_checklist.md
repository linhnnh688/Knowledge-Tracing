# Improve checklist: Knowledge-Tracing (Level 1 + Level 2)

> Rà ngày 2026-11-11 trên `main@b5c5d8f`. Phạm vi: `GKT/`, `GRKT/`, `level1/`, `level2/`, 2 notebook
> `kaggle_level{1,2}_full.ipynb`, dữ liệu `GKT/data/skill_builder_data.csv`.
> Mọi con số "đo được" dưới đây được đếm trực tiếp trên file CSV gốc (awk) hoặc đọc từ code. Con số
> "ước tính" thì có ghi rõ là ước tính.

## Cách đọc

Các mục được xếp theo **mức độ công sức, từ thấp đến cao**. Mỗi mục có thêm nhãn **tác động** riêng:

| Mức | Ý nghĩa | Công sức điển hình |
|---|---|---|
| **Mức 0: Chặn kết quả** | Sai về phương pháp. Số liệu hiện tại **không nên** đưa vào báo cáo trước khi sửa | 0.5–2 ngày, nhưng phải chạy lại toàn bộ |
| **Mức 1: Quick win** | Sửa vài dòng, không đổi kết quả khoa học | < 1 giờ/mục |
| **Mức 2: Refactor / DRY** | Gom code trùng, xoá code chết, chuẩn hoá cấu trúc | 0.5–1 ngày/mục |
| **Mức 3: Tăng tốc build/train** | Giảm thời gian chạy trên Kaggle | 0.5–2 ngày/mục |
| **Mức 4: Nâng chất lượng học thuật** | Thiết kế thực nghiệm, thống kê, baseline, tiêu chí | 1–5 ngày/mục |
| **Mức 5: Mở rộng (Level 3–4 của đề)** | Phần đề bài còn thiếu | nhiều ngày |

Nhãn tác động: 🔴 cao · 🟠 vừa · 🟢 thấp.


## Kết quả thực hiện Mức 1–2 (2026-11-11, nhánh `features/clean-dry-enhancement`)

**Không đổi kết quả khoa học:** golden test so code gốc với code mới trên CPU, gồm bundle ASSIST09
đầy đủ (5 loại đồ thị), chuỗi GRKT đầy đủ, train DKT/GKT-Transition/GKT-MHA/GKT-Zero (hash
trọng số), 10 case xAI, dashboard, toàn bộ `run_dataset` Level 2 (GRKT train/zero, baseline, quỹ
đạo, calibration, đa seed). Kết quả **khớp từng bit**; chỉ khác đường dẫn tuyệt đối của file dữ
liệu và tên file checkpoint (thêm `-seed42`). Code gốc chạy hai lần cũng khớp từng bit, nên phép
so sánh có ý nghĩa.

**Cập nhật 2026-11-12 — đối chiếu `GAP_LEVEL1.md` §6.4:** cherry-pick `09d9431` (BatchNorm eval —
nguyên nhân 10/10 case xAI không tái lập), sửa occlusion trong notebook, thêm kiểm tra nạp lại
checkpoint (GAP 6.4.5), tiền xử lý trùng khít nhánh `fix/assist2009-leak-batchnorm`. Chi tiết:
`GAP_LEVEL1.md` §6.6.

**Phát hiện mới trong lúc làm** (chưa sửa, cần quyết định):

- **2.10** 🟠 Fingerprint baseline chứa `data_path` **tuyệt đối** (`level2/baselines.baseline_fingerprint`),
  nên checkpoint train ở `/kaggle/working/...` sẽ bị từ chối khi resume ở đường dẫn khác (máy local,
  dataset Kaggle mount khác). Nên dùng tên file + hash nội dung.
- **2.11** 🟠 Định nghĩa `loss` khác nhau giữa Level 1 (trung bình theo batch) và GRKT (trung bình
  theo tương tác) — xem 2.3.
- **2.12** 🟢 `pd.read_csv` trên `skill_builder_data.csv` báo `DtypeWarning` (cột `skill_name` kiểu
  trộn). Vô hại cho số liệu hiện tại nhưng nên khai báo `dtype` khi sửa 0.1/0.2.
- **2.13** 🟢 `GKT.metrics.KTLoss` gọi `roc_auc_score` cho MỌI batch (kết quả bị bỏ, sinh cảnh báo) —
  tốn thời gian; thuộc Mức 3.
- **1.1** là phát hiện sai (xem mục 1.1).

---

## Mức 0: Chặn kết quả (sửa trước mọi việc khác)

### [x] 0.1 🔴 Rò nhãn do dòng trùng đa khái niệm trong ASSISTments 2009

> **Trạng thái 2026-11-12:** Đã sửa trong `kt_common/interactions.py` (luôn bật, không còn chế độ cũ): gộp dòng trùng `(user_id, order_id)`, giữ dòng đầu. ASSIST09: 458.766 → 282.619 tương tác, 121.580 → 0 cặp trùng liền kề, 3.884 → 3.852 người học, 123 → 112 khái niệm. **Hạn chế:** 11 kỹ năng chỉ xuất hiện ở vị trí kỹ năng thứ hai nên bị mất; Q-matrix đa khái niệm cần sửa mô hình. `DATA_VERSION` ghi vào fingerprint/manifest/`run_config.json` nên checkpoint & zip cũ bị từ chối. Số liệu `results/` là TRƯỚC bản sửa — phải train lại trên Kaggle.

**Hiện trạng (đo được trên `skill_builder_data.csv`):**

| Chỉ số | Giá trị |
|---|---:|
| Tổng số dòng | 525.534 |
| Dòng trùng `order_id` (một lượt làm bài bị nhân bản cho mỗi skill) | **178.674 (34%)** |
| Cặp dòng trùng **đứng liền nhau** trong chuỗi của cùng một người học | **124.165** |
| Trong các cặp liền nhau đó, số cặp **có cùng nhãn `correct`** | **124.165 (100%)** |

`GKT/processing.py:136` coi mỗi dòng là một bước thời gian. Khi một câu hỏi gắn 2 skill, chuỗi có
hai bước liên tiếp `(skill A, đúng)` rồi `(skill B, đúng)`. Mô hình dự đoán bước *t+1* sau khi đã
thấy nhãn của bước *t*, mà hai bước này là cùng một lượt trả lời. Như vậy nhãn cần đoán đã lộ sẵn
trong input. Đây đúng là lỗi mà các tài liệu sau đã chỉ ra:

* Xiong, Zhao, Van Inwegen & Beck, *Going Deeper with Deep Knowledge Tracing*, EDM 2016. Họ cho thấy
  AUC của DKT trên ASSIST09 bị thổi phồng chủ yếu do dòng trùng; gộp lại thì AUC giảm mạnh.
* Liu et al., *pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models*,
  NeurIPS 2022 Datasets & Benchmarks. Phần tiền xử lý của họ gộp dòng theo `order_id`, rồi đánh giá
  ở mức câu hỏi (question-level).

Ghi chú trong `level2/dataset_spec.py` (spec `ASSIST2009.note`) viết *"Mỗi dòng đúng một skill_id,
nên không explode Q-matrix"*. Câu này đúng về hình thức nhưng **sai về bản chất**: file gốc đã được
explode sẵn. Level 2 dùng lại đúng chuỗi của Level 1 (có `_assert_align`), nên **GRKT trên assist2009
cũng mắc cùng lỗi**. Junyi không bị, vì mỗi dòng log là một lượt với một `ucid`.

**Việc cần làm:**
- [ ] Gộp theo `(user_id, order_id)`: một lượt = một bước, giữ `skill_ids` dạng list (Q-matrix).
- [ ] Chọn một trong hai cách xử lý đa khái niệm và ghi lý do trong báo cáo:
  - (a) **Concept-level kiểu pyKT**: vẫn trải lượt thành *k* bước khái niệm, nhưng loss/metric chỉ
    tính ở **bước cuối của nhóm**, và input của các bước trong nhóm **không chứa nhãn** của nhóm.
  - (b) **Question-level**: embedding của bước = trung bình (hoặc tổng) embedding các khái niệm.
    GKT cập nhật đồng thời *k* nút. GRKT đã hỗ trợ sẵn `knows` dạng list, chỉ cần bỏ chặn
    `max_concepts > 1` ở `level2/grkt_data.py` (hàm `build_grkt_data`).
- [ ] Thêm test tự động: không có hai bước liên tiếp nào cùng `order_id` (xem 2.9).
- [ ] Chạy lại toàn bộ Level 1 và Level 2 trên assist2009. Báo cáo cả số cũ lẫn số mới, kèm một đoạn
  "lesson learned". Giám khảo đánh giá cao việc tự phát hiện leak.

### [x] 0.2 🔴 Chuỗi không được sắp theo thời gian

> **Trạng thái 2026-11-12:** Đã sửa cùng chỗ: sắp ổn định theo `order_id` trong từng người học, giữ thứ tự người học theo lần xuất hiện đầu tiên. 25.577 → 0 lần đi lùi. Split vẫn là `random_split(seed=42)` trên danh sách người học mới, nên tập test KHÁC tập test của `results/` cũ.

Đo được: có **27.231 lần `order_id` giảm** giữa hai dòng liên tiếp của cùng một người học. Ví dụ
user 64525 có thứ tự `33022709 → 28186893`. File gốc được nhóm theo skill/assignment, không theo thời
gian. `GKT/processing.py` không sort trước khi `groupby`. Riêng Junyi đã được sort ở `dataset_spec.py`,
trong khi ASSIST09 thì chưa.

Hệ quả: (1) yêu cầu "chỉ dùng tương tác **trước** thời điểm dự đoán" ở YEU_CAU §6.1 bị vi phạm về
mặt ngữ nghĩa, vì mô hình có thể được học từ lượt làm sau để đoán lượt làm trước. (2) Đồ thị
Transition đếm cặp chuyển tiếp *A→B* theo thứ tự sai, nên cạnh tiên quyết bị nhiễu.

- [ ] `df.sort_values(["user_id", "order_id"], kind="mergesort")` trước bước groupby, làm ở **một
  nơi duy nhất** (xem 2.1).

### [~] 0.3 🔴 Cắt chuỗi `max_seq_len=200` vứt bỏ 48% dữ liệu

> **Trạng thái 2026-11-12:** **Không phải quick win** (cửa sổ trượt phá giả định "một item = một người học" ở explain/dashboard/Level 2), chưa sửa. Đã làm phần đo: `data_stats` (`truncated_rows`, `truncated_frac`) ghi vào bundle, `run_config.json` và `summary.json`. Sau bản sửa: 29,8% tương tác bị cắt (trước: 48,1%).

Đo được (sau lọc ≥ 3 tương tác): 503/3.884 người học có hơn 200 dòng. Phần bị cắt là **220.823/458.766
dòng ≈ 48%**. Con số này tính cả dòng trùng của 0.1, nên sau khi gộp sẽ thấp hơn, nhưng vẫn đáng kể.
`processing.py:117-125` chỉ giữ 200 bước **đầu tiên**.

Hai hệ quả học thuật:
* **Lệch mẫu**: mô hình chỉ thấy giai đoạn đầu của người học, nên trạng thái mastery giai đoạn sau
  (giai đoạn mà Level 2 cần mô tả) không bao giờ được huấn luyện hay đánh giá.
* Đồ thị Transition cũng chỉ được đếm từ phần đầu chuỗi.

- [ ] Làm theo pyKT: **chia mỗi chuỗi thành các đoạn 200 bước không chồng lấn**. Mọi đoạn của một
  người học vẫn nằm trọn trong split của người đó, nên split theo người học vẫn được giữ.
- [ ] Hoặc, ở bước đánh giá, dùng cửa sổ trượt có lịch sử (stride 1, mỗi dự đoán nhìn tối đa 200 bước
  trước). Cách này đắt hơn nhưng không mất bước nào.
- [ ] Báo cáo số tương tác được dùng ở mỗi split.

### [~] 0.4 🟠 So sánh không công bằng về dung lượng mô hình và số epoch

> **Trạng thái 2026-11-12:** **Không phải quick win** (tune cần GPU). Đã làm phần đo: `n_params` trong kết quả `run_experiment`, `level1_baseline`, `run_grkt_experiment`, `metrics.json` Level 1 và `summary.json` Level 2.

| | DKT / GKT | GRKT |
|---|---|---|
| hidden | 32 (`level1/pipeline.py`, `hid_dim=32`) | `D_HIDDEN=128`, `K_HIDDEN=16` |
| epoch | 20 (có early stop) | 5 (theo `GAP_LEVEL2.md` §1.1) |
| tune siêu tham số | không | không |

Câu "GRKT thắng" đang bị nhiễu bởi chênh lệch dung lượng. Khi chạy lại sau 0.1–0.3:
- [ ] Báo cáo **số tham số** của từng mô hình.
- [ ] Cho mọi mô hình cùng một **ngân sách tune** trên tập validation (ví dụ grid nhỏ `hidden ∈ {32,
  64, 128}`, `lr ∈ {1e-3, 3e-3}`, dropout `{0, 0.2}`) và cùng quy tắc early stop.

---

## Mức 1: Quick win (< 1 giờ mỗi mục)

- [x] ~~**1.1** `.gitignore` có dòng `requirements.txt`~~ — **phát hiện sai**: dòng đó là output của
  lệnh `ls requirements*` chạy ngay sau `cat .gitignore` lúc rà, `.gitignore` không chứa nó. Không cần sửa.
- [x] **1.2** 🟠 `requirements.txt` chỉ có cận dưới (`torch>=2.0`, `pandas>=2.0`, …), nên không tái lập
  được môi trường. Thêm `requirements.lock` (hoặc dùng `uv pip compile`) chốt đúng phiên bản đã sinh
  `results/level1_out/environment.txt`.
  → **Đã làm (2026-11-11):** Đã thêm `requirements-lock.txt` (phiên bản từ `results/level1_out/environment.txt`).
- [x] **1.3** 🟢 `set_seed` (`level1/pipeline.py:20`) chưa set `PYTHONHASHSEED` và
  `torch.use_deterministic_algorithms(True, warn_only=True)`. Thêm cả hai, kèm
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` khi chạy GPU.
  → **Đã làm (2026-11-11):** `kt_common/seed.py`; trên CPU golden test khớp từng bit, trên GPU có thể lệch nhẹ so với run cũ.
- [x] **1.4** 🟢 Tên checkpoint phụ thuộc magic number: `fname = f"{name}.pt" if seed == 42 else ...`
  (`pipeline.py:399`). Nên luôn dùng `"{name}-seed{seed}.pt"` và ghi một `manifest.json` (`name → seed,
  config hash, git sha`). Đây là gốc rễ của lỗi "trộn seed 456" từng xảy ra ở Level 2.
  → **Đã làm (2026-11-11):** `kt_common/checkpoint.py` (`checkpoint_path`/`find_checkpoint`/`record_checkpoint`); bên đọc vẫn nhận tên cũ. Nhãn trong bảng (`GRKT-train`, `DKT`) giữ nguyên.
- [x] **1.5** 🟢 `build_transition_graph` dùng `np.vectorize(inv)` rồi `np.diag(r_inv).dot(graph)`
  (`processing.py:240`), tức nhân ma trận *K×K* không cần thiết. Thay bằng
  `graph / np.where(rowsum == 0, 1, rowsum)[:, None]`.
  → **Đã làm (2026-11-11):** Nhân với nghịch đảo hàng (khớp từng bit với bản cũ).
- [x] **1.6** 🟢 `df.groupby("user_id").filter(lambda q: len(q) >= min_seq_len)` (`processing.py:95`,
  lặp lại ở `grkt_data.py`) chạy một lambda Python cho mỗi nhóm. Thay bằng
  `df[df.groupby("user_id")["user_id"].transform("size") >= min_seq_len]`, nhanh hơn khoảng 10–50 lần.
  → **Đã làm (2026-11-11):** Trong `kt_common.interactions.load_interactions` (dùng chung hai cấp).
- [x] **1.7** 🟢 Xoá nhánh `if model_type == "GIKT"` trong `_epoch_loop` (`pipeline.py:199`) và hàm
  `run_gikt_experiment` (`pipeline.py:411`), vì GIKT đã bị loại khỏi pipeline.
  → **Đã làm (2026-11-11):** Đã xoá.
- [x] **1.8** 🟢 `run_level1` (`pipeline.py:417`) không được notebook nào gọi, mặc định dùng file
  smoke `assistment_test15.csv`, và thiếu Shuffled/MHA so với notebook. Hoặc xoá hàm này, hoặc biến nó
  thành entry-point CLI duy nhất (xem 2.6).
  → **Đã làm (2026-11-11):** Viết lại thành entry-point: `run_level1(graph_types, seeds, config)` + CLI `python -m level1`.
- [x] **1.9** 🟢 Thay `print` bằng `logging`, kèm một tham số `verbose`/level. Log Kaggle sẽ đọc được hơn
  và có thể tắt khi chạy multi-seed.
  → **Đã làm (2026-11-11):** `kt_common/log.py` (logger `kt`, mặc định INFO ra stdout nên notebook vẫn thấy). Giữ `print` ở `level1/dashboard.py` vì đó là output hiển thị của widget.
- [x] **1.10** 🟢 Đặt cấu hình `ruff` + `pre-commit` (format + lint + `pyupgrade`). Code hiện có
  `Variable(...)` (API PyTorch < 0.4) trong `GKT/models.py` và nhiều import thừa.
  → **Đã làm (2026-11-11):** `pyproject.toml` `[tool.ruff]` + `.pre-commit-config.yaml`; `ruff check .` sạch. Không bật auto-format/pyupgrade toàn repo (diff lớn, không đổi hành vi). `Variable(...)` trong `GKT/` giữ nguyên (no-op, mã vendored).

---

## Mức 2: Refactor / DRY / best practice

### [x] 2.1 🔴 Một nguồn tiền xử lý duy nhất (Single Source of Truth)

> **Trạng thái 2026-11-11:** `kt_common/interactions.py`; `GKT/processing.load_dataset` và `level2/grkt_data.build_grkt_data` cùng gọi `load_interactions`/`iter_users`/`truncate_len`. Giữ `_assert_align` làm hàng rào. Chưa thêm cột `split`/parquet (thuộc Mức 3). **2026-11-12:** mục 0.1/0.2 sửa ở đúng chỗ này (bỏ chế độ cũ theo yêu cầu); thêm `DATA_VERSION`, `interaction_stats`; checkpoint không mang đúng `data_version` bị từ chối thay vì nạp âm thầm.

Hiện tại logic tiền xử lý **được viết hai lần**: `GKT/processing.load_dataset` cho Level 1 và
`level2/grkt_data.build_grkt_data` cho Level 2. Lần thứ hai chép lại từng bước (dropna, filter, factorize,
groupby, cắt), rồi dùng `_assert_align` để kiểm tra rằng hai bản cho ra cùng kết quả. Kiểm tra như vậy
là tốt, nhưng nó chỉ cần thiết vì code bị lặp.

Đề xuất: tạo module `kt_data/` sinh **một** đối tượng `InteractionTable` (parquet/npz), gồm `user_id,
order, skill_ids(list), problem_id, correct, t_ms, split`. Sau đó:
* Level 1 dựng `KTDataset` từ bảng này.
* Level 2 dựng `UserSequence` từ bảng này.
* Đồ thị (Transition, rel/pre) nhận `train_users` làm tham số bắt buộc. Nhờ vậy việc rò dữ liệu test
  vào đồ thị trở nên **không thể xảy ra** theo cấu trúc API, không cần dựa vào kỷ luật người viết code.

Việc 0.1, 0.2 và 0.3 chỉ cần sửa **một lần** ở module này.

### [x] 2.2 🟠 Gom helper trùng giữa `level2/baselines.py` và `level2/grkt_api.py`

> **Trạng thái 2026-11-11:** `kt_common/checkpoint.py`. Vòng epoch DKT/GKT gộp thành `level1.pipeline.fit` (dùng bởi `train_one` và `train_baseline_resumable`). Vòng GRKT giữ riêng (resume theo batch + RNG) nhưng dùng helper chung. Định dạng payload (`format`, khoá) không đổi.

| Chức năng | `baselines.py` | `grkt_api.py` |
|---|---|---|
| Lưu file atomic | `_atomic_save` :224 | `_atomic_torch_save` :184 |
| state_dict lên CPU | `_cpu_state` :233 | `_cpu_state_dict` :193 |
| Optimizer state lên CPU | `_optim_cpu` :237 | `_optimizer_state_cpu` :197 |
| Optimizer state về device | `_optim_to_device` :251 | `_optimizer_state_to_device` :212 |
| Kiểm tra fingerprint | `_check_baseline_fingerprint` :294 | `_check_fingerprint` :251 |

Ngoài ra còn `_as_id` (`level1/pipeline.py:81`, `level2/grkt_data.py:328`) và `_skill_name`
(`level1/explain.py:62`, `level1/dashboard.py:16`).

- [ ] Tạo `common/checkpoint.py`, chứa class `ResumableCheckpoint` (save/load, RNG, optimizer,
  fingerprint), và `common/ids.py`.
- [ ] Gộp hai vòng train resumable (`train_baseline_resumable` và `train_grkt`) thành **một** `Trainer`
  generic, nhận `forward_fn` và `loss_fn`. Hiện có ba vòng train gần giống nhau: `level1._epoch_loop`,
  `baselines.train_baseline_resumable`, `grkt_api.train_grkt`.

### [x] 2.3 🟠 Gom tính metric về một chỗ

> **Trạng thái 2026-11-11:** `kt_common/metrics.auc_acc` + `PROB_EPS`. **Phát hiện thêm:** `loss` Level 1 = trung bình của NLL trung bình từng batch, `loss` GRKT = NLL trung bình theo tương tác → cột `loss` giữa DKT/GKT và GRKT không cùng định nghĩa. Chưa đổi (đổi số liệu) — xem mục 2.10.

AUC/ACC/NLL đang được tính ở ba nơi (`level1/pipeline.py:230`, `level2/baselines.py:195`,
`level2/grkt_api.py:446-470`), mỗi nơi có cách mask và clamp riêng. Calibration lại ở chỗ thứ tư.
- [ ] Viết `common/metrics.py::binary_metrics(y_true, y_prob) -> {auc, acc, nll, brier, ece}`, dùng
  chung một quy ước clamp `[1e-6, 1-1e-6]`. Như vậy bảng Level 1 và Level 2 chắc chắn so sánh được với
  nhau.

### [x] 2.4 🟠 Bỏ hack `sys.path` cho `GKT/`

> **Trạng thái 2026-11-11:** `GKT/` import tương đối; `pyproject.toml` (`pip install -e .`); 2 notebook full chỉ chèn REPO vào `sys.path`, `from GKT.metrics import KTLoss`. `GKT/train.py` chạy bằng `python -m GKT.train`.

`level1/paths.py` chèn `GKT/` vào `sys.path`, khiến `models`, `processing`, `metrics` trở thành tên
module toàn cục, dễ va chạm tên. GRKT đã được chuyển sang import tương đối; GKT nên làm tương tự.
- [ ] Đổi sang import tương đối, thêm `pyproject.toml` và cài bằng `pip install -e .`. Notebook chỉ còn
  `!pip install -e /kaggle/working/Knowledge-Tracing`.

### [x] 2.5 🟠 Cấu hình bằng dataclass thay cho 15+ kwargs truyền tay

> **Trạng thái 2026-11-11:** `level1.ExperimentConfig`, `level2.experiment.Level2Config` (ghi vào `summary.json` khoá `config`); `run_dataset` tách thành `_step_*` B1–B8; `dataset_spec.py` 1.086 → 599 dòng (+ `canonicalize.py`, `kaggle_io.py`, re-export tên cũ). Kwargs cũ vẫn chạy.

`run_experiment` lặp lại tham số của `load_kt_data`, `build_model` và `train_one`. `run_dataset` ở
Level 2 dài khoảng 595 dòng (`level2/experiment.py:142-737`).
- [ ] Viết `@dataclass ExperimentConfig` (data, model, train, graph), serialize ra
  `run_config.json` và hash thành fingerprint (dùng luôn cho 2.2).
- [ ] Tách `run_dataset` thành các hàm bước B1…B8, mỗi hàm ≤ 80 dòng, nhận và trả `Context`.
  `dataset_spec.py` (1.086 dòng) nên tách riêng phần chuẩn hoá Junyi thành `datasets/junyi.py`.

### [~] 2.6 🟠 Notebook mỏng hơn nữa và có CLI

> **Trạng thái 2026-11-11:** **Một phần:** có CLI `python -m level1`. **Chưa** làm mỏng notebook: notebook Level 1 còn bản riêng của vòng train resumable, `ShuffledRow`, quy ước checkpoint — cần chạy lại trên Kaggle GPU để kiểm chứng nên để lại. `archive/` chưa dời (cần quyết định của nhóm).

Hai notebook "full" có 3.827 và 3.126 dòng JSON. Nhiều cell chứa logic: resume zip, hợp nhất bảng,
xử lý `sys.modules` cũ.
- [ ] Đưa các logic đó vào package và thêm CLI `python -m kt run --config configs/level1.yaml`.
  Notebook chỉ còn: cài đặt, gọi CLI hoặc hàm, hiển thị.
- [ ] Gom `archive/kaggle_*.ipynb` và `archive/outputs/` vào một release hoặc tag. Hiện thư mục `archive/`
  trùng với `results/` và gây nhầm khi đọc.

### [x] 2.7 🟢 Xoá code chết và vendored không dùng

> **Trạng thái 2026-11-11:** Đã xoá `level1/gikt*.py`, `GRKT/Runner.py`, `GRKT/Training.py`. Giữ VAE/PAM trong `GKT/models.py` (dùng chung nhánh code với MHA) và `GKT/train.py` (đã sửa import + README).

- [ ] `level1/gikt.py`, `level1/gikt_data.py` (không nằm trên đường chạy).
- [ ] `GRKT/Runner.py`, `GRKT/Training.py` (đã được thay bằng `level2/grkt_api.py`, và Runner chia
  k-fold trên cửa sổ, tức là phạm §6.4).
- [ ] Phần VAE/PAM trong `GKT/models.py` nếu không dùng trong báo cáo. **Giữ MHA**: `level1/pipeline.py`
  (`graph_type == "MHA"`) và `level1/explain.py` đang dùng, và `results/level1_out/checkpoints/GKT-MHA-*.pt`
  phụ thuộc vào nó.
- [ ] `GKT/train.py` + `GKT/command.sh` là đường chạy được nhắc trong `GKT/README.md` và đã sinh
  `GKT/outputs/gkt_benchmark_summary.csv`. Nếu xoá thì cập nhật README đó, hoặc chuyển vào `archive/`.

### [ ] 2.8 🟠 Vệ sinh repo và artifact

> **Trạng thái 2026-11-11:** **Chưa làm — cần quyết định của nhóm:** gỡ 57 file `.pt` khỏi git và đổi LFS là thao tác khó hoàn tác, ảnh hưởng notebook đang clone repo. Gộp tài liệu GAP/BAO_CAO cũng cần người viết báo cáo quyết định.

- [ ] Repo đang track **57 file `.pt`, khoảng 71 MB**. `.gitattributes` đẩy `*.json` sang LFS nhưng máy
  không có git-lfs, nên `split.json`/`metrics.json` chỉ là pointer (ghi trong `GAP_LEVEL1.md`). Nên
  hoặc cài git-lfs thật, hoặc **bỏ LFS cho `.json`** (file nhỏ) và đưa `.pt` lên GitHub Release hoặc
  Kaggle Dataset.
- [ ] Docs đang chồng chéo: `GAP_LEVEL{1,2}.md` có nhiều lớp "cập nhật ngày …" chồng lên nhau, cộng thêm
  `BAO_CAO.md`, `KE_HOACH.md` và `docs/0x_*.md`. Nên giữ một file trạng thái hiện hành và đưa lịch sử
  vào git log hoặc `docs/history/`.

### [x] 2.9 🔴 Bộ test tự động (hiện chưa có thư mục `tests/`)

> **Trạng thái 2026-11-11:** `tests/` (21 test: 19 pass + 2 xfail, ~15 s CPU): split rời nhau, split không phụ thuộc đồ thị, đồ thị chỉ từ train (có đối chứng), dự đoán nhân quả DKT/GKT/MHA, Level 2 khớp Level 1, quy ước checkpoint + manifest + fingerprint, resume theo deadline, smoke `run_dataset`. 2 test `xfail(strict)` ghi nhận lỗi 0.1/0.2. Chưa có GitHub Actions.

Nên có test cho các bất biến chống rò rỉ, chạy dưới 1 phút trên CPU với file 15 người học:
- [ ] `train_users ∩ test_users = ∅`, và split giữa Level 1 và Level 2 trùng nhau.
- [ ] Đồ thị chỉ phụ thuộc train: thay đổi chuỗi của một người học test thì đồ thị **không đổi**.
- [ ] Không có hai bước liên tiếp cùng `order_id` (bắt lỗi 0.1); `order` tăng dần trong từng người học
  (bắt lỗi 0.2).
- [ ] Dịch nhãn đúng: dự đoán ở vị trí *t* không thay đổi khi thay nhãn ở *t* hoặc *t+1*, chỉ thay đổi
  khi thay nhãn ở *< t*. Đây là test nhân quả trực tiếp trên mô hình.
- [ ] Smoke test: train 1 epoch mỗi mô hình, loss hữu hạn, checkpoint nạp lại được.
- [ ] Chạy test trên GitHub Actions (CPU) cho mỗi lần push.

---

## Mức 3: Giảm thời gian build/train

> Thời gian thực đo của từng bước chưa được ghi lại (cột `seconds=0.0` trong `level2_all_datasets.csv`).
> **Việc đầu tiên** là profile: ghi `time.perf_counter()` cho từng bước và lưu vào `run_manifest.json`.
> Các mức tăng tốc dưới đây chỉ là ước tính.

- [ ] **3.1** 🔴 **Đọc CSV và tiền xử lý một lần, cache lại.** Mỗi biến thể ablation lại gọi `load_kt_data`,
  tức đọc lại 525k dòng, chạy lambda filter và vòng Python `get_data` (cell 3.4 và vòng đa seed tạo
  bundle mới cho từng loại đồ thị). Split không đổi, chỉ đồ thị đổi. Nên dựng bundle **một lần** rồi
  `bundle.with_graph(graph_type)`. Cache `InteractionTable` (2.1) ra parquet hoặc npz.
- [ ] **3.2** 🟠 Vector hoá `build_transition_graph`: thay vòng lặp đôi Python bằng
  `np.add.at(G, (q[:-1], q[1:]), 1)` trên mảng nối của các người học train.
- [ ] **3.3** 🟠 **Junyi**: chuẩn hoá mất 15–30 phút mỗi phiên Kaggle (`kaggle_level2` §0.3). Nên upload
  `junyi_canonical.parquet` thành Kaggle Dataset riêng (private) và chỉ chuẩn hoá lại khi hash cấu hình
  thay đổi. Parquet cũng nhỏ và đọc nhanh hơn CSV nhiều lần.
- [ ] **3.4** 🟠 **Giảm số lần đồng bộ GPU↔CPU mỗi batch** trong `_epoch_loop`:
  - `torch.isfinite(p.grad).all().item()` được gọi **cho từng tham số** (`pipeline.py:183`), mỗi lần là
    một lần sync. Thay bằng `total = clip_grad_norm_(...)` rồi `if torch.isfinite(total)` (một lần sync),
    hoặc dùng `error_if_nonfinite`.
  - `float(loss.cpu())` và `.cpu().numpy()` mỗi batch: nên cộng dồn tensor trên GPU và chuyển về CPU
    một lần ở cuối epoch.
- [ ] **3.5** 🟠 **Giảm padding bằng bucketing theo độ dài.** GKT và GRKT đều lặp Python qua từng bước
  thời gian (`GKT/models.py:456`, `GRKT/model/GRKT.py:471`), nên chi phí tỉ lệ với **độ dài lớn nhất
  của batch**. Dùng sampler gom các chuỗi có độ dài gần nhau (sort trong cửa sổ, ví dụ 50×batch, rồi
  shuffle các batch). Theo ước tính, cách này giảm được 30–50% số bước rỗng.
- [ ] **3.6** 🟠 **Tối ưu bước thời gian của GKT** (`_aggregate`, `GKT/models.py:158-197`):
  - `x_idx_mat`, `self.emb_x(x_idx_mat)` và `concept_idx_mat` được tạo lại ở **mỗi bước**, dù không phụ
    thuộc *t*. Nên tính một lần mỗi forward.
  - Phép `one_hot @ x_embedding` tương đương `self.emb_x(xt)`, tức chỉ là tra embedding thay cho nhân
    ma trận dày `[B, 2K] × [2K, E]`.
  - `_agg_neighbors` dùng `repeat(1, K, 1)`, tạo tensor `[B, K, 2(H+E)]` ở mỗi bước. Có thể tách lớp
    tuyến tính trên phần nối thành hai lớp `W_self·h_self + W_nb·h_nb` rồi broadcast, đỡ phải cấp phát.
- [ ] **3.7** 🟢 `torch.autocast("cuda", dtype=torch.float16)` cùng `GradScaler` trên T4: tăng tốc khoảng
  1.5–2× với GRU/Linear. Cần kiểm tra AUC không đổi (sai lệch < 1e-3).
- [ ] **3.8** 🟢 DataLoader: `num_workers=2`, `pin_memory=True`, `persistent_workers=True`. Tiền xử lý
  sẵn tensor thay vì `torch.LongTensor(list)` trong `pad_collate` ở mỗi batch.
- [ ] **3.9** 🟢 Mô hình nhỏ (hidden 32, batch 32) không tận dụng hết T4. Có thể tăng batch lên 128 và
  scale lr theo, **hoặc** chạy song song nhiều seed hay biến thể trong cùng một tiến trình bằng
  `torch.func.vmap`/`stack_module_state`, hoặc nhiều tiến trình trên một GPU.
- [ ] **3.10** 🟢 Level 2 train lại DKT/GKT trên assist2009 (vì checkpoint Level 1 từng bị ghi đè). Sau
  khi có 1.4 (tên theo seed và manifest có fingerprint), hãy **nạp lại** checkpoint Level 1 thay vì train
  lại.
- [ ] **3.11** 🟢 `torch.compile` cho GRKT: lợi ích không chắc chắn vì có vòng Python theo *t*. Chỉ
  nên thử sau 3.5–3.7.

---

## Mức 4: Nâng chất lượng học thuật

### 4.A Thiết kế đánh giá

- [ ] **4.1** 🔴 **Khoảng tin cậy theo người học, không chỉ theo seed.** Độ lệch chuẩn theo 3–5 seed chỉ đo
  phương sai của khởi tạo, chưa đo phương sai do mẫu người học test. Cần thêm:
  - **Bootstrap theo người học** (lấy mẫu lại người học test 1.000 lần, tính AUC) để có CI 95%.
  - **DeLong test** (DeLong et al., 1988) cho cặp AUC trên cùng tập test, hoặc paired bootstrap.
  - Với nhiều phép so sánh (nhiều biến thể đồ thị), hiệu chỉnh Holm–Bonferroni.
  `results/level1_out/tables/paired_tests.csv` đã có t-test theo seed; cần ghi rõ nó kiểm định điều gì.
- [ ] **4.2** 🟠 **Cross-validation 5-fold theo người học** như giao thức pyKT (giữ 20% làm test cố định,
  80% còn lại chia 5 fold train/val). Cách này đáng tin hơn một split duy nhất với seed 42, và tương
  thích với yêu cầu §6.5 cho dữ liệu tự thu ở Level 3.
- [ ] **4.3** 🟠 **Baseline mạnh hơn DKT.** Hiện tại chỉ có DKT và các biến thể GKT. Nên thêm ít nhất
  **simpleKT** (Liu et al., ICLR 2023), hoặc **AKT** (Ghosh et al., KDD 2020). Đề bài khuyến nghị
  simpleKT, và đây là các baseline không đồ thị mạnh nhất trong pyKT. Nếu GRKT không thắng simpleKT thì
  kết luận "đồ thị giúp ích" phải được viết lại cho phù hợp.
- [ ] **4.4** 🟢 Phân tích theo nhóm: AUC theo độ dài lịch sử (bucket 1–10, 11–50, …), theo độ phổ biến
  của khái niệm, và với người học có lịch sử ngắn. Đây là "phân tích nâng cao" mà §7 gợi ý.
- [ ] **4.5** 🟢 Bỏ hoặc ghi chú rõ các bài **scaffolding** (`original=0`, đo được 71.402/346.860 lượt
  duy nhất). Đây là câu hỏi phụ do hệ thống sinh ra sau khi người học sai, nên tương quan nhãn rất mạnh.
  Nhiều công trình chỉ giữ `original=1`. Cần chọn cách xử lý và ghi lý do.

### 4.B Ablation đồ thị (vai trò của cấu trúc)

- [ ] **4.6** 🔴 **Identity ≡ Zero theo cấu trúc mô hình.** GKT luôn tách riêng nút tự thân (`f_self` ở
  `_agg_neighbors`), nên `A=I` và `A=0` cho kết quả **giống hệt nhau**. Bảng Level 1 có hai dòng trùng
  số (0.7549) là dấu hiệu của điều này. Nên bỏ Identity khỏi bảng, hoặc giải thích rõ đây là kiểm chứng
  cài đặt (sanity check) chứ không phải một ablation.
- [ ] **4.7** 🟠 Thêm ablation **giữ bậc** (degree-preserving rewiring / configuration model): xáo cạnh
  nhưng giữ nguyên bậc vào và ra của mỗi nút. Nếu Transition ≈ Shuffled (hiện t ≈ 0.4) mà ≫ Zero, thì
  lợi ích đến từ **mật độ hoặc độ chuẩn hoá** chứ không phải từ **cấu trúc**. Giả thuyết này cần được
  kiểm tra tường minh.
- [ ] **4.8** 🟠 So sánh các **cách dựng cạnh** (§7 cột mở rộng):
  - Transition count (hiện có) và Transition có điều kiện đúng/sai (phiên bản trong bài báo GKT,
    Nakagawa et al., WI 2019).
  - Đồng xuất hiện chuẩn hoá theo **PMI/PPMI** (giảm thiên lệch của khái niệm phổ biến).
  - Làm thưa **top-k** hoặc ngưỡng, quét k ∈ {3, 5, 10} để thấy đường cong AUC theo mật độ.
- [ ] **4.9** 🟢 Báo cáo thống kê đồ thị: số cạnh, mật độ, phân phối bậc, tỉ lệ nút cô lập, so với số
  tương tác train. Một phần đã có trong `graph_stats.csv`, cần đưa vào báo cáo.

### 4.C Tiêu chí quỹ đạo tri thức (Level 2)

- [ ] **4.10** 🔴 Có 3/5 tiêu chí (`consistency_strengthen`, `consistency_spread`, `locality`) **bằng 1 theo
  cấu trúc** của GRKT (`GAP_LEVEL2.md` §0, mục 3). Một tiêu chí không thể sai thì không phải bằng chứng
  (không falsifiable). Cần:
  - Đo **cùng tiêu chí trên mastery suy ra từ DKT/GKT**: với DKT, mastery của khái niệm *c* tại *t* là
    `σ(output_c)`. Các mô hình này *có thể* vi phạm tiêu chí, nên phép so sánh có nghĩa, và đồng thời đáp
    ứng mục mở rộng §7 "so sánh quỹ đạo giữa mô hình có đồ thị và không đồ thị".
  - Định nghĩa toán học trong báo cáo cho từng tiêu chí, ví dụ tính nhất quán hướng:
    `DC = P( m_{t+1}(c) ≥ m_t(c) | r_t = 1, c ∈ N(q_t) ∪ {q_t} )`, kèm baseline ngẫu nhiên hoặc hoán vị
    để biết giá trị nào là "tốt".
- [ ] **4.11** 🟠 `precedence_gap` và `stability_*`: báo cáo kèm **CI bootstrap** và **phân phối theo người
  học** (violin plot), không chỉ giá trị trung bình.
- [ ] **4.12** 🟢 **Kiểm định ngoài (external validity)**: mastery dự đoán của khái niệm *c* tại *t* nên
  tương quan với tỉ lệ đúng *thực tế* của người học ở *k* lần làm *c* tiếp theo. Tính Spearman hoặc AUC
  của `m_t(c)` đối với `r_{t'}` khi `q_{t'} = c`. Tiêu chí này có thể sai, nên là bằng chứng mạnh.

### 4.D Calibration và báo cáo

- [ ] **4.13** 🟢 ECE dùng bin **equal-mass** (thay cho equal-width) và báo cáo kèm phân rã Brier
  (reliability, resolution, uncertainty; Murphy, 1973). Cân nhắc temperature scaling trên tập val như
  một ablation calibration.
- [ ] **4.14** 🟢 Mỗi bảng kết quả cần ghi đủ: dataset, số người học/tương tác sau lọc, cách xử lý đa
  khái niệm, max_seq_len, số seed, số tham số, thời gian train. Thêm "pyKT protocol checklist": những
  điểm khớp và lệch so với pyKT.

---

## Mức 5: Mở rộng theo đề (Level 3–4)

- [ ] **5.1** 🔴 **Level 3: xAI ≥ 10 trường hợp có bằng chứng trên đồ thị** cho GRKT:
  - Dùng attribution của GRKT (trọng số retrieval trên `rel_map`, `pre_map`) để liệt kê các lượt làm và
    các cạnh ảnh hưởng nhiều nhất.
  - **Fidelity định lượng**: deletion/insertion curve. Xoá top-k lượt hoặc cạnh được attribution chỉ ra
    rồi đo mức giảm của xác suất, so với xoá ngẫu nhiên. `level1/explain.py` đã có phép phản chứng kiểu
    lật một lượt và bỏ toàn bộ cạnh, có thể mở rộng từ đó.
- [ ] **5.2** 🔴 **Level 3: quiz tự thu** (10–20 người, 20–30 câu, 4–6 khái niệm, ≥ 400 tương tác): thiết
  kế Q-matrix và đồ thị tiên quyết do chuyên gia định nghĩa. Đánh giá transfer (pretrain trên assist2009,
  fine-tune hoặc zero-shot) với CV 5-fold theo người học.
- [ ] **5.3** 🟠 **Level 4 (+1.0)**, các hướng hợp với hiện trạng repo:
  - (a) **Học cấu trúc cạnh**: so đồ thị học được với Transition/PMI. Đây cũng là câu trả lời trực
    tiếp cho kết quả "đồ thị Transition không giúp" của Level 1.
  - (a) **Ràng buộc tiên quyết** đưa vào loss (phạt khi `m(B) > m(A)` với cạnh A→B), đo bằng chính
    `precedence_gap`.
  - (b) GNNExplainer cho GKT và kiểm tra fidelity.
  - Yêu cầu bắt buộc: có giả thuyết, ablation, và chạy trên ≥ 2 tập công khai.

---

## Thứ tự đề xuất

```
0.1 + 0.2 + 0.3 (qua 2.1) ─► 2.9 test ─► 3.1/3.4/3.5 (cho nhanh) ─► chạy lại L1 + L2
        │                                                                  │
        └─► 1.x quick wins làm song song                                   ▼
                                       4.1, 4.3, 4.6, 4.10 (cần cho báo cáo) ─► 5.x
```

Nguyên tắc: **sửa dữ liệu (Mức 0) trước khi tối ưu hay viết báo cáo**, vì mọi con số hiện tại trên
assist2009 sẽ thay đổi sau 0.1–0.3.

## Tài liệu tham khảo

1. Piech et al. *Deep Knowledge Tracing*. NeurIPS 2015.
2. Nakagawa, Iwasawa, Matsuo. *Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network*. WI 2019.
3. Xiong, Zhao, Van Inwegen, Beck. *Going Deeper with Deep Knowledge Tracing*. EDM 2016.
4. Liu et al. *pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models*. NeurIPS 2022 (D&B).
5. Liu et al. *simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing*. ICLR 2023.
6. Ghosh, Heffernan, Lan. *Context-Aware Attentive Knowledge Tracing (AKT)*. KDD 2020.
7. Cui et al. *GRKT*. KDD 2024 (mã nguồn vendored trong `GRKT/`; trích tên bài đầy đủ từ paper gốc khi viết báo cáo).
8. DeLong, DeLong, Clarke-Pearson. *Comparing the Areas under Two or More Correlated ROC Curves*. Biometrics 1988.
9. Murphy. *A New Vector Partition of the Probability Score*. J. Applied Meteorology 1973.
10. Ying et al. *GNNExplainer: Generating Explanations for Graph Neural Networks*. NeurIPS 2019.
