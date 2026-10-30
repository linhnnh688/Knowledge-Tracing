# Knowledge-Tracing

Đồ án *Theo dõi tri thức người học bằng học máy trên đồ thị* (nhánh đúng/sai).

## Cấp độ 1

- Baseline không đồ thị: **DKT** (`GKT/models.py`)
- Mô hình đồ thị: **GKT** (PyTorch, Nakagawa et al.)
- Ablation: GKT `Transition` vs `Dense` vs `Identity` vs `Zero` vs `Shuffled`
- 5 case đúng + 5 case sai kèm user_id, timestep, lịch sử và láng giềng trên đồ thị
- Bằng chứng tính toán cho từng case (`level1/explain.annotate_evidence`, chạy tự động trong
  `collect_cases`): **phản chứng nếu–thì** — lật kết quả của từng lần làm bài trong cửa sổ lịch sử
  và đo `Δp` của dự đoán; **ablation đồ thị tại chính case đó** — đặt `A := 0` rồi chạy lại, số
  `Δp` cho biết đồ thị đóng góp bao nhiêu cho *lần dự đoán này* chứ không phải trung bình cả tập;
  và danh sách cạnh mạnh nhất của khái niệm đích kèm thời điểm người học từng chạm vào nó.
  Tắt bằng `collect_cases(..., evidence=False)`; chọn 5 case thuộc 5 khái niệm khác nhau bằng
  `distinct_skills=True` (mặc định).
- Dashboard chọn người học (`level1/dashboard.py`): xác suất dự đoán, lịch sử, trạng thái tri thức (chuẩn $h_t$), xuất `learner_view.json`

Huấn luyện được gọi từ package `level1/` (không nhét logic vào notebook).

## Chạy trên Kaggle GPU

1. Push repo lên GitHub.
2. Tạo notebook Kaggle GPU, upload / copy [`kaggle_level1.ipynb`](kaggle_level1.ipynb).
3. Notebook sẽ `git clone` repo này, `sys.path` rồi gọi tuần tự `load_kt_data` / `run_experiment` / `collect_cases` / `export_learner_view` từ package `level1/`.
4. `QUICK_RUN=True` dùng `assistment_test15.csv` (smoke test, **không dùng cho báo cáo**). `QUICK_RUN=False` (mặc định) train `skill_builder_data.csv` (ASSISTments 2009).
5. `RUN_MULTI_SEED=True` (mặc định) chạy mean±std 3 seed với **split cố định seed=42** cho DKT + GKT-Transition + các ablation.

> GIKT đã được loại khỏi pipeline (`level1.pipeline.run_gikt_experiment` raise
> `NotImplementedError`); cấp độ 1 chuẩn hoá hoàn toàn trên DKT + GKT.

Không `pip install torch` trên Kaggle — image đã có bản GPU. Cell cài đặt của
`kaggle_level2.ipynb` chỉ pip-install gói thật sự thiếu (`importlib.util.find_spec`) nên
**pandas 3.x / numpy 2.x của image mới nhất được giữ nguyên**; package `level1/` và `level2/`
và toàn bộ notebook Cấp độ 2 vẽ hình bằng matplotlib thuần (`axes.hist`, `axes.barh`), không
dùng accessor `Series.hist`/`Series.barh` (pandas đã bỏ). `kaggle_level1.ipynb` đang chạy ổn
định nên **không sửa** — nếu gặp `AttributeError: 'Series' object has no attribute 'hist'` ở
cell EDA của nó, sửa một dòng `user_len.hist(bins=40, ax=ax, ...)` thành
`ax.hist(user_len.to_numpy(), bins=40, ...)`.

## Chạy local

```bash
pip install -r requirements.txt
python -c "from level1.pipeline import run_level1; run_level1(data_file='assistment_test15.csv', epochs=2)"
```

## Kết quả Cấp độ 1 (ASSISTments 2009, split theo người học seed=42)

Số liệu dưới lấy từ artifact đã commit trong `outputs/level1/` (`level1_metrics.csv`,
`multiseed_metrics.csv`, `cases.md`) — chạy full `skill_builder_data.csv`, không phải smoke test.

**Một seed (seed=42):**

| Mô hình | Đồ thị | Test AUC | Test Acc | Test NLL |
|---|---|---|---|---|
| DKT | không dùng | **0.7866** | **0.7340** | **0.5272** |
| GKT-Transition | Transition (train only) | 0.7443 | 0.6953 | 0.5772 |
| GKT-Dense | đầy đủ | 0.7486 | 0.7062 | 0.5629 |
| GKT-Identity | đơn vị | 0.7549 | 0.7082 | 0.5570 |
| GKT-Zero | bỏ hết cạnh | 0.7549 | 0.7082 | 0.5570 |
| GKT-Shuffled | cạnh bị xáo trộn | 0.7354 | 0.6971 | 0.5730 |

**3 seed, split cố định (chỉ đổi khởi tạo + thứ tự batch), mean ± std Test AUC:**

| Mô hình | Test AUC | Test Acc | Test NLL |
|---|---|---|---|
| DKT | **0.7860 ± 0.0006** | 0.7339 ± 0.0008 | 0.5276 ± 0.0004 |
| GKT-Transition | 0.7640 ± 0.0066 | 0.7169 ± 0.0070 | 0.5525 ± 0.0079 |
| GKT-Identity | 0.7551 ± 0.0055 | 0.7054 ± 0.0069 | 0.5580 ± 0.0066 |
| GKT-Zero | 0.7551 ± 0.0055 | 0.7054 ± 0.0069 | 0.5580 ± 0.0066 |
| GKT-Shuffled | 0.7656 ± 0.0008 | 0.7196 ± 0.0018 | 0.5474 ± 0.0006 |

**Ba kết luận trung thực với số liệu:**

1. **DKT > mọi biến thể GKT** (Δ AUC = +0.0220, SE chênh lệch = 0.0039, t ≈ 5.7). On ASSIST09
   với một concept mỗi câu hỏi, baseline không đồ thị vẫn là mô hình mạnh nhất — báo cáo cần nói
   rõ điều này thay vì tô vẽ.
2. **Có đồ thị tốt hơn không đồ thị, nhưng chưa có ý nghĩa thống kê**: Transition − Zero =
   +0.0089 (SE = 0.0050, t ≈ 1.8 với n = 3).
3. **Cấu trúc cạnh *không* được chứng minh**: Shuffled − Transition = +0.0016 (t ≈ 0.4) — phá
   hoàn toàn quan hệ "KC nào dẫn đến KC nào" mà AUC không đổi. Lợi ích ở mục (2) đến từ **mật độ
   hàng xóm / độ lớn của $A$**, không từ hình dạng đồ thị. Đây chính là lý do Cấp độ 2 chuyển sang
   GRKT với `rel_map` + `pre_map` và tiêu chí `consistency_spread`.

Ghi chú kỹ thuật: `GKT-Identity` và `GKT-Zero` cho AUC/Acc trùng khít nhau **đúng theo toán học**,
không phải trùng ngẫu nhiên. Trong `GKT/models.py:137-142`, cột $c \neq q_t$ của
`neigh_features` = `graph[q_t, c]·f(...) + graph[c, q_t]·f(...)`; hàng/cột của $q_t$ trong $A=I$
toàn 0 ở mọi $c \neq q_t$, còn ô $c = q_t$ thì `models.py:168` **ghi đè** bằng `self_features`.
Vậy $A=I$ và $A=0$ cho cùng một biểu thức. `Test Loss` của hai biến thể chỉ lệch ~5e-9
(0.5580415551 vs 0.5580415495) — đó là sai số float của thứ tự cộng, không phải khác biệt mô hình.
Hệ quả: ablation "ma trận đơn vị" mà đề bài liệt kê **không tách được** khỏi ablation "bỏ hết cạnh";
chính vì vậy pipeline thêm biến thể `Shuffled` (cùng số cạnh, đã phá cấu trúc) để có một phép thử
đồ thị thật sự.

## Cấp độ 2 — GRKT (KDD 2024)

Mô hình: **GRKT** — *knowledge retrieval / memory strengthening / learning-forgetting* trên
đồ thị khái niệm (`rel_map` đồng xuất hiện + `pre_map` tiên quyết). Mã nguồn upstream
[JJCui96/GRKT](https://github.com/JJCui96/GRKT) nằm trong `GRKT/`, đã được **package-hoá**
và sửa có kiểm soát — danh sách đầy đủ 7 chỗ sửa ở cell markdown "§1" của
`kaggle_level2.ipynb` và trong `KE_HOACH.md`.

Pipeline thực nghiệm trong package `level2/` (notebook chỉ điều phối):

| Module | Nhiệm vụ |
|---|---|
| `level2/dataset_spec.py` | Một dạng CSV chuẩn hoá duy nhất (`CANONICAL_COLUMNS`) cho mọi tập dữ liệu; ba spec `assist2009` / `assist2009_smoke` / `junyi`, chuẩn hoá Junyi, dò file trong `/kaggle/input`, in cột thật khi tên cột đổi |
| `level2/grkt_data.py` | Đổi bundle Cấp độ 1 → chuỗi GRKT (1-based, trục phút có chặn), **đối chiếu từng người học** với Cấp độ 1, đếm `rel/pre` chỉ từ train, sinh biến thể đồ thị, in `leak_report` |
| `level2/grkt_api.py` | `make_args`/`build_grkt`/`train_grkt`/`evaluate_grkt`/`run_grkt_experiment` — thay `GRKT/Runner.py` (k-fold trên cửa sổ ⇒ rò rỉ người học) |
| `level2/trajectory.py` | Ghi m(u,c,t) (`m_pre`, `m_tilde`, `m_strong`), xuất npz/CSV, heatmap + đường cong, 4 tiêu chí hợp lý |
| `level2/calibration.py` | ECE / ACE / Brier + reliability diagram |
| `level2/baselines.py` | Nạp checkpoint Cấp độ 1 (`outputs/level1/*.pt`) và đánh giá DKT/GKT trên cùng split; train lại nếu checkpoint không khớp |
| `level2/experiment.py` | `run_dataset` / `run_level2`: 8 bước B1→B8 cho một (hoặc nhiều) tập dữ liệu |

### Chạy trên Kaggle GPU

1. Attach dataset **Junyi Academy**: mở
   [junyiacademy/learning-activity-public-dataset-by-junyi-academy](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy),
   bấm *Agree/Subscribe* **trước** (dataset đặt `requiresSubscription=True`, chưa subscribe thì
   `Add input` im lặng không mount gì), rồi `Add input` vào notebook. Bundle ~1.6 GB, bên trong có
   `Log_Problem.csv` (16.217.311 dòng / 72.630 người học), `Info_Content.csv`, `Info_UserData.csv`.
   `level2.dataset_spec.find_kaggle_input()` tự dò file này; không tìm thấy thì
   `list_kaggle_inputs()` in ra đúng những gì đang có trong `/kaggle/input`. Đây là yêu cầu
   "≥ 2 tập dữ liệu công khai" của đề bài.
2. Upload/copy [`kaggle_level2.ipynb`](kaggle_level2.ipynb). Notebook tự `git clone` repo.
3. `QUICK_RUN=True`: tập `assist2009_smoke` (15 người học, dùng lại file smoke test của
   Cấp độ 1), 2 epoch, chỉ đồ thị `train` — kiểm tra mã trong vài phút, **không dùng số liệu
   cho báo cáo**. `MAX_USERS` chỉ có tác dụng với Junyi (tập phải chuẩn hoá từ file gốc).
4. `QUICK_RUN=False`: `assist2009` đầy đủ + `junyi` (cap 4000 người), GRKT với đồ thị `train`,
   quỹ đạo mastery + tiêu chí hợp lý, `MAX_SEQ_LEN=200` (đúng cấu hình Cấp độ 1 để checkpoint
   DKT/GKT nạp lại được đánh giá trên cùng cửa sổ). Ba công tắc ngoài phạm vi yêu cầu Cấp độ 2
   — `RUN_GRAPH_ABLATIONS`, `RUN_CALIBRATION`, `RUN_MULTI_SEED` — **mặc định `False`**: ablation
   cấu trúc đồ thị và đa seed là yêu cầu Cấp độ 1 (đã có số liệu), còn đề bài chỉ yêu cầu chọn
   **một** phân tích nâng cao và nhóm chọn quỹ đạo mastery. Bật lại khi session còn thời lượng
   (mỗi biến thể đồ thị là một lần train GRKT riêng, tốn gấp ~4 thời gian). GRKT nặng hơn GKT
   nhiều lần; nếu hết giờ, giảm `EPOCHS` trước khi giảm `MAX_SEQ_LEN`, và **không** đổi
   `SPLIT_SEED` (seed split phải giữ 42 để cùng một split với Cấp độ 1).
5. **Junyi cần nhiều RAM hơn ASSIST.** File gốc `Log_Problem.csv` có hàng chục triệu dòng, nên
   `_canonicalize_junyi` chỉ đọc các cột khớp `JUNYI.columns` (log in ra "đọc N/M cột").
   `MAX_USERS` lấy mẫu SAU khi đã nạp file nên **không** giảm RAM lúc đọc; nếu notebook vẫn bị
   `Killed`, chạy riêng Junyi ở notebook CPU hoặc rút `JUNYI.columns` xuống đúng tên cột thật.
6. Sản phẩm ghi dưới `/kaggle/working/level2_out/<dataset>/`: `split.json`,
   `leak_report.txt`, `graphs_train_only.npz`, `level2_metrics.csv`,
   `trajectories/{traj_*.npz, mastery_*.csv, heatmap_*.png, curves_*.png}`,
   `trajectory_metrics.csv`, `summary.json`, `GRKT-<graph>.pt`; nối nhiều dataset ở
   `level2_all_datasets.csv`. Cấu hình mặc định không sinh `calibration.csv`, `reliability.png`
   và `multiseed_metrics.csv` (hai bước cuối tắt) — chỉ có khi bật `RUN_CALIBRATION` /
   `RUN_MULTI_SEED`.
7. **Cell §8.5** gói cả cây `level2_out/` thành `/kaggle/working/level2_output.zip` để tải về
   một lần từ tab Output (Kaggle xoá `/kaggle/working` khi session kết thúc). Mặc định bỏ
   `junyi_canonical.csv` (tái sinh được) và file > `MAX_MB=150`; đặt `KEEP_CKPTS=False` nếu chỉ
   cần số liệu + ảnh. Chạy lại cell bất cứ lúc nào để gói bổ sung — zip cũ bị ghi đè.
8. **Cứu một lần chạy đã train xong nhưng chết ở bước sau:** B4 lưu `GRKT-<graph>.pt` vào
   `SAVE_DIR/<dataset>/` NGAY khi mỗi biến thể train xong, và toàn bộ bước vẽ hình
   (heatmap/đường cong/reliability) đã bọc trong `_safe_plot` — lỗi matplotlib chỉ in cảnh báo,
   không làm mất `result`. Nếu vẫn phải chạy lại cell §3 trong cùng session, đặt
   `RESUME_GRKT_CKPT=True`: B4 nạp trọng số cũ thay vì train lại (tiết kiệm hàng giờ GPU Kaggle),
   các bước B5→B8 chạy bình thường. **Chỉ bật khi không đổi** `SPLIT_SEED`, `VARIANTS`,
   `COLD_THRESH`, `GRAPH_THRESH`, `D_HIDDEN` so với lần đã train: `load_state_dict(strict=True)`
   phát hiện khác kiến trúc nhưng KHÔNG phát hiện khác đồ thị hay khác split.
   (`torch.load(..., weights_only=True)` — checkpoint chỉ chứa tensor.)

### Kiểm tra đối chiếu bắt buộc (cell §4)

Đường nền DKT chạy lại trên split mà Cấp độ 2 dùng phải tái hiện AUC **0.7866** đã công bố
ở Cấp độ 1 (dung sai < 0.005, với `MAX_SEQ_LEN=200`). Chênh lệch lớn hơn nghĩa là
`MAX_SEQ_LEN`/`min_seq_len`/file CSV đã lệch — bảng so sánh GRKT khi đó vô nghĩa. Chế độ
`QUICK_RUN` dùng `assist2009_smoke` nên cell này tự in thông báo "không kiểm tra được đối
chiếu"; chỉ số chỉ có giá trị sau lần chạy `QUICK_RUN=False`.

### Chạy local

```bash
python -c "from level2.experiment import run_dataset; run_dataset('assist2009_smoke', save_root='outputs/level2', epochs=1, batch_size=16, max_seq_len=50, variants=('train',), run_baselines=False, record_trajectory=True, run_calibration=True)"
```

Dùng `assist2009_smoke` (15 người học) cho lần chạy thử; bỏ `run_baselines=False` khi muốn so
với DKT/GKT. Đổi sang `'assist2009'` / `'junyi'` cho số liệu báo cáo.

## Định vị giao thức so với pyKT

Đề bài tham chiếu pyKT (`pykt-toolkit`), nên bảng dưới liệt kê **chỗ giống** và **chỗ lệch có
chủ đích** của giao thức nhóm đang dùng — cần khai cả hai trong mục "định vị giao thức" của báo
cáo, không nói chung chung là "làm theo pyKT".

| Khía cạnh | Nhóm làm | Quan hệ với pyKT |
|---|---|---|
| Chia dữ liệu | theo **người học**, `seed=42`, tỉ lệ `0.6/0.2/0.2`, mỗi người học là MỘT chuỗi | **Khớp** nguyên tắc: không người học nào xuất hiện ở hai tập. pyKT cũng tách theo user (kết hợp fold cố định theo từng dataset). |
| Ngưỡng chuỗi tối thiểu | `min_seq_len=3` (`GKT/processing.load_dataset`) | **Lệch**: pyKT hay dùng ngưỡng cao hơn (≥5) cho ASSISTments. Giữ 3 vì đổi ngưỡng là đổi **tập người học**, tức đổi split — sẽ vô hiệu checkpoint DKT/GKT đã công bố ở Cấp độ 1 và làm mục đối chiếu §4 vô nghĩa. |
| Cắt chuỗi | `max_seq_len=200` cho cả ba mô hình | **Khớp** (pyKT cũng cắt theo trần độ dài). Đây cũng là lý do `MAX_SEQ_LEN` ở Cấp độ 2 phải đúng bằng 200. |
| Đồ thị khái niệm | transition/đồng xuất hiện + `rel_map`/`pre_map` của GRKT **chỉ đếm từ tập train** | **Khớp**, và là điểm chống rò chính. Bản GRKT gốc nạp `assist09.bin` vốn dựng đồ thị trên toàn bộ dữ liệu — đã bỏ đường đó (`ALLOWED_DATASETS` trong `GRKT/Training.py`). |
| Nhãn / đặc trưng | input tại bước `t` không chứa `r_t`; nhãn là tương tác `t+1` (shift 1 bước), `filt` loại vị trí padding | **Khớp** công thức dự đoán bước tiếp theo. |
| AUC | gom **toàn bộ** prediction của test set rồi tính MỘT `roc_auc_score` (`level1.pipeline._epoch_loop`, `level2.grkt_api._eval_items`) | **Khớp** — pyKT báo cáo AUC toàn cục, không phải trung bình theo batch (trung bình batch là cách Cấp độ 1 từng làm và đã sửa). |
| Q-matrix / đa khái niệm | một dòng = một `skill_id`; dòng có `skill_id` rỗng bị **loại** (`dropna`) chứ không nhân bản theo nhiều KC | **Lệch**: pyKT dùng Q-matrix `question → {KC}` cho câu hỏi đa kỹ năng. File ASSISTments đã commit (`skill_builder_data.csv`, `assistment_test15.csv`) có đúng MỘT `skill_id` mỗi dòng, và `spec.max_concepts=1`, nên không có thông tin để explode. Khi nào có Q-matrix thật thì phải thêm cột danh sách kỹ năng và sửa `build_grkt_data._assert_align` (hiện raise `NotImplementedError` cho `max_concepts > 1`). |
| Từ điển chỉ số | `pd.factorize(skill_id, sort=True)` / `factorize(problem_id)` trên **toàn file** | **Lệch về mặt chữ, không rò rỉ**: đây chỉ là ánh xạ nhãn → chỉ số, không ước lượng thống kê hay dùng nhãn nào. Đổi lại nó đảm bảo một khái niệm có cùng chỉ số ở cả ba tập, điều kiện để checkpoint Cấp độ 1 nạp được ở Cấp độ 2. |
| Nhãn đúng/sai Junyi | không có cột `correct` thì suy từ `answer_count`: 1 → đúng, ≥3 → sai, **bỏ** giá trị 2 | **Khớp** quy ước tiền xử lý của pyKT cho dữ liệu nhiều lần thử; nguồn nhãn luôn được in ra (`correct_origin`). |
| Trục thời gian của GRKT | phút cộng dồn **theo từng người học**, chặn 240 phút/bước, `alpha=0.01` | **Lệch** với preprocess gốc của GRKT (dùng mốc thời gian thô): `exp(-(n+1)·Δt·k)` với Δt thô underflow về 0 ⇒ mô hình "quên sạch" sau một bước. Lý do ở `KE_HOACH.md`, bảng khai báo chỉnh sửa. |

Hai khác biệt còn lại nằm ở phạm vi thực nghiệm, không ở giao thức: Cấp độ 2 dùng **một split
duy nhất** thay vì k-fold trên cửa sổ như `GRKT/Runner.py` (k-fold trên cửa sổ đặt cùng một người
học vào cả train lẫn test), và chạy GRKT với `k_hop=1`, `k_hidden=16` cho vừa ngân sách GPU T4
thay vì cấu hình lớn của bài báo.

## Lưu ý về file JSON đã commit

`.gitattributes` chuyển toàn bộ `*.json` sang Git LFS, nên `outputs/level1/split.json`,
`metrics.json`, `learner_view.json` trong repo **là con trỏ LFS chứ không phải nội dung**
(máy dev này thậm chí không cài `git-lfs`). Vì vậy không code nào được đọc ngược các file đó:
`level2` tái sinh split bằng đúng tham số của `level1.pipeline.load_kt_data`
(`seed=42, train_ratio=0.6, val_ratio=0.2, min_seq_len=3, max_seq_len=200`) và chứng minh nó
khớp bằng `_assert_align` + AUC đường nền ở mục 4 của notebook. Kết quả cấp 2 ghi mới
(`summary.json`, `reliability_rows.csv`, ...) nằm trong `/kaggle/working`, không phụ thuộc LFS.

