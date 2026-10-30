# Knowledge-Tracing

Đồ án *Theo dõi tri thức người học bằng học máy trên đồ thị* (nhánh đúng/sai).

## Cấu trúc repo

| Đường dẫn | Nội dung |
|---|---|
| `GKT/` | DKT + GKT gốc của Cấp độ 1 (`models.py`, `processing.py`, `metrics.py`) và dữ liệu ASSISTments 2009 |
| `GRKT/` | Mã GRKT của bài báo (KDD 2024), đã package-hoá + sửa có kiểm soát (danh sách ở `kaggle_level2.ipynb` §1) |
| `level1/` | Pipeline Cấp độ 1: `load_kt_data`, `save_split`, `train_one`, `explain`, `dashboard` |
| `level2/` | Pipeline Cấp độ 2: spec dữ liệu, chuỗi GRKT, train/eval, quỹ đạo mastery, calibration, `run_dataset` |
| `outputs/level1/` | Artifact Cấp độ 1 đã commit (metrics, `cases.md`, checkpoint `.pt`) |
| `kaggle_level1.ipynb`, `kaggle_level2.ipynb` | Notebook điều phối trên Kaggle GPU — chỉ gọi hàm trong package |
| `KE_HOACH.md`, `YEU_CAU.md`, `GAP_LEVEL1.md` | Kế hoạch/tiến độ, đề bài viết lại, rà soát khoảng trống Cấp độ 1 |

`outputs/level2/` chỉ sinh ra khi chạy notebook, không commit.

**Trạng thái:** Cấp độ 1 đã chạy xong (artifact trong `outputs/level1/`). Cấp độ 2 đã có mã và
notebook; số liệu chỉ hình thành sau lần chạy `QUICK_RUN=False` trên Kaggle. Mọi sửa đổi phải
**commit + push** trước khi chạy, vì notebook `git clone` từ GitHub chứ không dùng file local.

## Cấp độ 1

- Baseline không đồ thị: **DKT** (`GKT/models.py`)
- Mô hình đồ thị: **GKT** (PyTorch, Nakagawa et al.)
- Ablation: GKT `Transition` vs `Dense` vs `Identity` vs `Zero` vs `Shuffled`
- 5 case đúng + 5 case sai kèm user_id, timestep, lịch sử và láng giềng trên đồ thị
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

Không `pip install torch` trên Kaggle — image đã có bản GPU.

## Chạy local

```bash
pip install -r requirements.txt
python -c "from level1.pipeline import run_level1; run_level1(data_file='assistment_test15.csv', epochs=2)"
```

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

1. Attach dataset **Junyi Academy** (bấm *Agree/Subscribe* trước, rồi `Add input`) — yêu cầu
   "≥ 2 tập dữ liệu công khai" của đề bài.
2. Upload/copy [`kaggle_level2.ipynb`](kaggle_level2.ipynb). Notebook tự `git clone` repo.
3. `QUICK_RUN=True`: tập `assist2009_smoke` (15 người học, dùng lại file smoke test của
   Cấp độ 1), 2 epoch, chỉ đồ thị `train` — kiểm tra mã trong vài phút, **không dùng số liệu
   cho báo cáo**. `MAX_USERS` chỉ có tác dụng với Junyi (tập phải chuẩn hoá từ file gốc).
4. `QUICK_RUN=False`: `assist2009` đầy đủ + `junyi` (cap 4000 người), 4 biến thể đồ thị,
   đa seed `[42, 123, 456]`, `MAX_SEQ_LEN=200` (đúng cấu hình Cấp độ 1 để checkpoint DKT/GKT
   nạp lại được đánh giá trên cùng cửa sổ). GRKT nặng hơn GKT nhiều lần; nếu hết giờ, giảm
   `EPOCHS` và `VARIANTS` trước khi giảm `MAX_SEQ_LEN`, và **không** đổi `SPLIT_SEED`
   (seed split phải giữ 42 để cùng một split với Cấp độ 1).
5. Sản phẩm ghi dưới `/kaggle/working/level2_out/<dataset>/`: `split.json`,
   `leak_report.txt`, `graphs_train_only.npz`, `level2_metrics.csv`,
   `trajectories/{traj_*.npz, mastery_*.csv, heatmap_*.png, curves_*.png}`,
   `trajectory_metrics.csv`, `calibration.csv`, `reliability.png`, `reliability_rows.csv`,
   `multiseed_metrics.csv`, `summary.json`, `GRKT-<graph>.pt`, `baselines/*.pt`
   (chỉ khi phải huấn luyện lại đường nền, ví dụ Junyi); nối nhiều dataset ở
   `level2_all_datasets.csv`.

### Dữ liệu Junyi — ánh xạ cột

Bản Kaggle đang dùng (`Log_Problem.csv`, 3,0 GB, 16,2 triệu dòng) **không có cột tên `exercise`**:

| Cột chuẩn hoá | Cột gốc | Ghi chú |
|---|---|---|
| `user_id` | `uuid` | 72.758 người học |
| `skill_id` | `ucid` | exercise/content id (1.326 khái niệm), join được với `Info_Content.csv` |
| `problem_id` | `upid` | 25.785 câu hỏi |
| `correct` | `is_correct` | đúng ngay lần đầu **và** không dùng gợi ý |
| `timestamp_ms` | `timestamp_TW` | UTC+8, làm tròn 15 phút ⇒ Δt trong một phiên thường = 0 |
| `ms_first_response` | `total_sec_taken` | đơn vị giây, đổi sang ms |

Chuẩn hoá chỉ đọc đúng 6 cột này và in lại ánh xạ đã chọn. `junyi_canonical.csv` được dùng lại
nếu đã tồn tại (đổi `MAX_USERS`/`min_interactions` thì phải xoá file trước khi chạy lại).
Cell §2 của notebook in header thật của file dò được và liệt kê mọi CSV trong `/kaggle/input`,
nên nếu Kaggle đổi schema thì nhìn log là biết ngay cột nào lệch.
Vì timestamp bị làm tròn 15 phút, `mean_abs_delta_learn_forget`/`forget_slope` trên Junyi sẽ
bằng 0 ở phần lớn bước — đó là giới hạn của dữ liệu, không phải lỗi mô hình.

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

## Lưu ý về file JSON đã commit

`.gitattributes` chuyển toàn bộ `*.json` sang Git LFS, nên `outputs/level1/split.json`,
`metrics.json`, `learner_view.json` **trong commit là con trỏ LFS**, không phải nội dung
(máy dev có `git-lfs` nên working tree đã được smudge, nhưng bản clone mới không chắc như vậy).
Vì thế không code nào được đọc ngược các file đó:
`level2` tái sinh split bằng đúng tham số của `level1.pipeline.load_kt_data`
(`seed=42, train_ratio=0.6, val_ratio=0.2, min_seq_len=3, max_seq_len=200`) và chứng minh nó
khớp bằng `_assert_align` + AUC đường nền ở mục 4 của notebook. Kết quả cấp 2 ghi mới
(`summary.json`, `reliability_rows.csv`, ...) nằm trong `/kaggle/working`, không phụ thuộc LFS.

