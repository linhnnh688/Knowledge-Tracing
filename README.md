# Knowledge-Tracing

Đồ án *Theo dõi tri thức người học bằng học máy trên đồ thị* (nhánh đúng/sai).

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
5. **Junyi cần nhiều RAM hơn ASSIST.** File gốc `Log_Problem.csv` có hàng chục triệu dòng, nên
   `_canonicalize_junyi` chỉ đọc các cột khớp `JUNYI.columns` (log in ra "đọc N/M cột").
   `MAX_USERS` lấy mẫu SAU khi đã nạp file nên **không** giảm RAM lúc đọc; nếu notebook vẫn bị
   `Killed`, chạy riêng Junyi ở notebook CPU hoặc rút `JUNYI.columns` xuống đúng tên cột thật.
6. Sản phẩm ghi dưới `/kaggle/working/level2_out/<dataset>/`: `split.json`,
   `leak_report.txt`, `graphs_train_only.npz`, `level2_metrics.csv`,
   `trajectories/{traj_*.npz, mastery_*.csv, heatmap_*.png, curves_*.png}`,
   `trajectory_metrics.csv`, `calibration.csv`, `reliability.png`,
   `multiseed_metrics.csv`, `summary.json`, `GRKT-<graph>.pt`; nối nhiều dataset ở
   `level2_all_datasets.csv`.

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
`metrics.json`, `learner_view.json` trong repo **là con trỏ LFS chứ không phải nội dung**
(máy dev này thậm chí không cài `git-lfs`). Vì vậy không code nào được đọc ngược các file đó:
`level2` tái sinh split bằng đúng tham số của `level1.pipeline.load_kt_data`
(`seed=42, train_ratio=0.6, val_ratio=0.2, min_seq_len=3, max_seq_len=200`) và chứng minh nó
khớp bằng `_assert_align` + AUC đường nền ở mục 4 của notebook. Kết quả cấp 2 ghi mới
(`summary.json`, `reliability_rows.csv`, ...) nằm trong `/kaggle/working`, không phụ thuộc LFS.

