# Knowledge-Tracing

Đồ án *Theo dõi tri thức người học bằng học máy trên đồ thị* (nhánh đúng/sai).

## Cấp độ 1

- Baseline không đồ thị: **DKT** (`GKT/models.py`)
- Mô hình đồ thị: **GKT** (PyTorch, Nakagawa et al.) và **GIKT** port PyTorch (bản gốc TensorFlow 1.12 trong `GIKT/` không chạy trên Kaggle)
- Ablation: GKT `Transition` vs `Identity` vs `Zero` vs `Shuffled`; GIKT có/không đồ thị câu hỏi–kỹ năng
- 5 case đúng + 5 case sai kèm user_id, timestep, lịch sử và láng giềng trên đồ thị
- Dashboard chọn người học (`level1/dashboard.py`): xác suất dự đoán, lịch sử, trạng thái tri thức (chuẩn $h_t$), xuất `learner_view.json`

Huấn luyện được gọi từ package `level1/` (không nhét logic vào notebook).

## Chạy trên Kaggle GPU

1. Push repo lên GitHub.
2. Tạo notebook Kaggle GPU, upload / copy [`kaggle_level1.ipynb`](kaggle_level1.ipynb).
3. Notebook sẽ `git clone` repo này, `sys.path` rồi gọi tuần tự `load_kt_data` / `run_experiment` / `collect_cases` / `export_learner_view` từ package `level1/`.
4. `QUICK_RUN=True` dùng `assistment_test15.csv` (smoke test, **không dùng cho báo cáo**). `QUICK_RUN=False` (mặc định) train `skill_builder_data.csv` (ASSISTments 2009).
5. `RUN_MULTI_SEED=True` (mặc định) chạy mean±std 3 seed với **split cố định seed=42** cho DKT + GKT-Transition + các ablation.
6. `RUN_GIKT=True` mới chạy GIKT (mặc định `False` để tiết kiệm tài nguyên).

Không `pip install torch` trên Kaggle — image đã có bản GPU.

## Chạy local

```bash
pip install -r requirements.txt
python -c "from level1.pipeline import run_level1; run_level1(data_file='assistment_test15.csv', epochs=2, run_gikt=False)"
```
