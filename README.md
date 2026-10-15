# Knowledge-Tracing

Đồ án *Theo dõi tri thức người học bằng học máy trên đồ thị* (nhánh đúng/sai).

## Cấp độ 1

- Baseline không đồ thị: **DKT** (`GKT/models.py`)
- Mô hình đồ thị: **GKT** (PyTorch, Nakagawa et al.) và **GIKT** port PyTorch (bản gốc TensorFlow 1.12 trong `GIKT/` không chạy trên Kaggle)
- Ablation: GKT `Transition` vs `Identity` vs `Zero`; GIKT có/không đồ thị câu hỏi–kỹ năng
- 5 case đúng + 5 case sai kèm lịch sử và láng giềng trên đồ thị

Huấn luyện được gọi từ package `level1/` (không nhét logic vào notebook).

## Chạy trên Kaggle GPU

1. Push repo lên GitHub.
2. Tạo notebook Kaggle GPU, upload / copy [`kaggle_level1.ipynb`](kaggle_level1.ipynb).
3. Notebook sẽ `git clone` repo này, `sys.path` rồi gọi `level1.run_level1(...)`.
4. `QUICK_RUN=True` dùng `assistment_test15.csv` (smoke test). Tắt flag để train `skill_builder_data.csv` (ASSISTments 2009).

Không `pip install torch` trên Kaggle — image đã có bản GPU.

## Chạy local

```bash
pip install -r requirements.txt
python -c "from level1.pipeline import run_level1; run_level1(data_file='assistment_test15.csv', epochs=2, run_gikt=True)"
```
