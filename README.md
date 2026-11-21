# Knowledge Tracing bằng học máy trên đồ thị — tái lập GKT và GRKT

Đồ án *Project 02 — Theo dõi tri thức người học bằng học máy trên đồ thị*. Repo tái lập và đánh giá
phản biện hai phương pháp:

- **GKT** — *Graph-based Knowledge Tracing* (Nakagawa và cộng sự, 2019).
- **GRKT** — *Graph-based Reasonable Knowledge Tracing* (Cui và cộng sự, KDD 2024).

Mọi mô hình dùng chung một giao thức chống rò rỉ: chia theo người học, đồ thị chỉ dựng từ tập train,
chuỗi sắp theo `order_id`, và gộp các dòng trùng của câu nhiều khái niệm trên ASSISTments 2009
(bản tiền xử lý `order-dedup-v1`).

## Cấu trúc thư mục

```
.
├── GKT/                 Mã GKT (upstream, đã sửa) + dữ liệu ASSISTments 2009 (GKT/data/)
├── GRKT/                Mã GRKT của tác giả (upstream)
├── level1/              Pipeline Cấp độ 1: dữ liệu, huấn luyện DKT/GKT, xAI, dashboard
├── level2/              Pipeline Cấp độ 2: dữ liệu Junyi, đồ thị rel/pre train-only, GRKT, quỹ đạo, calibration
├── kaggle_level1_full.ipynb   Notebook chạy Cấp độ 1 trên Kaggle (bản sạch, không output)
├── kaggle_level2_full.ipynb   Notebook chạy Cấp độ 2 trên Kaggle (bản sạch, không output)
├── results/
│   ├── level1_out/      Kết quả Cấp độ 1 + kaggle_level1_executed.ipynb (bản đã chạy, có output)
│   └── level2_moderate/ Kết quả Cấp độ 2 + kaggle_level2_executed.ipynb (bản đã chạy, có output)
├── docs/                Tài liệu giải thích lý thuyết, mã nguồn và notebook
├── YEU_CAU.md           Yêu cầu đồ án (diễn giải từ đề gốc)
├── Project02_Knowledge_Tracing.pdf   Đề gốc
└── requirements.txt
```

Mỗi thư mục kết quả chứa bảng chỉ số (`tables/` hoặc `*_metrics.csv`), checkpoint của mọi lần chạy,
tệp chia `split.json`, báo cáo chống rò rỉ `leak_report.txt`, cấu hình khoá `run_config.json` và
`report_summary.md` sinh tự động.

## Kết quả chính

Số liệu trên tập test, đọc từ `results/`. Kiểm định thống kê đầy đủ có trong `results/level1_out/tables/paired_tests.csv` và
`results/level2_moderate/assist2009/multiseed_paired.csv`.

**Cấp độ 1 — ASSISTments 2009, 3 seed (mean ± std)**

| Mô hình | AUC | ACC | NLL |
|---|---|---|---|
| **DKT** | **0,7608 ± 0,0005** | **0,7228 ± 0,0007** | **0,5500 ± 0,0004** |
| GKT-Transition | 0,7358 ± 0,0010 | 0,7081 ± 0,0009 | 0,5720 ± 0,0002 |
| GKT-Shuffled | 0,7290 ± 0,0045 | 0,7069 ± 0,0025 | 0,5764 ± 0,0030 |
| GKT-Zero | 0,7183 ± 0,0027 | 0,6952 ± 0,0037 | 0,5833 ± 0,0027 |

DKT vượt mọi biến thể GKT (DKT − GKT-Transition = +0,0250, p = 0,001). Đồ thị Transition giúp GKT so
với bỏ cạnh (+0,0175, p = 0,010) nhưng chưa hơn đồ thị xáo cạnh có ý nghĩa (+0,0068, p = 0,19).

**Cấp độ 2 — GRKT (seed 42; dòng GRKT ASSISTments là 3 seed)**

| Mô hình | ASSISTments 2009 AUC | Junyi (tập con 3.000 người) AUC |
|---|---|---|
| **GRKT-train** | **0,7694 ± 0,0006** | **0,7896** |
| DKT | 0,7608 | 0,7501 |
| GRKT-zero (bỏ đồ thị) | 0,7551 ± 0,0011 | — |
| GKT-Transition | 0,7325 | 0,7202 |
| GKT-Zero | 0,7233 | 0,7225 |

Với GRKT, đồ thị đóng góp +0,0143 AUC, nhất quán qua 3 seed (p < 0,001). Không có đồ thị, GRKT thấp
hơn DKT.

## Chạy lại

```bash
pip install -r requirements.txt
```

Thực nghiệm đầy đủ chạy trên Kaggle (GPU T4). Notebook tự clone repo này và tải dữ liệu Git LFS.

**Cấp độ 1** — mở `kaggle_level1_full.ipynb`, đặt `QUICK_RUN = False`. Để tái lập đúng số trong
báo cáo, dùng cấu hình của bản đã chạy (`results/level1_out/kaggle_level1_executed.ipynb`):
`SEEDS = [42, 123, 456]` và `TRAIN_MODELS = ["DKT", *ABLATION_MODELS]` (DKT, GKT-Transition,
GKT-Zero, GKT-Shuffled). Bản sạch mặc định chạy thêm GKT-MHA, Identity, ShuffledRow với 5 seed.

**Cấp độ 2** — thêm dataset Junyi Academy
(`junyiacademy/learning-activity-public-dataset-by-junyi-academy`) vào Input, mở
`kaggle_level2_full.ipynb`, chạy với cấu hình mặc định (`JUNYI_MAX_USERS = 3000`,
`SEEDS = [42, 123, 456]`). Kết quả được đóng gói định kỳ thành `level2_moderate_output.zip`; nếu phiên
Kaggle bị ngắt, gắn zip đó làm Input của phiên mới để resume.

Đặt `QUICK_RUN = True` trong cả hai notebook để chạy thử nhanh trên 15 người học
(`GKT/data/assistment_test15.csv`); số liệu từ chế độ này không dùng cho báo cáo.

Lưu ý: các tệp `.csv` và `.json` được lưu bằng Git LFS — cần `git lfs install` trước khi clone.

## Tài liệu

Xem [`docs/README.md`](docs/README.md) để có lộ trình đọc tài liệu về lý thuyết GKT/GRKT, mã nguồn
`level1/`, `level2/` và từng cell của hai notebook.
