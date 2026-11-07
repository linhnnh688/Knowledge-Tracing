# Tài liệu Kiến trúc Mã nguồn & Thực nghiệm Knowledge Tracing

Tài liệu này cung cấp hướng dẫn toàn diện, có cấu trúc và chi tiết về toàn bộ mã nguồn của dự án **Knowledge Tracing (KT)** trong kho mã nguồn này, bao gồm:
1. Hai mô hình nền tảng: **GKT** (Graph-based Knowledge Tracing, Nakagawa et al., 2019) và **GRKT** (Graph-based Retrieval-Knowledge Tracing, Cui et al., KDD 2024).
2. Hai gói mã nguồn điều phối thực nghiệm: [`level1/`](file:///root/Knowledge-Tracing/level1) và [`level2/`](file:///root/Knowledge-Tracing/level2).
3. Hai notebook thực nghiệm đầy đủ trên môi trường Kaggle: [`kaggle_level1_full.ipynb`](file:///root/Knowledge-Tracing/kaggle_level1_full.ipynb) và [`kaggle_level2_full.ipynb`](file:///root/Knowledge-Tracing/kaggle_level2_full.ipynb).

---

## 1. Sơ đồ Cấu trúc Hệ thống

```
Knowledge-Tracing/
├── GKT/                                # Mã nguồn upstream GKT & DKT (Nakagawa et al., 2019)
│   ├── models.py                       # Các lớp GKT, DKT, MultiHeadAttention, VAE
│   ├── layers.py                       # MLP, EraseAddGate, Attention, Encoder/Decoder
│   ├── processing.py                   # Đọc dữ liệu, chia split, dựng đồ thị Transition
│   ├── metrics.py                      # KTLoss (NLLLoss), VAELoss
│   ├── utils.py                        # Xử lý đồ thị, Gumbel-Softmax, phân phối Gaussian
│   └── train.py                        # CLI huấn luyện upstream
│
├── GRKT/                               # Mã nguồn upstream GRKT (Cui et al., KDD 2024 - đã patched)
│   ├── model/GRKT.py                   # Mô hình GRKT 4 pha (Retrieval, Strengthen, Forgetting)
│   ├── BaseModel.py                    # Khung sườn huấn luyện và đánh giá epoch
│   ├── Runner.py                       # Vòng lặp K-fold upstream (lưu ý rò rỉ cửa sổ)
│   ├── Training.py                     # Entry point CLI upstream
│   ├── metric.py                       # AUC, ACC
│   ├── utils.py                        # feature2tensor, seed, padding
│   └── data/assist09/                  # Script tiền xử lý và sinh assist09.bin
│
├── level1/                             # Module điều phối Cấp độ 1 (bọc GKT, sửa lỗi, xAI)
│   ├── paths.py                        # Quản lý đường dẫn, chèn GKT vào sys.path
│   ├── pipeline.py                     # load_kt_data, build_model, train_one, run_experiment
│   ├── explain.py                      # 10 case study, phản chứng (counterfactual flips, ablation)
│   └── dashboard.py                    # Trích xuất trạng thái ẩn h_t, xuất learner_view.json
│
├── level2/                             # Module điều phối Cấp độ 2 (bọc GRKT, multi-dataset, quỹ đạo)
│   ├── dataset_spec.py                 # Hợp đồng chuẩn hóa 8 cột, chunked parser Junyi, .meta.json
│   ├── grkt_data.py                    # Dựng chuỗi 1-based, build_relevance_maps (train only), leak_report
│   ├── grkt_api.py                     # Thay thế Runner.py, train_grkt, checkpoint fingerprint, resume
│   ├── baselines.py                    # Chạy DKT/GKT trên cùng split, quản lý checkpoint_seed
│   ├── trajectory.py                   # Thu thập quỹ đạo mastery m(u,c,t), tiêu chí thật vs kiểm chứng
│   ├── calibration.py                  # Đo hiệu chỉnh ECE, ACE, Brier score, reliability diagram
│   ├── archive.py                      # Đóng gói zip tự chứa, luồng nền PeriodicPackager chống timeout 12h
│   └── experiment.py                   # Điều phối toàn bộ thực nghiệm (run_dataset, run_level2)
│
├── kaggle_level1_full.ipynb            # Notebook Cấp độ 1 đầy đủ: 5 seed, ablation, xAI, calibration
├── kaggle_level2_full.ipynb            # Notebook Cấp độ 2 đầy đủ: ASSIST09 + Junyi tập con, quỹ đạo, dashboard
└── docs/                               # Thư mục tài liệu giải thích chi tiết
    ├── README.md                       # Bản đồ tổng quan (file hiện tại)
    ├── 01_gkt_module.md                # Giải thích chi tiết module GKT/
    ├── 02_grkt_module.md                # Giải thích chi tiết module GRKT/
    ├── 03_level1_module.md              # Giải thích chi tiết module level1/
    ├── 04_level2_module.md              # Giải thích chi tiết module level2/
    ├── 05_kaggle_level1_notebook.md     # Giải thích chi tiết notebook kaggle_level1_full.ipynb
    └── 06_kaggle_level2_notebook.md     # Giải thích chi tiết notebook kaggle_level2_full.ipynb
```

---

## 2. Bản đồ Tài liệu Chi tiết

Mỗi module và notebook được phân tích sâu trong từng tệp tài liệu chuyên trách:

| Tệp tài liệu | Nội dung trọng tâm | Đường dẫn tham chiếu |
|---|---|---|
| [**01_gkt_module.md**](file:///root/Knowledge-Tracing/docs/01_gkt_module.md) | Giải thích cấu trúc mô hình GKT (2019), DKT (baseline LSTM), các biến thể đồ thị, cơ chế lan truyền GNN + GRU, Erase-Add Gate, và danh mục 11 lỗi/nhược điểm trong mã nguồn upstream. | [`GKT/`](file:///root/Knowledge-Tracing/GKT) |
| [**02_grkt_module.md**](file:///root/Knowledge-Tracing/docs/02_grkt_module.md) | Giải thích thuật toán GRKT (KDD 2024), cấu trúc 4 pha (Tính trước $\rightarrow$ Truy xuất Retrieval $\rightarrow$ Củng cố Strengthening $\rightarrow$ Chuyển bước Transition theo thời gian $\Delta t$), đồ thị $rel\_map$ & $pre\_map$, và 15 điểm rò rỉ/gotchas của upstream. | [`GRKT/`](file:///root/Knowledge-Tracing/GRKT) |
| [**03_level1_module.md**](file:///root/Knowledge-Tracing/docs/03_level1_module.md) | Phân tích package `level1`: chuẩn hóa nạp dữ liệu, cố định split người học seed 42, sửa lỗi $res\_len=2$ của GKT, tính AUC toàn cục, xAI phản chứng (flip, ablation $A:=0$, occlusion), và dashboard chuẩn Frobenius norm. | [`level1/`](file:///root/Knowledge-Tracing/level1) |
| [**04_level2_module.md**](file:///root/Knowledge-Tracing/docs/04_level2_module.md) | Phân tích package `level2`: hợp đồng 8 cột CSV, bộ đếm $rel/pre$ train-only, `grkt_api` thay thế Runner, đối chiếu split chặt chẽ (`_assert_align`), đo đạc quỹ đạo mastery $m(u,c,t)$, ECE/Brier calibration, và luồng đóng gói tự động `PeriodicPackager`. | [`level2/`](file:///root/Knowledge-Tracing/level2) |
| [**05_kaggle_level1_notebook.md**](file:///root/Knowledge-Tracing/docs/05_kaggle_level1_notebook.md) | Hướng dẫn và giải thích từng cell trong `kaggle_level1_full.ipynb`: vòng lặp 5 seed có resume, kiểm định thống kê paired t-test + bootstrap CI 95% + Cohen's $d_z$, ablation suy luận, 10 case study có bằng chứng định lượng, và đóng gói zip kép. | [`kaggle_level1_full.ipynb`](file:///root/Knowledge-Tracing/kaggle_level1_full.ipynb) |
| [**06_kaggle_level2_notebook.md**](file:///root/Knowledge-Tracing/docs/06_kaggle_level2_notebook.md) | Hướng dẫn và giải thích từng cell trong `kaggle_level2_full.ipynb`: cấu hình Junyi tập con 3.000 người, liên kết kiểm chứng với Cấp độ 1, phân tích quỹ đạo mastery (tiêu chí thật vs kiểm chứng cài đặt), dashboard Cấp 2, và cơ chế khôi phục phiên sau 12h. | [`kaggle_level2_full.ipynb`](file:///root/Knowledge-Tracing/kaggle_level2_full.ipynb) |

---

## 3. So sánh Cốt lõi: GKT vs. GRKT

| Khía cạnh | GKT (Nakagawa et al., 2019) | GRKT (Cui et al., KDD 2024) |
|---|---|---|
| **Biểu diễn tri thức** | Ma trận trạng thái ẩn $h_t \in \mathbb{R}^{C \times H}$ (mỗi khái niệm 1 vector ẩn). | Vector mastery $h[c] \in \mathbb{R}^{KH}$ kèm bộ đếm số lần làm bài $count[c]$. |
| **Cơ chế cập nhật** | Mạng hồi quy trên đồ thị: GNN tổng hợp thông tin lân cận $\rightarrow$ Erase-Add Gate $\rightarrow$ GRU cập nhật toàn bộ khái niệm. | 4 pha tường minh: Truy xuất (GNN) $\rightarrow$ Dự đoán $\rightarrow$ Củng cố theo đúng/sai $\rightarrow$ Chuyển bước theo $\Delta t$ (học trễ & quên). |
| **Vai trò thời gian** | **Không sử dụng** mốc thời gian giữa các lần làm bài. | **Trung tâm**: hàm suy giảm lũy thừa $\exp(-(count+1)\Delta t \cdot k)$ cho cả học và quên. |
| **Cấu trúc đồ thị** | 1 ma trận quan hệ $A \in \mathbb{R}^{C \times C}$ (tĩnh: Transition/Dense hoặc học: PAM/MHA/VAE). | 2 ma trận nhị phân: đồng xuất hiện ($rel\_map$) và tiên quyết ($pre\_map$, cùng $sub\_map = pre^T$). |
| **Thời điểm dự đoán** | Feature bước $i$ chứa cả đáp án $r_i$; mô hình cập nhật trạng thái rồi dự đoán $r_{i+1}$. | Dự đoán $score_i$ được tính **trước** khi dùng đáp án $r_i$ để củng cố trạng thái. |
| **Không gian chỉ số** | **0-based**: Khái niệm $0 \dots C-1$, padding $-1$. Feature: $2c + r$. | **1-based**: Index 0 dành cho padding. Khái niệm và Problem ID đều bắt đầu từ 1. |
| **Ưu điểm** | Đơn giản, hội tụ nhanh, nhẹ về bộ nhớ, hỗ trợ đồ thị tự học qua Attention/VAE. | Diễn giải được rõ ràng: đo lường được lượng kiến thức tăng khi đúng, giảm khi sai, và tốc độ quên theo thời gian thực. |
| **Nhược điểm** | Trạng thái bị GRU trộn lẫn khó giải thích trực tiếp mức thành thạo của học sinh. | Tốn nhiều tài nguyên GPU (nhiều phép nhân tensor lớn), rất nhạy với đơn vị đo thời gian $\Delta t$. |

---

## 4. Nguyên tắc Thiết kế & Phòng chống Rò rỉ (Anti-Leakage)

Toàn bộ hệ thống trong repo tuân thủ các nguyên tắc khoa học nghiêm ngặt:

1. **Phân tách người học (User-wise splitting)**:
   - Dữ liệu luôn được chia thành `train / val / test` (tỉ lệ mặc định 0.6 / 0.2 / 0.2) theo **người học** (User ID), không chia theo tương tác rời rạc hoặc cắt cửa sổ trượt.
   - Giao của ba tập luôn bằng rỗng: $\text{train} \cap \text{val} = \text{train} \cap \text{test} = \text{val} \cap \text{test} = \emptyset$.
2. **Đồ thị chỉ xây từ tập huấn luyện (Train-only graphs)**:
   - Cả ma trận chuyển tiếp của GKT (`Transition`) và đồ thị quan hệ của GRKT (`rel_map`, `pre_map`) đều chỉ được tính toán dựa trên các tương tác của người học trong tập `train`. Dữ liệu `val` và `test` hoàn toàn không tham gia vào việc dựng đồ thị.
3. **Độ trễ thời gian hợp lý (Temporal causal consistency)**:
   - Nhãn tương lai không bao giờ bị rò rỉ vào bước hiện tại.
4. **Không trộn cấu hình khi Resume**:
   - Mọi checkpoint và phiên chạy đều lưu kèm `fingerprint` cấu hình (tên tập, số người, siêu tham số, seed). Khi tiếp tục phiên chạy (resume) sau khi hết hạn 12 giờ trên Kaggle, hệ thống kiểm tra đối chiếu chặt chẽ, ngăn chặn tuyệt đối việc trộn lẫn các lần chạy khác nhau.
