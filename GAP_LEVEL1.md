# Gap Analysis — `kaggle_level1.ipynb` + package `level1/` so với yêu cầu Cấp độ 1

> Ngày rà soát: 15/10/2026. Đối chiếu với `YEU_CAU.md` (viết lại từ đề cương `Project02_Knowledge_Tracing.pdf`).
> Ghi chú trạng thái repo khi rà: `kaggle_level1.ipynb` và `KE_HOACH.md` đã sửa nhưng **chưa commit/push**; notebook trên Kaggle **clone từ GitHub** nên sẽ chạy bản HEAD cũ.
>
> **Trạng thái cập nhật sau khi sửa code (cùng ngày):** đã fix P0.1 (đa seed split cố định), P0.2 (QUICK_RUN=False), P1.5 (dashboard), P1.6 (user_id/t trong case), P1.7 (AUC global), P1.8 (thêm Shuffled ablation), P2.10 (requirements pin + README sửa). **Còn lại:** P0.3 (commit artifact — cần chạy notebook), P0.4 (push GitHub), P1.9 (định vị pyKT), P2.11 (báo cáo 12–18 trang), P2.12 (chạy full + chứng minh vai trò đồ thị).

---

## 1. Tổng quan hiện trạng

`kaggle_level1.ipynb` (21 cells, không output lưu sẵn) đóng vai trò điều phối:

| Cell | Nội dung |
|---|---|
| Clone + sys.path | Clone `linhnnh688/Knowledge-Tracing` vào `/kaggle/working`, thêm `REPO` và `REPO/GKT` vào `sys.path` |
| Cấu hình | `QUICK_RUN=True` → `assistment_test15.csv` (15 người học), 8 epochs, bs=16, seq=80; ngược lại `skill_builder_data.csv` (ASSIST-2009), 20 epochs |
| EDA | Histogram số tương tác/người học |
| 3.1 | `load_kt_data` (Transition, seed=42) + `save_split → split.json` |
| 3.2 | Huấn luyện **DKT** (baseline không đồ thị) |
| 3.3 | Huấn luyện **GKT-Transition** |
| 3.4 | Ablation GKT với **Dense / Identity / Zero** |
| 3.5 | GIKT (mở rộng) — `RUN_GIKT=False` (bỏ qua) |
| 4 | Bảng tổng hợp AUC/Acc/NLL, biểu đồ loss/AUC theo epoch, biểu đồ cột |
| 4.5 | Đa seed `[42,123,456]` cho DKT + GKT-Transition — `RUN_MULTI_SEED=False` (bỏ qua) |
| 5 | xAI: 5 case đúng + 5 case sai từ GKT-Transition → `cases.md` |
| 6 | Xuất `metrics.csv/json`, `split.json`, `cases.md`, checkpoints |

Package `level1/` (đã xác minh mã):

- `pipeline.py` — `set_seed`, `load_kt_data` (gọi `GKT/processing`), `save_split` (`:89-103`), `build_model`, `train_one` (early stopping theo val AUC), `evaluate`, `run_experiment`, `run_gikt_experiment`, `run_level1` (không được notebook gọi).
- `explain.py` — `collect_cases` (`:38-113`): ghi `y_true/y_prob`, lịch sử 8 bước, top-5 láng giềng đồ thị; `format_cases` (`:116-140`) kèm template `_explain_one`.
- `gikt_data.py`, `gikt.py`, `paths.py` — GIKT PyTorch (tắt mặc định).

**Điểm đã xác minh đúng (không phải gap):**

1. Split theo người học thật: mỗi item dataset = 1 chuỗi người học, `random_split` seed 42 (`GKT/processing.py:132-138`) → không người học nào bị chia đôi.
2. Transition graph chỉ build từ train (`processing.py:150`, `:172-197`).
3. Không rò nhãn tương lai: label dịch 1 bước (`GKT/metrics.py:27`), DKT dự đoán `t+1` (`models.py:489-493`).
4. Metrics AUC/Acc/NLL có thật (`GKT/metrics.py:35-52`).
5. Case study có cả lịch sử lẫn láng giềng đồ thị (`explain.py:81-96`, `:105`).
6. Code đa seed có (mean/std 3 seed) nhưng **đang tắt** và **chỉ 2 mô hình**.

---

## 2. Bảng đối chiếu yêu cầu

### 2.1. Cốt lõi Cấp độ 1

| # | Yêu cầu | Trạng thái | Bằng chứng / vấn đề |
|---|---------|-----------|----------------------|
| L1.1 | ≥1 tập dữ liệu công khai | **THỎA** ⚠️ | ASSIST-2009 có trong repo. Nhưng `QUICK_RUN=True` mặc định → 15 người học, split 9/3/3 → AUC test trên ~3 người là vô nghĩa, **không dùng được cho báo cáo**. |
| L1.2 | Baseline không đồ thị **và** mô hình đồ thị (GKT/GCN/GAT/GraphSAGE) | **THỎA** | DKT (`GKT/models.py:454-526`), GKT (`models.py:18-317`). |
| L1.3 | Ablation: bỏ cạnh **hoặc** ma trận đơn vị | **THỎA** ⚠️ | Identity/Zero/Dense chạy mặc định. **Nhưng Identity ≡ Zero**: `models.py:168` ghi đè ô self → `A=I` và `A=0` cho output giống hệt nhau (số liệu `GKT/outputs/gkt_benchmark_summary.csv` trùng hàng). Thiếu edge-shuffle (tùy chọn). |
| L1.4 | 5 đúng + 5 sai kèm bằng chứng lịch sử + khái niệm liên quan | **MỘT PHẦN** | Code có (`explain.py:116-160`). Thiếu: `user_id`/`t` để truy vết về người học; trạng thái tri thức không hiển thị; `why` là template cố định; chọn theo confidence cao nhất, không đại diện; **`cases.md` không tồn tại trong repo**. |

### 2.2. Chống rò rỉ (§6 — áp dụng mọi cấp)

| # | Yêu cầu | Trạng thái | Bằng chứng / vấn đề |
|---|---------|-----------|----------------------|
| X1 | Chỉ dùng tương tác trước thời điểm dự đoán | **THỎA** | Label dịch 1 bước; input chỉ đến `t`. |
| X2 | Đồ thị xây từ tập train | **THỎA** | `processing.py:150,172-183`; GIKT cũng lọc train (`level1/gikt_data.py:96-103`). |
| X3 | Cùng 1 split cố định cho mọi mô hình + nộp split file hoặc seed+mã | **MỘT PHẦN** | Trong 1 lần chạy: mọi mô hình dùng chung split seed 42. **Nhưng**: (i) `split.json` sinh lúc chạy, **không commit**; (ii) loop đa seed truyền `seed=s` vào `load_kt_data` (`kaggle_level1.ipynb:499-507`) → **split và đồ thị đổi theo seed**, mean±std lẫn lộn phương sai split + phương sai khởi tạo. |
| X4 | Tách người học mới theo người học | **THỎA** | Một item = một người; không người nào vừa train vừa test. |
| X5 | Bám chuẩn pyKT | **MỘT PHẦN** | Đúng nguyên tắc; không dùng pykt-toolkit. `min_seq_len=3` (plan ghi <5); không explode câu nhiều skill (drop NaN `skill_id`, `processing.py:72`); vocab fit trên toàn dữ liệu (`:78`). |

### 2.3. Đánh giá (§7)

| # | Yêu cầu | Trạng thái | Bằng chứng / vấn đề |
|---|---------|-----------|----------------------|
| E1 | AUC, accuracy, BCE/NLL | **THỎA** ⚠️ | Có đủ. **Nhưng `test_auc` = trung bình AUC theo batch** (`pipeline.py:166-172`), không phải 1 AUC global trên toàn test set → ước lượng nhiễu, dễ bị soi khi đối chiếu pyKT/simpleKT. |
| E2 | Mean + std ≥3 seed (**bắt buộc** khi báo cáo) | **CHƯA THỎA** | `RUN_MULTI_SEED=False` (`:478`); cell in "Bỏ qua đánh giá đa seed". Phạm vi chỉ DKT + GKT-Transition; các ablation/single-graph chạy seed=42. Section đa seed chưa có trong HEAD đã commit. |
| E3 | Ablation vai trò đồ thị trong kết quả báo cáo | **MỘT PHẦN** | Chạy mặc định nhưng 1 seed. Số liệu smoke set hiện có: Transition ≈ Identity ≈ Zero (0.55/0.53/0.53; đa seed 0.589/0.591) → **chưa chứng minh được vai trò cấu trúc đồ thị**. |

### 2.4. Sản phẩm nộp (§8)

| # | Yêu cầu | Trạng thái | Bằng chứng / vấn đề |
|---|---------|-----------|----------------------|
| D1 | Mã nguồn + requirements/environment + hướng dẫn chạy | **MỘT PHẦN** | Có `requirements.txt` (7 gói, **không pin version**, không `environment.yml`, thiếu torch). README `:18` nói notebook gọi `run_level1(...)` — **sai**, notebook gọi `load_kt_data`/`run_experiment` từng bước. Thay đổi local chưa push. |
| D2 | Split cố định hoặc mã tạo split + mô tả chống rò rỉ | **MỘT PHẦN** | Mã + seed có; `split.json` **không commit**; mô tả rò rỉ chỉ ở markdown notebook/KE_HOACH. |
| D3 | Báo cáo 12–18 trang + slide + bảng phân công | **CHƯA THỎA** | Không có file báo cáo/slide nào trong repo. |
| D4 | Notebook/dashboard: chọn người học → xác suất, lịch sử, trạng thái tri thức, bằng chứng | **CHƯA THỎA** | 0 hit cho `Dropdown/widgets/interact/streamlit/gradio`. Chỉ có `cases.md` tĩnh; không export trạng thái `ht` của GKT (`models.py:297-308` giữ nội bộ); không có `user_id` để chọn. |
| D5 | Phân tích chi tiết 5 đúng + 5 sai | **MỘT PHẦN** | Code có nhưng chưa chạy/commit artifact; giải thích là template. |

### 2.5. Ghi chú ngoài phạm vi Cấp độ 1

- Chưa có code `level2/`/`level3/` (chỉ có kế hoạch trong `KE_HOACH.md:231-245`).
- `KE_HOACH.md:225` đánh dấu đa seed `[x]` và `:229` nói hai notebook "xuất toàn bộ kết quả Cấp độ 1" — **không đúng thực tế**: `multiseed_metrics.csv`, `cases.md`, `split.json` đều không có trong tree; chỉ có `GKT/outputs/gkt_*_summary.csv` từ smoke run.

---

## 3. Việc cần làm, ưu tiên từng mức

### P0 — chặn điểm (bắt buộc làm trước)

1. **Bật và mở rộng đa seed**
   - Đặt `RUN_MULTI_SEED=True` (`kaggle_level1.ipynb:478`).
   - Thêm `GKT-Dense/Identity/Zero` vào `multi_configs` (`:485-488`) vì ablation là khiếu nại bắt buộc.
   - **Sửa confound split**: load split **một lần** với `split_seed=42`, tái sử dụng `bundle` cho mọi seed; chỉ đổi seed huấn luyện trong `run_experiment`. Tức: cho phép truyền split/`bundle` có sẵn vào `pipeline.py:262-273`; trong notebook `:499-507` ngừng truyền `seed=s` vào `load_kt_data`.
   - Commit `multiseed_metrics.csv` kết quả.

2. **Chạy kết quả báo cáo trên dữ liệu đầy đủ**
   - `QUICK_RUN=False` (ASSIST-2009 full). QUICK_RUN chỉ dùng kiểm thử, không đưa vào báo cáo.

3. **Commit artifact**
   - `outputs/level1/{split.json, metrics.csv, metrics.json, cases.md, multiseed_metrics.csv}` (`.gitignore` đang không chặn `outputs/`).

4. **Đồng bộ remote**
   - Push thay đổi local (hoặc ngừng phụ thuộc clone GitHub trong notebook phân tích) — nếu không, Kaggle chạy bản HEAD cũ (không đa seed, `RUN_GIKT=True`).

### P1 — bắt buộc theo đề

5. **Dashboard chọn người học** (§8)
   - Bước tối thiểu: (i) thêm `user_id` vào `KTDataset.__getitem__` và `pad_collate` (`GKT/processing.py:23-38`) hoặc dùng `Subset.indices` + `user_ids`; (ii) chạy test loader theo từng người học, lưu `p̂` từng bước, lịch sử và `ht` của GKT (`models.py:297-308` — thêm hook `return_state=True`) vào `learner_view.json`; (iii) render bằng `ipywidgets.Dropdown` (không thêm dependency) hoặc app Streamlit ~20 dòng.

6. **Làm case study truy vết được**
   - Ghi `user_id` và `t` vào record trong `explain.py:97-106` và header `format_cases:121-124`.
   - Thay template `_explain_one` (`:143-160`) bằng bằng chứng thật: occlusion `Δŷ` từng bước; trọng số láng giềng **cho lần dự đoán cụ thể** (đã có `graph[skill]` — báo `adj-weight × đóng góp trạng thái láng giềng`, không chỉ top-5 trọng số thô).

7. **Sửa cách tính AUC**
   - `pipeline.py:151-173`: gom `pred`/`label` cả loader, tính 1 `roc_auc_score` cuối; giữ AUC theo batch chỉ để theo dõi train.

8. **Sửa/ghi chú ablation Identity ≡ Zero**
   - Vì `models.py:168` ghi đè ô self nên `A=I` và `A=0` trùng toán học — ghi chú rõ trong báo cáo, **hoặc** thêm ablation thật khác: **edge-shuffle** (hoán vị hàng của các phần tử khác 0 trong Transition, ~10 dòng ở `processing.py:172-197`) và/hoặc giữ đường chéo của Transition.

9. **Định vị pyKT**
   - Either dùng `pykt-toolkit` cho bước split/preprocess, hoặc viết mục "định vị giao thức" trong README/báo cáo: cách pipeline khớp pyKT và điểm lệch (`min_seq_len=3` vs 5, không explode đa skill, vocab trên toàn dữ liệu).

### P2 — trau chuốt / báo cáo

10. **Vệ sinh requirements**: pin version (`torch==…`, `scikit-learn==…`), thêm `environment.yml`; sửa README `:18-19` (sai về `run_level1`, cờ `RUN_GIKT`).
11. **Báo cáo 12–18 trang** (§8); đồng thời sửa checklist lạc quan trong `KE_HOACH.md:225` và `:229` cho khớp thực tế.
12. **Câu chuyện thực nghiệm**: số hiện tại cho thấy **không có lợi ích đồ thị** (Transition 0.589±0.088 vs Identity 0.591±0.086; DKT 0.395±0.138 trên smoke set). Cần chạy full ASSIST-2009 + 3 seed; nếu Transition vẫn ≤ Identity → chưa đạt tiêu chí "chứng minh vai trò cấu trúc đồ thị" → cân nhắc cách dựng cạnh khác (co-occurrence câu nhiều khái niệm, cạnh tiên quyết Junyi — xem `KE_HOACH.md:69-72`).

---

## 4. Tổng kết điểm số

| Trạng thái | Số mục | Gồm |
|-----------|--------|-----|
| **THỎA** | 6 | Dataset, baseline+đồ thị, biến thể ablation, split theo người, đồ thị từ train, không nhãn tương lai |
| **ĐÃ FIX (code xong, cần chạy/commit)** | 5 | Đa seed split cố định, AUC global, Shuffled ablation, dashboard learner view, user_id/t trong case study |
| **MỘT PHẦN** | 4 | Case 5+5 (code fix, chưa có artifact), file split (chưa commit), pyKT, gói tái lập (đã pin version) |
| **CHƯA THỎA** | 4 | Artifact commit (cần chạy notebook), push GitHub, báo cáo 12–18 trang, bằng chứng thực nghiệm vai trò đồ thị |

**Thứ tự thực hiện còn lại:**
1. Push GitHub (P0.4) — để Kaggle clone được code mới.
2. Chạy notebook full ASSIST-2009 + RUN_MULTI_SEED=True trên Kaggle GPU (P0.2, P2.12).
3. Commit artifact: `outputs/level1/{split.json, metrics.csv, cases.md, multiseed_metrics.csv, learner_view.json}` (P0.3).
4. Viết báo cáo 12–18 trang, đối chiếu số liệu đa seed (P2.11, P2.12).
5. (Tuỳ chọn) Định vị pyKT trong README/báo cáo (P1.9).
