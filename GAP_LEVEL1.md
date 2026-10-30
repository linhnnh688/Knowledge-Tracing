# Gap Analysis — `kaggle_level1.ipynb` + package `level1/` so với yêu cầu Cấp độ 1

> Ngày rà soát: 15/10/2026. Đối chiếu với `YEU_CAU.md` (viết lại từ đề cương `Project02_Knowledge_Tracing.pdf`).
> Ghi chú trạng thái repo khi rà: `kaggle_level1.ipynb` và `KE_HOACH.md` đã sửa nhưng **chưa commit/push**; notebook trên Kaggle **clone từ GitHub** nên sẽ chạy bản HEAD cũ.
>
> **Trạng thái cập nhật sau khi sửa code (cùng ngày):** đã fix P0.1 (đa seed split cố định), P0.2 (QUICK_RUN=False), P1.5 (dashboard), P1.6 (user_id/t trong case), P1.7 (AUC global), P1.8 (thêm Shuffled ablation), P2.10 (requirements pin + README sửa). **Còn lại:** P0.3 (commit artifact — cần chạy notebook), P0.4 (push GitHub), P1.9 (định vị pyKT), P2.11 (báo cáo 12–18 trang), P2.12 (chạy full + chứng minh vai trò đồ thị).

> **Trạng thái cập nhật 30/10/2026 (Cấp độ 2 đã xong code):** P0.4 đã xong — toàn bộ `level1/`,
> `level2/`, `GRKT/` và hai notebook đã commit + push lên `origin/main` (notebook Kaggle clone từ
> đó; `git rev-list origin/main...HEAD` = 0/0). P1.9 đã xong — mục "Định vị giao thức so với pyKT"
> trong `README.md` liệt kê chỗ khớp/chỗ lệch và lý do.
> P0.3 **đã xong**: `outputs/level1/{level1_metrics.csv, multiseed_metrics.csv, cases.md,
> split.json, metrics.json, learner_view.json, *.pt}` đã commit ở `143904b`, chạy trên **full
> ASSISTments 2009** (không phải smoke). Lưu ý `.gitattributes` đẩy `*.json` sang LFS trong khi
> máy chưa cài git-lfs → `split.json`/`metrics.json`/`learner_view.json` đang là **pointer blob**,
> chỉ `*.csv` và `cases.md` là nội dung thật (đã ghi rõ trong `README.md`).
> P2.12 **đã có số liệu**: Transition 0.7640±0.0066 vs Zero 0.7551±0.0055 (Δ=+0.0089, t≈1.8) và
> Shuffled 0.7656±0.0008 ≈ Transition (t≈0.4) → **chưa chứng minh được vai trò *cấu trúc* đồ thị
> ở Cấp độ 1**; `README.md` đã báo cáo trung thực điểm này và Cấp độ 2 dùng tiêu chí
> `consistency_spread`/`locality`/`forget_slope` của quỹ đạo GRKT làm bằng chứng thay thế.
> **Còn mở:** P2.11 (báo cáo 12–18 trang), và chạy full Cấp độ 2 trên Kaggle (`QUICK_RUN=False`).

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

- `level2/` + notebook `kaggle_level2.ipynb` (GRKT, KDD 2024) đã viết xong và vendored `GRKT/`;
  `level3/` (explain + quiz + chuyển giao) vẫn chỉ là kế hoạch trong `KE_HOACH.md`.
- `KE_HOACH.md:225` đánh dấu đa seed `[x]` và `:229` nói hai notebook "xuất toàn bộ kết quả Cấp độ 1" — **tại thời điểm rà 15/10 là không đúng**; đến 30/10 nhận định đó **đã đúng** (`multiseed_metrics.csv`, `cases.md`, `split.json`, `learner_view.json`, 6 checkpoint đã commit ở `143904b`).

### 2.6. Đối chiếu lại 30/10/2026 (sau lần chạy full ASSISTments 2009)

Các hàng ở §2–§5 dưới đây là trạng thái **15/10**; cột "30/10" là trạng thái đã kiểm tra lại bằng
artifact đã commit + code hiện tại. Chi tiết số liệu: `README.md` § *Kết quả Cấp độ 1*.

| # | Trạng thái 15/10 | Trạng thái 30/10 | Ghi chú |
|---|---|---|---|
| L1.1 | THỎA ⚠️ (QUICK_RUN mặc định) | **THỎA** | `QUICK_RUN=False` là mặc định; số liệu báo cáo lấy từ `skill_builder_data.csv` full. |
| L1.3 | THỎA ⚠️ (thiếu edge-shuffle) | **THỎA** | `GKT-Shuffled` đã chạy và có số riêng (0.7656±0.0008). |
| L1.4 / D5 | MỘT PHẦN (`cases.md` chưa tồn tại) | **THỎA** | `outputs/level1/cases.md` đã commit, có `user_id`, `t`, `skill=7:Venn Diagram`, lịch sử 8 bước, láng giềng Transition (train only). 30/10: `annotate_evidence` thay template `why` bằng số **tính lại từ mô hình** (phản chứng lật kết quả từng bước + ablation `A := 0` riêng cho case + cạnh chạm-lần-cuối); `distinct_skills=True` hết cảnh 5 case cùng một skill. Artifact mới cần một lần chạy lại notebook. |
| X3 | MỘT PHẦN (split không commit, đổi theo seed) | **THỎA** | Split load một lần seed=42 rồi tái dùng cho cả 3 seed; `split.json` đã commit (dạng LFS pointer — xem `README.md`). |
| E1 | THỎA ⚠️ (AUC theo batch) | **THỎA** | Gom toàn bộ loader rồi tính một `roc_auc_score` (`level1/pipeline.evaluate`). |
| E2 | CHƯA THỎA | **THỎA** | `RUN_MULTI_SEED=True`, 5 mô hình × 3 seed, mean±std trong `multiseed_metrics.csv`. |
| E3 | MỘT PHẦN (số smoke set) | **MỘT PHẦN** | Có số full + 3 seed, nhưng kết luận vẫn là **không chứng minh được vai trò cấu trúc đồ thị**: Shuffled ≈ Transition (t≈0.4), Transition − Zero chỉ t≈1.8. Hướng xử lý: sang Cấp độ 2 (GRKT + `consistency_spread`). |
| D1 | MỘT PHẦN | **THỎA** | `requirements.txt` đã pin, README đã sửa mô tả pipeline. |
| D2 | MỘT PHẦN | **THỎA** | split commit + mục "Định vị giao thức so với pyKT" + mô tả chống rò rỉ ở `README.md`/`KE_HOACH.md`. |
| D4 | CHƯA THỎA | **THỎA** (code + artifact) | `level1/dashboard.py` + `learner_view.json` (đã commit, 113 MB qua LFS). |
| D3 | CHƯA THỎA | **CHƯA THỎA** | Báo cáo 12–18 trang và slide **vẫn là việc còn lại duy nhất chặn nộp** (P2.11). |


---

## 3. Việc cần làm, ưu tiên từng mức

### P0 — chặn điểm (bắt buộc làm trước)

> Cách đọc mục này: mỗi việc gạch đầu dòng là **kế hoạch gốc 15/10**; dòng `✅ 30/10` ghi trạng thái
> thật sau khi code xong và đã chạy full trên Kaggle.

1. **Bật và mở rộng đa seed**
   - Đặt `RUN_MULTI_SEED=True` (`kaggle_level1.ipynb:478`).
   - Thêm `GKT-Dense/Identity/Zero` vào `multi_configs` (`:485-488`) vì ablation là khiếu nại bắt buộc.
   - **Sửa confound split**: load split **một lần** với `seed=42`, tái sử dụng `bundle` cho mọi seed; chỉ đổi seed huấn luyện.
   - Commit `multiseed_metrics.csv` kết quả.
   - ✅ 30/10: cả 4 ý đã xong — `multiseed_metrics.csv` có 5 mô hình × 3 seed, split cố định, đã commit (`143904b`) và push.

2. **Chạy kết quả báo cáo trên dữ liệu đầy đủ**
   - `QUICK_RUN=False` (ASSIST-2009 full).
   - ✅ 30/10: mặc định `QUICK_RUN=False`; số trong `outputs/level1/*.csv` là full ASSISTments 2009.

3. **Commit artifact**
   - `outputs/level1/{split.json, metrics.csv, metrics.json, cases.md, multiseed_metrics.csv}`.
   - ✅ 30/10: đã commit, kèm 6 checkpoint `.pt` và `learner_view.json`. ⚠ `.gitattributes` đẩy `*.json`
     sang LFS mà máy dev chưa cài git-lfs → ba file JSON trong repo là **pointer**, không phải nội dung;
     `*.csv`/`cases.md` là nội dung thật. Muốn người khác tải được JSON thì cài git-lfs rồi push lại,
     hoặc đổi `.gitattributes` bỏ rule `*.json` (thao tác này cần chủ ý, chưa làm).

4. **Đồng bộ remote**
   - ✅ 30/10: `origin/main` == `HEAD` (`git rev-list --left-right --count` = 0/0) → Kaggle clone đúng bản mới.

### P1 — bắt buộc theo đề

5. **Dashboard chọn người học** (§8)
   - ✅ 30/10: `level1/dashboard.py` dùng `ipywidgets.Dropdown`, `export_learner_view` ghi
     `learner_view.json` (p̂ từng bước + lịch sử + trạng thái tri thức chuẩn hoá).
     Chưa làm: thêm `user_id` vào `KTDataset.__getitem__` (đã đi đường `Subset.indices` + `user_ids_from_loader`).

6. **Làm case study truy vết được** — *xong (30/10, phần bằng chứng tính toán)*
   - ✅ `user_id` và `t` có trong record và header `cases.md` (xem `outputs/level1/cases.md`:
     `user=78903 t=64 skill=7:Venn Diagram y=1 p=0.952`).
   - ✅ `annotate_evidence` (`level1/explain.py`) thay template bằng số liệu **tính lại từ mô hình**:
     (i) phản chứng nếu–thì — lật outcome từng bước trong cửa sử (`features[h] = 2*skill + (1-correct)`,
     đúng cách mã hoá của `GKT/processing.py:85`) rồi chạy lại forward → `Δp` từng bước;
     (ii) ablation đồ thị **riêng cho case đó** — `model.graph.data := 0` trong một lần forward rồi khôi
     phục → `Δp` do đồ thị; (iii) cạnh mạnh nhất ở khái niệm đích kèm `direction` và thời điểm người học
     từng chạm vào nó. Có `p_recheck` đối chiếu giá trị ghi trong batch; lệch quá `tol=0.02` → in
     "counterfactual: not available (...)" thay vì bịa số.
     Tính tương thích ngược: `collect_cases(..., evidence=True)` là tham số mới ở cuối, chữ ký cũ vẫn
     chạy; toàn bộ phần bằng chứng bọc `try/except` nên `kaggle_level1.ipynb` (cell 22, **không sửa**)
     không thể vì nó mà chết.
   - ✅ Chọn case: `distinct_skills=True` mặc định → 5 case đúng nằm trên 5 khái niệm khác nhau, thay
     vì 5 lần cùng một skill như `cases.md` cũ.
   - Kiểm chứng: không có torch/pandas local, nên đã chạy toàn bộ đường đi của
     `collect_cases → annotate_evidence → format_cases` trên một **stub torch tối giản**
     (fake model deterministic, 2 người học × 6 bước): index `pos/t/idx` thẳng hàng, `evidence=False`
     không in dòng phản chứng, hai nhánh guard (position ngoài phạm vi, `p_recheck` lệch) báo đúng.
     **Chưa kiểm chứng bằng PyTorch thật** → cần một lần chạy `QUICK_RUN=True` trên Kaggle trước khi
     đưa `cases.md` mới vào báo cáo.
   - Còn bỏ ngỏ (P2, không chặn nộp): GNNExplainer/fidelity như đề bài gợi ý ở Cấp độ 3, và occlusion
     "xoá hẳn một bước" (thay vì lật kết quả) — xoá bước làm dịch chỉ số nên chưa làm.

7. **Sửa cách tính AUC**
   - ✅ 30/10: `level1.pipeline.evaluate` gom toàn bộ `pred`/`label` của loader rồi tính một
     `roc_auc_score`; AUC theo batch chỉ còn dùng để theo dõi train/early-stop.

8. **Sửa/ghi chú ablation Identity ≡ Zero**
   - ✅ 30/10: đã thêm ablation `Shuffled` (hoán vị cạnh, giữ nguyên số cạnh) **và** ghi chú lý do
     `A=I ≡ A=0` trong `README.md` (dẫn `GKT/models.py:137-142,168`). Kết quả thực đo xác nhận trùng
     nhau từng chữ số AUC.

9. **Định vị pyKT**
   - ✅ 30/10: `README.md` § "Định vị giao thức so với pyKT" — bảng khớp/lệch (split user-wise seed 42,
     0.6/0.2/0.2, `min_seq_len=3` vs ≥5, `max_seq_len=200`, đồ thị train-only, label shift-1, AUC global,
     không explode Q-matrix, vocab cả file) + đoạn về single-split vs k-fold của `GRKT/Runner.py`.

### P2 — trau chuốt / báo cáo

10. **Vệ sinh requirements**
    - ✅ 30/10: `requirements.txt` đã ghi dải phiên bản (`torch>=2.0`, `pandas>=2.0`, `numpy>=1.24,<3`, …)
      kèm chú thích lý do **không hard-pin**: môi trường đích là Kaggle image (Python 3.13, pandas 3.x);
      hard-pin sẽ buộc pip thay pandas/numpy của image ngay trong cell cài đặt — chính là cách đã làm
      hỏng môi trường ở các lần chạy trước. Code `level1/` + `level2/` viết tương thích cả pandas 2 lẫn 3
      (vẽ hình bằng matplotlib thuần) nên dải phiên bản là đủ tái lập. README cũng đã sửa mô tả sai
      về `run_level1(...)` và cờ `RUN_GIKT`.
    - ❌ Còn thiếu `environment.yml` — chỉ cần cho chạy local; không chặn nộp.
11. **Báo cáo 12–18 trang + slide + bảng phân công** (§8) — việc chặn nộp duy nhất còn lại.
    - ✅ 30/10: `KE_HOACH.md` đã bỏ/mô tả lại checklist lạc quan, `README.md` đã có mục số liệu Cấp độ 1
      và mục định vị pyKT — hai mục này chính là nguyên liệu cho §4 và §6 của báo cáo.
    - ❌ File báo cáo và slide chưa tồn tại trong repo.
12. **Câu chuyện thực nghiệm** — *đã chạy full, kết luận như sau*:
    - Số full ASSIST-2009 + 3 seed (split cố định): DKT **0.7860±0.0006**, GKT-Transition 0.7640±0.0066,
      GKT-Shuffled 0.7656±0.0008, GKT-Identity/Zero 0.7551±0.0055.
    - Transition − Zero = +0.0089 (SE 0.0050, t≈1.8): có đồ thị giúp nhưng **không có ý nghĩa thống kê** ở n=3.
    - Shuffled − Transition = +0.0016 (t≈0.4): **cấu trúc cạnh không tạo khác biệt** → tiêu chí
      "chứng minh vai trò cấu trúc đồ thị" **không đạt ở Cấp độ 1** trên dữ liệu 1 khái niệm/câu hỏi.
    - Hướng xử lý đã chọn: không chế số; báo cáo trung thực + chuyển bằng chứng sang Cấp độ 2, nơi
      GRKT có đồ thị **hai loại cạnh** (`rel_map` đồng xuất hiện + `pre_map` tiên quyết) và tiêu chí
      quỹ đạo `consistency_spread` đo trực tiếp "thay cạnh có làm mastery lan truyền đúng hướng không"
      (mức AUC không phân biệt được).

---

## 4. Tổng kết điểm số

**15/10** (nguyên trạng khi rà):

| Trạng thái | Số mục | Gồm |
|-----------|--------|-----|
| **THỎA** | 6 | Dataset, baseline+đồ thị, biến thể ablation, split theo người, đồ thị từ train, không nhãn tương lai |
| **ĐÃ FIX (code xong, cần chạy/commit)** | 5 | Đa seed split cố định, AUC global, Shuffled ablation, dashboard learner view, user_id/t trong case study |
| **MỘT PHẦN** | 4 | Case 5+5 (code fix, chưa có artifact), file split (chưa commit), pyKT, gói tái lập (đã pin version) |
| **CHƯA THỎA** | 4 | Artifact commit (cần chạy notebook), push GitHub, báo cáo 12–18 trang, bằng chứng thực nghiệm vai trò đồ thị |

**30/10** (sau khi chạy full và commit `143904b` + các commit level2):

| Trạng thái | Số mục | Gồm |
|-----------|--------|-----|
| **THỎA** | 14 | 6 mục gốc + đa seed split cố định, AUC global, Shuffled ablation, dashboard `learner_view.json`, user_id/t trong case study, artifact đã commit, push GitHub, định vị pyKT, requirements, **bằng chứng phản chứng + ablation đồ thị từng case** (`annotate_evidence`) |
| **MỘT PHẦN** | 2 | Vai trò **cấu trúc** đồ thị ở Cấp độ 1 (Shuffled ≈ Transition — chủ động báo cáo trung thực, bằng chứng chuyển sang tiêu chí quỹ đạo GRKT Cấp độ 2); `cases.md` trong repo vẫn là bản chạy **trước** khi có `annotate_evidence` → cần chạy lại notebook một lần để cập nhật artifact |
| **CHƯA THỎA** | 2 | Báo cáo 12–18 trang + slide + bảng phân công (P2.11); số liệu Cấp độ 2 trên Kaggle (`QUICK_RUN=False`, hai tập dữ liệu) |

**Thứ tự thực hiện còn lại:**
1. Chạy `kaggle_level2.ipynb` với `QUICK_RUN=False` (assist2009 + junyi, 4 biến thể đồ thị, đa seed) →
   `level2_output.zip` (cell §8.5) chứa toàn bộ artifact.
2. Chạy lại `kaggle_level1.ipynb` nếu cần số liệu mới; commit artifact Cấp độ 2 vào `outputs/level2/` (P0.3 phần level2).
3. Viết báo cáo 12–18 trang + slide + bảng phân công (P2.11) — nguyên liệu số liệu đã có trong
   `README.md` § *Kết quả Cấp độ 1* và § *Định vị giao thức so với pyKT*.
4. (P1.6 phần còn lại) Occlusion `Δŷ` cho case study, thêm tham số tuỳ chọn mặc định tắt để không đổi
   hành vi cell đang chạy trong `kaggle_level1.ipynb`.
5. (Tuỳ chọn, cần chủ ý) Bỏ rule `*.json` trong `.gitattributes` hoặc cài git-lfs để ba file JSON thật
   sự có nội dung khi tải repo.
