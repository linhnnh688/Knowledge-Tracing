# Gap Analysis — `kaggle_level2.ipynb` + package `level2/` + `GRKT/` so với yêu cầu Cấp độ 2

> Ngày rà soát: 01/11/2026. Đối chiếu với `YEU_CAU.md` (viết lại từ đề cương
> `Project02_Knowledge_Tracing.pdf`), chủ yếu §5 (cấp 2), §6 (chống rò rỉ), §7 (đánh giá), §8 (sản phẩm).
> Artifact được rà: `outputs/level2/` — commit `ff0fb82` ("level 2 outputs"), 29 file, 2 tập dữ liệu.
> Cách rà: đọc toàn bộ `level2/*.py` + `GRKT/model/GRKT.py`, đối chiếu từng con số trong
> `level2_metrics.csv` / `trajectory_metrics.csv` / `summary.json` / `split.json`, và so trực tiếp
> split Cấp độ 1 với Cấp độ 2.

---

## 0. Kết luận nhanh

| # | Kết luận | Mức |
|---|---|---|
| 1 | **Tái lập GRKT (KDD 2024) trên 2 tập công khai, cùng split seed 42, cùng AUC/ACC/NLL**: ĐẠT. GRKT thắng đường nền trên cả hai tập (assist2009 +0,0107 AUC; junyi +0,0510 AUC). Split Cấp độ 2 **trùng khít** `outputs/level1/split.json` (từng user_id). | ✅ |
| 2 | **Trực quan hoá mastery theo khái niệm × thời gian**: ĐẠT. Heatmap + đường cong + `mastery_*.csv` + `traj_*.npz`; giá trị mastery không suy biến (p05 0,22 → p95 0,82). | ✅ |
| 3 | **"Đo ≥ 2 tiêu chí hợp lý của quỹ đạo"**: KHÔNG ĐẠT trên thực chất. 4 tiêu chí được định nghĩa, nhưng **3 tiêu chí bị đóng băng ở 1,0 do cấu trúc mô hình** (`consistency_strengthen`, `consistency_spread`, `locality`) — không thể sai, nên không phải bằng chứng. Chỉ còn `forget_slope`/`pearson_dt_dm` là số đo thật. | ❌ |
| 4 | **Bằng chứng vai trò đồ thị ở Cấp độ 2** (thứ `GAP_LEVEL1.md` hứa thay cho thất bại ở Cấp 1): CHƯA CÓ. Chỉ train 1 biến thể `train`; `RUN_GRAPH_ABLATIONS=False` nên không có `zero`/`shuffled`/`dense` để so. | ❌ |
| 5 | **Bảng so sánh trộn seed**: 3 dòng đường nền đọc `outputs/level1/*.pt`, nhưng các file này đã bị vòng đa seed **ghi đè bằng seed 456**, trong khi bảng Cấp độ 2 ghi `seed=42`. GKT-Transition lệch 0,7443 → 0,7722 (4,2σ). | ⚠️ |
| 6 | **Sản phẩm §8**: còn thiếu báo cáo 12–18 trang + slide + bảng phân công; thiếu dashboard "chọn người học" cho Cấp 2; thiếu 3 seed cho GRKT; thiếu `calibration.csv`/`multiseed_metrics.csv`/lịch sử train. | ❌ |

Nói ngắn: phần **mô hình + dữ liệu + chống rò rỉ** đã xong và có số thật; phần **tiêu chí quỹ đạo** là chỗ hổng kỹ thuật lớn nhất — số liệu 1,0 hiện tại không chứng minh được gì và nếu đưa vào báo cáo nguyên trạng sẽ bị chất vấn ngay.

---

## 1. Hiện trạng

### 1.1 Cấu hình đã sinh ra artifact (notebook §0.2)

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| `QUICK_RUN` | `False` | chạy báo cáo, không phải smoke |
| `DATASETS` | `["assist2009", "junyi"]` | đủ yêu cầu ≥ 2 tập |
| `EPOCHS` | **5** | ⚠️ đường nền chạy 20 epoch (`BASELINE_EPOCHS=20`) — xem F4 |
| `BATCH_SIZE` / `MAX_SEQ_LEN` | 32 / 200 | `MAX_SEQ_LEN=200` khớp Cấp độ 1 ⇒ checkpoint nạp lại được |
| `VARIANTS` | `("train",)` | `RUN_GRAPH_ABLATIONS=False` ⇒ không có dense/zero/shuffled |
| `RUN_CALIBRATION` | `False` | không sinh `calibration.csv` |
| `RUN_MULTI_SEED` | `False` | không sinh `multiseed_metrics.csv` |
| `RESUME_GRKT_CKPT` | `True` | artifact có `seconds=0.0` vì nạp checkpoint thay vì train |
| `D_HIDDEN/K_HIDDEN/K_HOP` | 128 / 16 / 1 | `k_hop=1` là mấu chốt của F1 |
| `GRAPH_THRESH` / `COLD_THRESH` | 0,6 / 5 | |
| `ALPHA` / `GAP_CAP_MIN` | 0,01 / 240 phút | trục thời gian = phút cộng dồn, chặn trên |
| `SPLIT_SEED` | 42 | bất di bất dịch, khớp Cấp độ 1 |

Lưu ý: `summary.json` **không ghi** các giá trị trên, cũng không ghi `epochs`, `resumed_from`,
`lr`, `k_hidden`. Người đọc artifact không thể biết GRKT được train 5 hay 20 epoch (F5).

### 1.2 Artifact có / không

| File | assist2009 | junyi | Ghi chú |
|---|---|---|---|
| `split.json` | ✅ | ✅ | trùng `outputs/level1/split.json` với assist2009 |
| `leak_report.txt` | ✅ | ✅ | có số giao tập = 0 |
| `graphs_train_only.npz` | ✅ | ✅ | `rel`, `pre`, `rel_bin`, `pre_bin` — đồ thị train |
| `level2_metrics.csv` | ✅ 4 dòng | ✅ 4 dòng | 1 seed |
| `trajectory_metrics.csv` | ✅ 1 dòng | ✅ 1 dòng | chỉ biến thể `train` |
| `trajectories/{traj,mastery,heatmap,curves}_train.*` | ✅ | ✅ | 150 user ghi quỹ đạo, 25 user xuất CSV, heatmap 1 user |
| `GRKT-train.pt` | ✅ | ✅ | checkpoint cuối, có fingerprint |
| `summary.json` | ✅ | ✅ | thiếu siêu tham số (F5) |
| `baselines/{DKT,GKT-Transition,GKT-Zero}.{pt,json}` | — | ✅ | assist2009 nạp thẳng checkpoint Cấp độ 1 |
| `calibration.csv` / `reliability.png` | ❌ | ❌ | `RUN_CALIBRATION=False` — **tuỳ chọn** theo §7 |
| `multiseed_metrics.csv` | ❌ | ❌ | `RUN_MULTI_SEED=False` — xem E2 |
| `traj_{dense,zero,shuffled}.*` | ❌ | ❌ | hệ quả `VARIANTS=("train",)` — xem F6 |
| lịch sử train (loss/val AUC theo epoch) | ❌ | ❌ | `history` có trong bộ nhớ nhưng không được ghi — F5 |

### 1.3 Số liệu thực tế trong artifact

**Dữ liệu** (`summary.json` → `data_stats`):

| | assist2009 | junyi |
|---|---|---|
| người học / tương tác | 3.884 / 237.943 | **46.229 / 4.019.325** |
| khái niệm (`n_knows`) | 123 | 200 (cap) |
| câu hỏi (`n_probs`) | 17.737 | 3.987 |
| split user train/val/test | 2.330 / 776 / 778 | 27.737 / 9.245 / 9.247 |
| tương tác train/val/test | 142.299 / 47.508 / 48.136 | 2.421.121 / 801.456 / 796.748 |
| cạnh `rel` / `pre` từ train | 2.720 / 2.642 | 24.908 / 14.468 |
| cạnh nếu dựng từ val | 14.024 tổng | 71.612 tổng |
| `time_mode` | `duration_ms` | `timestamp_ms` |
| `dt_zero_frac` | 0,531 | **0,881** |
| `dt_p95` (phút) | 3 | 240 (= cap) |

> ⚠️ Số Junyi trong artifact **không khớp tài liệu**: README (`:141`, `:150`), notebook (§0.2, §2)
> và `level2/dataset_spec.py:128` đều nói "toàn bộ 72.630 người học / 16,2 triệu lượt". Nguyên nhân
> gần như chắc chắn là hệ quả của `top_concepts=200` + lọc `min_interactions=10` (xem F3) — nhưng
> **phải sửa tài liệu**, vì nếu giảng viên đối chiếu sẽ thấy hai con số khác nhau.

**Mô hình** (`level2_metrics.csv`, split seed 42):

| Tập | Mô hình | AUC | ACC | NLL | val AUC | n_test |
|---|---|---|---|---|---|---|
| assist2009 | **GRKT-train** | **0,7958** | 0,7421 | 0,5156 | 0,8017 | 47.358 |
| assist2009 | DKT | 0,7851 | 0,7329 | 0,5282 | — | 47.358 |
| assist2009 | GKT-Transition | 0,7722 | 0,7253 | 0,5414 | — | 47.358 |
| assist2009 | GKT-Zero | 0,7620 | 0,7120 | 0,5505 | — | 47.358 |
| junyi | **GRKT-train** | **0,8157** | 0,8052 | 0,4238 | 0,8151 | 787.501 |
| junyi | GKT-Transition | 0,7673 | 0,7827 | 0,4640 | — | 787.501 |
| junyi | DKT | 0,7647 | 0,7849 | 0,4633 | — | 787.501 |
| junyi | GKT-Zero | 0,7557 | 0,7802 | 0,4699 | 0,7547 | 787.501 |

Δ(AUC) GRKT − DKT = **+0,0107** (assist2009), **+0,0510** (junyi). Δ GRKT − GKT-Transition lần lượt
**+0,0236** / **+0,0484**. `n_test` giống hệt nhau giữa GRKT và đường nền ⇒ cùng mặt nạ đánh giá ⇒
so sánh được (điểm tốt, đã xác minh).

**Quỹ đạo** (`trajectory_metrics.csv`):

| | assist2009 | junyi |
|---|---|---|
| users / steps | 150 / 9.048 | 150 / 13.975 |
| `consistency_strengthen` | **1,0** | **1,0** |
| `consistency_spread` | **1,0** | **1,0** |
| `locality` | 0,9999999997 | 0,9999999999 |
| `forget_slope` | −4,1e−4 | −9,6e−5 |
| `pearson_dt_dm` | −0,492 | −0,442 |
| `abs_delta_strengthen` | 0,0607 | 0,0270 |
| `abs_delta_learn_forget` | 0,00029 | 0,00054 |
| `abs_delta_retrieval` | 0,0273 | 0,0548 |

Ba số 1,0 đầu tiên là **triệu chứng của F1**, không phải kết quả.

### 1.4 Điểm đã xác minh đúng (không phải gap)

1. **Split Cấp độ 2 ≡ Cấp độ 1**: `outputs/level2/assist2009/split.json` trùng khít
   `outputs/level1/split.json` ở cả `train_users` / `val_users` / `test_users` (2.330 / 776 / 778,
   so sánh từng phần tử).
2. **Đồ thị chỉ dựng từ train**, có phép so sánh với val để chứng minh phụ thuộc tập:
   `level2/grkt_data.py:375` (`build_relevance_maps` chỉ nhận `splits["train"]` ở
   `experiment.py:298`), `leak_report.py` in "cạnh từ train = 14.024 / nếu từ val = 11.081".
3. **Không dùng nhãn hiện tại/tương lai**: GRKT chỉ đọc `knows[i+1]`, `times[i+1]` (danh tính +
   thời điểm câu kế, không dùng `corrs[i+1]`) — ghi trong `leak_report.txt`, khớp
   `GRKT/model/GRKT.py:643-679`.
4. **AUC/ACC/NLL toàn cục**: `level2/grkt_api.py:438-471` nối mọi batch rồi tính một lần, không
   lấy trung bình theo batch.
5. **Đối chiếu từng người học**: `build_grkt_data` gọi `_assert_align` cho mọi user
   (`grkt_data.py:335-354`) — sai một chỉ số là raise, không âm thầm so hai split khác nhau.
6. **Đồ thị tiên quyết có thật và hai chiều**: `pre_map` 2.642 cạnh (assist) / 14.468 (junyi);
   `adjacency_from_graphs` hợp rel + pre + pre.T.
7. **Mastery không suy biến**: trên `mastery_train.csv`, `m_strong` p05 = 0,221 / median = 0,549 /
   p95 = 0,816 (assist2009); 0,383 / 0,597 / 0,830 (junyi) ⇒ biểu đồ có nội dung, không phải hằng số.
8. **Tái lập được bằng checkpoint**: `GRKT-<kind>.pt` có fingerprint (split, số user, số cạnh,
   `batch_size`, `d_hidden`, `k_hidden`, `seed`, `lr`…) và từ chối resume khác cấu hình
   (`grkt_api.py:248-259`).

---

## 2. Bảng đối chiếu yêu cầu

### 2.1. Cốt lõi Cấp độ 2 (§5, dòng "2", điểm trần 8,5)

| # | Yêu cầu | Trạng thái | Bằng chứng / vấn đề |
|---|---|---|---|
| L2.1 | Trực quan hoá trạng thái tri thức theo **khái niệm** và theo **thời gian** | **THỎA** | `trajectories/heatmap_train.png` (concept × step, màu `sigmoid(m)`), `curves_train.png` (đường mastery của 4 KC/user), `mastery_train.csv` (bảng dài `m_pre/m_strong/m_tilde`), `traj_train.npz` (150 user × 3 mốc). Có cho **cả hai** tập. |
| L2.2 | Định nghĩa và đo **≥ 2 tiêu chí hợp lý** của quỹ đạo | **KHÔNG ĐẠT** | 4 tiêu chí có mã (`trajectory.py:152-323`) và có số, nhưng 3/4 **bị đóng băng ở 1,0** (F1). Thực chất chỉ còn `forget_slope` + `pearson_dt_dm` của **cùng một** tiêu chí (suy giảm theo thời gian), và cả hai đều rất nhỏ/mất nghĩa vì `dt_zero_frac` = 0,53 / 0,88. Thiếu 2 tiêu chí mà đề nêu tên: **nhất quán tiên quyết** và **ổn định với thay đổi nhỏ trong lịch sử**. |
| L2.3 | Triển khai/tái lập **1 phương pháp graph-aware công bố từ 2024** | **THỎA** (có điều kiện) | GRKT (Cui et al., KDD 2024), vendored trong `GRKT/`; 7 chỗ sửa được khai báo (`KE_HOACH.md` mục "Khai báo chỉnh sửa", notebook §1). Điều kiện: chỉ **5 epoch**, trong khi upstream mặc định `--epoch 500` + early stop (`GRKT/BaseModel.py:30`), còn đường nền chạy 20 epoch ⇒ so sánh chưa cân (F4). |
| L2.4 | Đánh giá trên **≥ 2 tập công khai** với **cùng quy trình chia + cùng chỉ số** | **MỘT PHẦN** | Cùng mã `load_kt_data`, cùng `split_seed=42`, cùng AUC/ACC/NLL, cùng `max_seq_len=200` ⇒ phần "cùng quy trình" ở mức **giao thức** đạt. Nhưng tiền xử lý **khác nhau có chủ đích**: assist2009 `min_interactions=3` vs junyi `10`; junyi còn cap **200/1.326** khái niệm và bỏ 75% lượt (F3). Phải khai báo rõ trong báo cáo, nếu không sẽ bị coi là hai quy trình khác nhau. |
| L2.5 | Vai trò **cấu trúc** đồ thị (bằng chứng thay thế cho Cấp 1) | **CHƯA THỎA** | Chỉ train `train`; không có dòng `dense`/`zero`/`shuffled` trong `level2_metrics.csv` hay `trajectory_metrics.csv`. Notebook §5.1/§5.2 có in hướng dẫn so `train` vs `zero` nhưng dữ liệu không tồn tại. |
| L2.6 | So sánh quỹ đạo có đồ thị vs không đồ thị *(mở rộng, không bắt buộc)* | **CHƯA** | Không có `traj_zero`; cũng không ghi quỹ đạo của DKT/GKT (chỉ GRKT). |

### 2.2. Chống rò rỉ (§6 — áp dụng mọi cấp)

| # | Yêu cầu | Trạng thái | Bằng chứng / vấn đề |
|---|---|---|---|
| X1 | Chỉ dùng tương tác **trước** thời điểm dự đoán | **THỎA** | GRKT dự đoán bước `i` từ `h` sau bước `i−1`; `knows/times[i+1]` chỉ là danh tính + thời điểm. Ghi rõ trong `leak_report.txt`. |
| X2 | Đồ thị suy từ tần suất phải dựng **từ train** | **THỎA** | `experiment.py:298` chỉ truyền `splits["train"]["sequences"]`; `leak_report` in cả số cạnh nếu dựng từ val (14.024 vs 11.081; 77.392 vs 71.612) để chứng minh đồ thị **phụ thuộc** tập train. |
| X3 | Mọi mô hình dùng **cùng một** split; nộp split hoặc seed + mã | **THỎA** | `split.json` từng tập đã commit; assist2009 trùng khít Cấp độ 1; `SPLIT_SEED=42` cố định và được truyền cho mọi mô hình. |
| X4 | Tách theo **người học** | **THỎA** | Mỗi `UserSequence` là một người; `leak_report` in `train∩val = train∩test = val∩test = 0`. |
| X5 | Bám quy trình chuẩn hoá **pyKT** | **MỘT PHẦN** | Định vị khớp/lệch đã ghi ở `README.md` §"Định vị giao thức so với pyKT". Lệch cần khai báo: **cap 200 khái niệm chọn trên toàn bộ dữ liệu** (`dataset_spec.py:546-564`, transductive — không dùng nhãn nhưng dùng thống kê của val/test), `min_interactions` khác nhau giữa hai tập, `k_hop=1`/`k_hidden=16` thay cho cấu hình bài báo. |
| X6 | Dữ liệu tự thu thập tách theo người / 5-fold | — | Chưa tới (Cấp độ 3). |

### 2.3. Đánh giá (§7)

| # | Yêu cầu | Trạng thái | Bằng chứng / vấn đề |
|---|---|---|---|
| E1 | AUC, accuracy, BCE/NLL | **THỎA** | Đủ 3 chỉ số, tính toàn cục, trên cùng `n_test` cho mọi mô hình. |
| E2 | Trung bình ± std của **≥ 3 seed** | **CHƯA THỎA** | Không có `multiseed_metrics.csv`; GRKT chỉ có 1 seed. Đường nền có 3 seed **nhưng ở Cấp độ 1** và đó là split khác lần đo ở đây (xem F2). §7 xếp mục này vào **bắt buộc** của bảng Nhiệm vụ 1 — GRKT đang được trình bày như kết quả dự đoán mới nên vẫn phải có. |
| E3 | Ablation vai trò đồ thị (bỏ cạnh / xáo trộn / ma trận đơn vị) | **CHƯA THỎA ở C2** | Có `graph_variant()` hỗ trợ đủ 4 kiểu (`grkt_data.py:461-504`) nhưng chưa chạy. Ở Cấp 1 đã chạy và **kết luận là không phân biệt được** ⇒ đây đúng là chỗ cần bằng chứng mới. |
| E4 | Trực quan mức nắm vững theo thời gian | **THỎA** | Như L2.1. |
| E5 | ≥ 2 tiêu chí hợp lý | **KHÔNG ĐẠT** | Như L2.2 (F1). |
| E6 | Phân tích nâng cao (chọn **một**): calibration / lịch sử ngắn / kiểm tra hợp lý trạng thái | **THỎA một phần** | Nhóm chọn "quỹ đạo mastery" làm phân tích nâng cao, nhưng chính phần đó đang hỏng (F1). `calibration.py` viết xong (ECE/ACE/Brier/reliability) mà chưa chạy: `summary.json` có `"calibration": []`. Nếu sửa F1 tốn thời gian, **bật `RUN_CALIBRATION=True`** là cách rẻ nhất để có một phân tích nâng cao thật. |
| E7 | Độ ổn định / người học mới *(mở rộng)* | **CHƯA** | Không có đánh giá lịch sử ngắn hay cold-start. |

### 2.4. Sản phẩm nộp (§8)

| # | Yêu cầu | Trạng thái | Bằng chứng / vấn đề |
|---|---|---|---|
| D1 | Mã + requirements + hướng dẫn chạy | **THỎA** | `requirements.txt` dải phiên bản + lý do không hard-pin; README có mục chạy Kaggle 8 bước. Thiếu `environment.yml` (chỉ cần cho máy local). |
| D2 | Split cố định / mã tạo split + mô tả chống rò rỉ | **THỎA** | `split.json` × 2 tập + `leak_report.txt` × 2 + mục pyKT trong README. |
| D3 | Báo cáo 12–18 trang + slide + bảng phân công | **CHƯA THỎA** | Chưa có file nào. Đây là việc **chặn nộp** duy nhất giống Cấp 1. |
| D4 | Notebook/dashboard: chọn người học → xác suất, lịch sử, trạng thái tri thức, bằng chứng | **MỘT PHẦN** | Cấp 1 có `level1/dashboard.py` + `learner_view.json`. Cấp 2 **không có** widget chọn người học: notebook §5.2 chỉ in `mastery_train.csv.head(12)` và một bảng `groupby("correct")`; muốn xem một user phải sửa `user_index=0` trong `plot_mastery_heatmap`. |
| D5 | Phân tích 5 đúng + 5 sai | **MỘT PHẦN (ở C1)** | `outputs/level1/cases.md` vẫn là bản **template cũ** (`why: History already has correct answers...`), chưa có `annotate_evidence` như `GAP_LEVEL1.md` mô tả. Cấp 2 không có case study riêng. |
| D6 | 10 case xAI trên đồ thị + quiz tự thu thập | **CHƯA** | `level3/` chưa tồn tại — ngoài phạm vi Cấp 2. |

---

## 3. Phát hiện sâu (nguyên nhân gốc, không chỉ là "thiếu file")

### F1 — 3 trong 4 tiêu chí quỹ đạo bị đóng băng ở 1,0 bởi cấu trúc mô hình ⚠️ nghiêm trọng

**Hiện tượng.** `consistency_strengthen = 1,0`, `consistency_spread = 1,0`,
`locality = 0,9999999997` trên **cả hai** tập. Chỉ `forget_slope` và `pearson_dt_dm` là số đo thật.

**Nguyên nhân.** Trong `GRKT/model/GRKT.py`:

* `know_master_proj` là `Positive_Linear(k_hidden, 1, "softmax")` (`:83`) — trọng số sau
  `softmax(0)` **luôn dương** (`positive_activate`, `:16-27`, `:30-41`);
* cập nhật trạng thái là `h = h + corr·relu(total_gain) − (1−corr)·relu(total_loss)` (`:583`,
  `:627`), với `total_gain`/`total_loss` lan qua các ma trận cũng dương (`agg_*`, `gain_*`,
  `loss_*` đều `Positive_Linear`).

⇒ `Δm = W·Δh` với `W > 0` và `Δh` là tổ hợp **không âm** của gain (hoặc loss) ⇒ **mọi** khái niệm
nhận cùng dấu với kết quả ⇒ `spread ≡ 1`. Và vì `k_hop = 1`, `Δh = 0` ngoài láng giềng 1-hop ⇒
`locality ≡ 1` (mẫu số = tử số). Hai đẳng thức này **không phụ thuộc đồ thị là train, shuffled hay
dense** — miễn có cạnh.

**Hệ quả.**

* `trajectory.py:163-167` viết "so sánh `train`/`shuffled`/`dense` ở số đo này chính là bằng chứng
  cấp 2" — **sai**; `trajectory.py:368-371` viết "với zero/shuffled, spread ≈ 0 ⇒ None, locality = 1"
  — cũng sai với `shuffled` (vẫn có cạnh ⇒ vẫn 1,0).
* `GIAI_THICH_LEVEL1_LEVEL2.md` (`:717`, `:964`) lặp lại nhận định này.
* `GAP_LEVEL1.md` §12 và `README.md` dựa vào "tiêu chí `consistency_spread` của Cấp 2" để bù cho
  việc Cấp 1 không chứng minh được vai trò đồ thị ⇒ **lập luận đó hiện không có cơ sở**.

**Cách sửa (chọn ≥ 2, rẻ nhất trước).**

1. **Bỏ hẳn `spread`/`locality` khỏi vai trò "bằng chứng"**, giữ lại làm *kiểm chứng cài đặt*
   (giống `strengthen`) và ghi đúng như vậy trong báo cáo. Việc này mất 30 phút và trung thực.
2. **Thêm 2 tiêu chí thật sự có thể sai** (đề nêu đúng tên hai cái này):
   * *Nhất quán tiên quyết*: dùng `pre_map` (có sẵn `maps_train["pre"]`) — với cạnh `c → d`, đo
     `P(m_c ≥ m_d)` trên các bước người học đã làm đúng `d`, so với baseline hoán vị cạnh.
     Đại lượng này **không** bị ép bởi dấu của `W`.
   * *Ổn định với thay đổi nhỏ trong lịch sử*: lật kết quả (hoặc bỏ) 1 bước cuối rồi chạy lại
     forward, đo tương quan hạng Spearman giữa hai vector mastery tại `t`. GRKT có
     `gumbel_softmax` ngay ở eval nên cần cố định seed; so `train` vs `zero` cho ra chênh lệch
     thật (đồ thị lan truyền ⇒ thay đổi rộng hơn).
3. **Nếu vẫn muốn dùng "độ lan toả"**: thay `locality` bằng entropy chuẩn hoá của `|Δm|` trên
   toàn bộ KC, hoặc tỉ lệ thay đổi ở láng giềng **2-hop** (với `k_hop=1` phải bằng 0 ⇒ đo được
   cả hai phía), rồi so giữa các biến thể.

### F2 — Bảng so sánh trộn seed: checkpoint Cấp 1 đã bị ghi đè bởi seed 456 ⚠️

**Hiện tượng.** Cùng một split (đã kiểm chứng trùng khít), cùng file CSV, cùng `max_seq_len`, nhưng:

| Mô hình | `outputs/level1/level1_metrics.csv` (seed 42) | Cấp độ 2 (`level2_metrics.csv`) | Δ |
|---|---|---|---|
| DKT | 0,7866 | 0,7851 | −0,0015 |
| GKT-Transition | 0,7443 | **0,7722** | **+0,0279** (≈ 4,2σ, σ = 0,0066) |
| GKT-Zero | 0,7549 | 0,7620 | +0,0071 |

**Nguyên nhân.** `kaggle_level1.ipynb` cell §4.5 chạy 3 seed trong cùng vòng lặp và **truyền
`save_dir=SAVE_DIR`**, còn `level1/pipeline.py:345-346` lưu `torch.save(..., f"{name}.pt")`.
⇒ Mỗi `outputs/level1/*.pt` là trọng số của **seed cuối (456)**, không phải seed 42 của bảng
`level1_metrics.csv`. `level2/baselines.py` nạp các `.pt` này mà không có fingerprint
(`load_level1_state` chỉ thử `load_state_dict`), rồi `_table_row` (`experiment.py:681`) ghi
`seed` lấy từ kết quả (42) ⇒ **nhãn seed sai**.

**Hệ quả.** Bảng Cấp độ 2 đang so "GRKT seed 42" với "DKT/GKT seed 456" và ghi cả 4 dòng là
`seed=42`. Chênh lệch GRKT − DKT vẫn dương nên kết luận chính không đảo, nhưng **không tái lập
được** và giảng viên có quyền hỏi đúng điểm này.

**Cách sửa (một trong hai).**

* Sửa Cấp 1: bỏ `save_dir` trong vòng đa seed, hoặc lưu `f"{name}-seed{s}.pt"`; chạy lại
  `kaggle_level1.ipynb` để `*.pt` = seed 42 (hoặc ghi rõ `.pt` là seed 456).
* Hoặc sửa Cấp 2: `baselines.py` đánh dấu kết quả bằng seed thật của checkpoint (đọc từ tên file
  hoặc từ metadata mới), và ghi `seed` đúng vào bảng; nếu không xác định được thì **train lại**
  đường nền bằng `allow_retrain_baselines=True` + `baseline_ckpt_dir=None` (chậm hơn nhưng sạch).

### F3 — Quy mô Junyi trong artifact khác tài liệu, và 75% lượt bị bỏ ⚠️

**Hiện tượng.** Artifact: **46.229 người / 4.019.325 lượt**; tài liệu: **72.630 / 16,2 triệu**.

**Nguyên nhân (theo mã).** `_canonicalize_junyi` gọi `_cap_concepts(frame, 200)`
(`dataset_spec.py:523-528`) **trên toàn bộ khung**, tức chỉ giữ 200/1.326 bài tập phổ biến nhất và
**xoá mọi lượt trên bài hiếm**; sau đó `ensure_canonical` lọc người học theo
`min_interactions=10`. Cả hai đều xảy ra **trước** khi chia split, nên con số cuối nhỏ hơn nhiều
so với mô tả "toàn bộ người học".

**Hệ quả.**

* Báo cáo **không thể** nói "chạy toàn bộ Junyi" — phải nói "Junyi sau khi giữ 200 bài phổ biến
  nhất: 46.229 người / 4,02 triệu lượt (25% tổng)". Không nói rõ sẽ bị coi là che bớt dữ liệu.
* "Khái niệm" ở Junyi thực chất là **exercise (`ucid`)**, không phải KC — cần một câu giải thích.
* Bước chọn top-200 dùng thống kê của **toàn bộ** dữ liệu (kể cả val/test) ⇒ lựa chọn transductive.
  Không rò rỉ nhãn, nhưng phải khai báo cùng chỗ với "vocab fit trên toàn file" của Cấp 1.

**Cách sửa.** Thêm dòng log/số liệu vào `summary.json` (`n_rows_raw`, `n_rows_after_cap`,
`n_concepts_total`, `n_concepts_kept`) và sửa README/notebook cho khớp. Nếu còn ngân sách GPU, chạy
thêm biến thể `top_concepts=500` hoặc bỏ cap trên một mẫu con để cho thấy kết luận không phụ thuộc
cap (tuỳ chọn, không chặn nộp).

### F4 — GRKT 5 epoch, đường nền 20 epoch: so sánh chưa cân ⚠️

Notebook đặt `EPOCHS = 5` cho GRKT nhưng `BASELINE_EPOCHS = 20` cho DKT/GKT. Upstream GRKT mặc
định `--epoch 500` + `early_stop` (`GRKT/BaseModel.py:30`). Artifact **không lưu lịch sử train**
nên không thể chứng minh GRKT đã hội tụ ở epoch 5. GRKT vẫn thắng (+0,0107 assist), nhưng nếu
giảng viên hỏi "vì sao 5?" thì hiện chưa có bằng chứng.

**Cách sửa.** Ghi `history` (train loss, val AUC từng epoch) vào `summary.json` + vẽ một hình
learning curve; tăng `EPOCHS` lên ≥ 20 cho assist2009 (rẻ) và giữ early stopping. Nếu giữ 5 epoch
thì phải chứng minh đường val AUC đã phẳng.

### F5 — Provenance artifact yếu: `seconds = 0,0`, không có lịch sử, không có siêu tham số

* `level2_metrics.csv` cột `seconds` = **0,0 cho cả 8 dòng** (cả đường nền): `baselines.py` không
  trả `train_seconds`, còn GRKT nạp checkpoint nên `out["seconds"] = 0.0`
  (`grkt_api.py:669`). Cột này hiện vô nghĩa — bỏ hoặc sửa.
* `summary.json` không có block `config` (epochs, lr, k_hidden, k_hop, alpha, thresh, gap_cap,
  top_concepts), không có `resumed_from`, không có `history`.
* Hệ quả: không ai đọc artifact có thể biết kết quả sinh từ cấu hình nào — trái với tinh thần §8
  ("hướng dẫn chạy lại"). Sửa: thêm `config` + `history` + `resumed_from` vào `summary.json`.

### F6 — Không có biến thể đồ thị, nên "tiêu chí quỹ đạo" không có phép so

`trajectory_metrics_table` cố ý dùng **cùng** `adj` của đồ thị train cho mọi biến thể
(`trajectory.py:368-371`) để số đo so sánh được. Nhưng lần chạy này chỉ có một biến thể, nên bảng
chỉ có 1 dòng và không có phép so nào. Notebook §5.1 in "so `train` vs `zero`" trong khi `zero`
không tồn tại. Tối thiểu cần `VARIANTS=("train", "zero")` trên **assist2009** (2 lần train, không
phải 4) để có một cặp so thật; thêm `"shuffled"` nếu còn giờ.

### F7 — Dashboard Cấp 2 thiếu, và artifact Cấp 1 đang lệch mô tả

* Không có widget chọn người học cho mastery; `level2/trajectory.py` đã có sẵn
  `plot_mastery_heatmap(traj, ..., user_index=...)` và `load_trajectories(npz)` — chỉ cần một
  `ipywidgets.Dropdown` + `load_trajectories` là xong (không cần train lại).
* `outputs/level1/cases.md` vẫn là bản template cũ, dù `GAP_LEVEL1.md` ghi đã có
  `annotate_evidence`. Nếu báo cáo trích `cases.md` sẽ trích phải bản cũ.
* `outputs/level1/*.json` **không** phải LFS pointer (đã kiểm tra: `metrics.json` đọc ra JSON
  thật, 1.085 byte) — khác với ghi chú trong `GAP_LEVEL1.md`; có thể sửa lại ghi chú đó.

---

## 4. Việc cần làm, ưu tiên từng mức

### P0 — chặn điểm Cấp độ 2

1. **Sửa tiêu chí quỹ đạo (F1).**
   - Trong báo cáo: đổi nhãn `strengthen`/`spread`/`locality` thành *"kiểm chứng cài đặt (kỳ vọng
     = 1)"*, **không** dùng làm bằng chứng vai trò đồ thị; sửa 3 chỗ doc (`trajectory.py:163-167`,
     `:368-371`, `GIAI_THICH_LEVEL1_LEVEL2.md:717,964`).
   - Thêm **2 tiêu chí thật**: `precedence_consistency` (dùng `pre_map`) và
     `stability_under_perturbation` (lật/bỏ 1 bước cuối → Spearman giữa hai vector mastery).
     Cài trong `level2/trajectory.py`, thêm cột vào `trajectory_metrics.csv`, đo trên
     `traj_train.npz` có sẵn (không cần train lại nếu chỉ cần 1 biến thể; cần train lại nếu muốn
     so với `zero`).
   - Chạy lại bước B6 (hoặc một cell riêng đọc `traj_*.npz`) để sinh `trajectory_metrics.csv` mới.

2. **Có ít nhất một phép so biến thể đồ thị (F6).**
   - `RUN_GRAPH_ABLATIONS=True` nhưng giới hạn `VARIANTS=("train","zero")` cho **assist2009**
     trước (2 lần train); nếu còn ngân sách thêm `"shuffled"`.
   - So bằng đại lượng **có thể sai**: `auc`, `abs_delta_retrieval`, `abs_delta_learn_forget`,
     ECE/Brier, và 2 tiêu chí mới — **không** so `spread`/`locality` (F1).

3. **Đa seed cho GRKT (E2).**
   - `RUN_MULTI_SEED=True`, `SEEDS=[42,123,456]`, split cố định; ghi `multiseed_metrics.csv`
     và commit. Nếu thời gian không cho phép 3 seed × 2 tập, làm **assist2009 trước** và ghi rõ
     Junyi chỉ 1 seed.

4. **Gỡ mâu thuẫn seed của đường nền (F2).**
   - Sửa `kaggle_level1.ipynb` §4.5 (bỏ `save_dir` hoặc đặt tên `-seed{s}`) rồi chạy lại; hoặc
     cho Cấp 2 train lại đường nền (`baseline_ckpt_dir=None`).
   - Thêm vào `summary.json` `checkpoint_seed` cho từng dòng đường nền.

### P1 — bắt buộc theo đề / ảnh hưởng kết luận

5. **Ghi đủ provenance (F5).** `summary.json` thêm `config` (mọi siêu tham số + `epochs` +
   `resumed_from` + `split_seed`), `history` của GRKT/đường nền, và vẽ learning curve. Bỏ hoặc sửa
   cột `seconds`.
6. **Làm rõ Junyi (F3).** Ghi số dòng gốc / sau cap / sau lọc, tổng số khái niệm trước cap, và
   sửa README + notebook + `dataset_spec.py:128` cho khớp artifact (46.229 / 4,02 triệu).
   Thêm một câu khai báo "chọn top-200 dùng thống kê toàn bộ dữ liệu (transductive, không dùng nhãn)".
7. **Cân lại số epoch (F4).** `EPOCHS≥20` cho assist2009 (và cho GRKT ở Junyi nếu còn giờ) hoặc
   chứng minh đã hội tụ bằng learning curve.
8. **Dashboard Cấp 2 (F7/D4).** Thêm cell `ipywidgets.Dropdown` chọn `user_id` → hiện
   heatmap + đường mastery + bảng `m_pre/m_strong/m_tilde` quanh các bước đúng/sai, dùng
   `load_trajectories("traj_train.npz")`. Không cần GPU.
9. **Chạy lại Cấp 1 để cập nhật `cases.md`** (nếu báo cáo sẽ trích), hoặc ghi chú rõ trong báo cáo
   rằng case study lấy từ `level1/explain.collect_cases(..., evidence=True)` và artifact cũ hơn.

### P2 — trau chuốt / tuỳ chọn

10. **Calibration (E6).** `RUN_CALIBRATION=True` — rẻ, có ngay `calibration.csv`, `reliability.png`,
    `reliability_rows.csv`; là "một phân tích nâng cao" hợp lệ nếu phần quỹ đạo chưa kịp sửa xong.
11. **Đánh giá lịch sử ngắn / cold-start** (E7): cắt 50% đầu chuỗi test rồi đo lại AUC — một cột
    trong bảng, dễ làm.
12. **Vẽ lại heatmap cho dễ đọc**: hiện `junyi` rộng ~200 bước với nhãn là `ucid` thô
    (`s5t...`); cắt cửa sổ 50 bước và dùng tên bài từ `Info_Content.csv` nếu có.
13. **`environment.yml`** cho chạy local (không chặn nộp).
14. **Báo cáo + slide + bảng phân công** (D3) — dùng số liệu đã có trong `README.md` §Cấp độ 2 và
    file này; **phải** ghi rõ 3 điểm: (i) 3 tiêu chí quỹ đạo là kiểm chứng cài đặt, không phải
    bằng chứng; (ii) Junyi bị cap 200 khái niệm; (iii) đường nền là seed nào.

---

## 5. Tổng kết mức đạt

| Trạng thái | Số mục | Gồm |
|---|---|---|
| **THỎA** | 9 | 2 tập dữ liệu, GRKT 2024, split user-wise trùng Cấp 1, đồ thị train-only, không dùng nhãn tương lai, AUC/ACC/NLL toàn cục, trực quan mastery, split + leak report đã commit, mã + requirements + hướng dẫn |
| **MỘT PHẦN** | 5 | "Cùng quy trình" giữa 2 tập (min_interactions/cap khác nhau), pyKT (cap 200 transductive), dashboard chọn người học, case study 5+5 (artifact C1 cũ), phân tích nâng cao (chọn quỹ đạo nhưng quỹ đạo hỏng) |
| **CHƯA THỎA** | 5 | **≥2 tiêu chí hợp lý (F1)**, **bằng chứng vai trò đồ thị (F6)**, **3 seed cho GRKT (E2)**, **provenance/seed đường nền (F2)**, **báo cáo + slide + bảng phân công (D3)** |

**Điểm trần 8,5 cần:** tái lập 1 phương pháp 2024 trên ≥2 tập (đã có) **+** ≥2 tiêu chí hợp lý
(chưa) **+** báo cáo (chưa). Hai mục còn lại chặn điểm; mục F2 ảnh hưởng độ tin cậy của bảng số
liệu, nên sửa trước khi viết báo cáo.

---

## 6. Thứ tự thực hiện còn lại

1. Sửa F1 (thêm 2 tiêu chí thật + đổi nhãn 3 tiêu chí kia thành "kiểm chứng cài đặt") — **không
   cần GPU** nếu chỉ đo trên `traj_train.npz` có sẵn; cần GPU nếu muốn kèm biến thể `zero`.
2. Chạy `RUN_GRAPH_ABLATIONS=True` với `VARIANTS=("train","zero")` trên assist2009 + `EPOCHS≥20`
   + `RUN_MULTI_SEED=True` + `RUN_CALIBRATION=True` (một phiên Kaggle, ước tính thấp hơn lần đầu vì
   `train` đã có checkpoint).
3. Gỡ F2 (seed đường nền): sửa notebook Cấp 1 rồi chạy lại, hoặc train lại đường nền ở Cấp 2.
4. Bổ sung F5 (config + history vào `summary.json`), F3 (số liệu Junyi), F7 (dashboard).
5. Viết báo cáo 12–18 trang + slide + bảng phân công, trích đúng các cảnh báo ở mục 4 (P2-14).
6. (Cấp độ 3, ngoài phạm vi file này) 10 case xAI trên đồ thị + quiz 400 tương tác.

---

## 7. Nhật ký sửa P0 — 2026-11-01

Bốn mục P0 đã sửa **trong mã**. Chưa có số liệu mới: artifact phải được sinh lại trong một
phiên Kaggle GPU (máy local không có numpy/torch). Mọi file dưới đây đã qua `py_compile` /
parse AST của notebook; **chưa chạy được** vì thiếu môi trường.

### P0-1 (F1) — tiêu chí quỹ đạo

* `level2/trajectory.py`:
  * thêm `INSTALLATION_CHECK_COLUMNS = ("consistency_strengthen", "consistency_spread",
    "locality")`; docstring của `directional_consistency`/`locality` và module ghi rõ ba số này
    ≡ 1 theo cấu trúc (Positive_Linear + softmax + relu + `k_hop=1`), **không phải bằng chứng**.
  * `precedence_consistency(traj, pre_map, n_permutations=5, seed=42)` → `precedence`,
    `precedence_permuted`, `precedence_gap`, `precedence_tie_frac`, `n_precedence_edges`,
    `n_precedence_samples`. Baseline hoán vị giữ nguyên bậc ra của mỗi đỉnh.
  * `stability_under_perturbation(...)` → `stability_spearman`, `stability_changed_frac`,
    `stability_abs_delta`, `n_stability_users`; lật **bước áp chót** (`perturb="flip_penultimate"`
    trong `record_trajectories`) để còn một bước lan truyền.
  * `trajectory_metrics_table(..., pre_map=, stability=)` thêm các cột trên vào
    `trajectory_metrics.csv`.
* `level2/experiment.py`: B6 tính stability **riêng từng biến thể** rồi mới dựng bảng; in
  legend phân biệt "tiêu chí thật" và "kiểm chứng cài đặt"; `out["stability"]` được giữ lại.
* Notebook §5 + cell bảng: tách 3 bảng (tiêu chí thật / kiểm chứng cài đặt / phân tách giai
  đoạn), thêm bảng so biến thể + ΔAUC(train − zero).
* Doc: `GIAI_THICH_LEVEL1_LEVEL2.md` §3.8, §4.2, §6.2, §6.3; `README.md`;
  `KE_HOACH.md` — bỏ nhãn "4 tiêu chí" và "spread là tiêu chí thật".

### P0-2 (F6) — so biến thể đồ thị

* `kaggle_level2.ipynb` §0.2: `RUN_GRAPH_ABLATIONS=True`, `VARIANTS=("train","zero")`,
  `ABLATION_DATASETS=("assist2009",)`, `VARIANTS_BY_DATASET`, truyền `variants_by_dataset=` vào
  `run_level2`. Junyi giữ `train`.
* So sánh dùng `AUC`, `abs_delta_retrieval`, `abs_delta_learn_forget`, `precedence_gap`,
  `stability_*` — **không** dùng `spread`/`locality`.

### P0-3 (E2) — đa seed

* `RUN_MULTI_SEED=True`, `SEEDS=[42,123,456]`, `MULTI_SEED_DATASETS=("assist2009",)`; Junyi
  1 seed và notebook ghi rõ giới hạn.
* `level2/grkt_api.py`: tên model/checkpoint theo seed (`GRKT-train-seed123.pt`…); seed 42 giữ
  tên cũ.
* `level2/experiment.py` B8: truyền `save_dir`, `resume_ckpt`, `deadline`, `ckpt_every_min` cho
  từng seed (trước đây `save_dir=None` ⇒ không checkpoint, không resume, không dừng theo ngân
  sách); nếu `TrainingPaused`, ghi `multiseed_metrics.csv` cho các seed đã xong rồi mới raise.

### P0-4 (F2) — seed đường nền

* `level1/pipeline.py`: checkpoint seed ≠ 42 đặt tên `<tên>-seed{s}.pt`, không ghi đè
  `DKT.pt`/`GKT-*.pt` của seed 42.
* `level2/baselines.py`: `checkpoint_seed()` đọc `-seed{s}.pt` → `<tên>.json` (fingerprint) →
  `checkpoint_seeds.json`; kết quả có `seed`, `split_seed`, `ckpt_seed`, `ckpt_source`.
* `level2/experiment.py`: `TABLE_COLUMNS` + `_table_row` phát `checkpoint_seed` (và `seed=None`
  thay vì đoán 42); cột này vào cả `level2_metrics.csv`, `level2_all_datasets.csv` và
  `summary.json` (`models[]`).
* Notebook Cấp 2: `USE_COMMITTED_CKPTS=False` mặc định ⇒ đường nền train lại seed 42; cell §4
  in `checkpoint_seed`, cảnh báo nếu checkpoint là seed ≠ 42 và không so trực tiếp với số đã
  công bố; dung sai nới 0.02 khi train lại.

### Còn phải làm để P0 "đóng" thật

1. Push code lên `origin/main` (notebook Kaggle clone repo), rồi chạy lại `QUICK_RUN=True` để
   bắt lỗi thời gian chạy (`stability_*`, `precedence_*`, B8 resume).
2. Chạy báo cáo `QUICK_RUN=False` → sinh lại `level2_out/assist2009/*` và `junyi/*`. **Không**
   dùng artifact cũ: chúng không có `precedence_gap`/`stability_*`, và bảng cũ trộn seed.
3. Điền số thật vào báo cáo, ghi rõ: (i) 3 cột `consistency_*`/`locality` là kiểm chứng cài
   đặt; (ii) Junyi 1 biến thể + 1 seed; (iii) đường nền là **train lại seed 42**, không phải
   checkpoint Cấp 1 (seed 456).
4. P1 chưa sửa: F4 (`EPOCHS=5` cho GRKT, đường nền 20 epoch), F5 (provenance `config`/`history`),
   F3 (số liệu Junyi), F7 (dashboard) vẫn nguyên trong §4.

---

## 8. Kế hoạch resume trong phiên ~3 giờ còn lại (kiểm tra 2026-11-01)

### 8.1 Checkpoint cũ có dùng lại được không — **có**

Đọc trực tiếp `format`/`fingerprint` trong payload (không cần torch) của
`outputs/level2/*/GRKT-train.pt`:

| Tập | format | epochs | seed | n_train/val/test | edges rel/pre | best val AUC |
|---|---|---|---:|---|---|---|
| assist2009 | `grkt-final-v1` | 5 | 42 | 2330 / 776 / 778 | 2720 / 2642 | 0.8017 |
| junyi | `grkt-final-v1` | 5 | 42 | 27737 / 9245 / 9247 | 24908 / 14468 | 0.8151 |

Cả hai khớp đúng cấu hình notebook (`BATCH_SIZE=32`, `D_HIDDEN=128`, `K_HIDDEN=16`, `K_HOP=1`,
`ALPHA=0.01`, `TAU=0.2`, `THRESH=0.5`, `LR=1e-3`, `SPLIT_SEED=42`) và khớp số người trong
`split.json`. Vì vậy `RESUME_GRKT_CKPT=True` ⇒ `_check_fingerprint` qua và **bỏ qua huấn luyện**,
chỉ chạy lại phần đánh giá/quỹ đạo. Đây là lý do 27 giờ GPU không mất.

### 8.2 Việc sẽ chạy lại và chi phí ước tính

| Việc | Vì sao phải chạy lại | Ước tính |
|---|---|---|
| B4 GRKT `train` | không (nạp checkpoint) | ~0, chỉ eval |
| B4 GRKT `zero` (assist2009) | P0-2, chưa từng train | 5 epoch assist ≈ 10 phút |
| B5 đường nền assist2009 | P0-4, `outputs/level1/*.pt` là seed 456 | DKT + GKT-Transition + GKT-Zero × 20 epoch ≈ 20–30 phút |
| B5 đường nền junyi | **không** — `baselines/*.json` fingerprint khớp, nạp lại | ~0 |
| B6 quỹ đạo + `stability_*` | P0-1, code mới | assist ≈ 2–5 phút; junyi ≈ 5–10 phút |
| B7 calibration | không (vẫn tắt) | 0 |
| B8 seed 123/456 (assist2009) | P0-3, seed 42 dùng lại ckpt | ≈ 10–20 phút |

Tổng ≈ 50–70 phút, nằm gọn trong `TIME_BUDGET_MIN = 120` (đã hạ từ 600). Thứ tự tập là
`["assist2009", "junyi"]`: assist2009 xong trước nên dù junyi bị dừng vì ngân sách, toàn bộ
sản phẩm P0 vẫn còn; cell §3 tự đóng gói `level2_output.zip` khi `TrainingPaused`.

### 8.3 Các bước

1. **Push code P0 lên `origin/main`** (notebook clone repo; chưa push thì Kaggle vẫn chạy bản
   cũ, không có `precedence_gap`/`stability_*`/checkpoint theo seed). Re-import
   `kaggle_level2.ipynb` từ repo để lấy đúng cell §0.2/§3.
2. Upload `outputs/level2` thành private Kaggle Dataset với **thư mục gốc tên `level2_out`**
   (cell §0.3 đòi tiền tố này) — nếu còn `level2_output.zip` do Kaggle sinh ra thì attach thẳng
   file đó. Attach lại dataset Junyi gốc.
3. Chạy §0 → §0.3 (phải in `GRKT đã train xong (có fingerprint) -> chỉ đánh giá`), rồi §3.
4. Chấp nhận `TrainingPaused` nếu có: tải zip rồi chạy tiếp phiên sau.

### 8.4 Cấu hình phải giữ nguyên để resume không bị từ chối

`EPOCHS=5`, `BATCH_SIZE=32`, `MAX_SEQ_LEN=200`, `MAX_USERS=None`, `GRAPH_THRESH=0.6`,
`COLD_THRESH=5`, `SPLIT_SEED=42`, `D_HIDDEN/K_HIDDEN/K_HOP/ALPHA/LR` như cũ. Riêng bật
`RUN_GRAPH_ABLATIONS=True` + `VARIANTS=("train","zero")` (chỉ assist2009) và
`RUN_MULTI_SEED=True` **không** ảnh hưởng fingerprint của biến thể `train`.
