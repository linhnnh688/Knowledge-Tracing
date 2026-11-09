# Báo cáo kỹ thuật — Theo dõi tri thức người học bằng học máy trên đồ thị

**Đồ án Project 02 · Cấp độ 1 và Cấp độ 2**

- Ngày tổng hợp: 09/11/2026 (Cấp 1); 07/11/2026 (Cấp 2)
- Dữ liệu báo cáo: `results/level1_out/` (Cấp 1, artifact 2026-11-09) và
  `results/level2_moderate/` (Cấp 2, artifact 2026-11-07)
- Mã nguồn: commit `639c908` (lần chạy Cấp 1), commit `154d4e4` (lần chạy Cấp 2)
- Môi trường: Kaggle, Tesla T4, torch 2.11.0+cu130, numpy 2.1.3, pandas 2.3.3,
  scipy 1.16.3, scikit-learn 1.6.1, matplotlib 3.10.0

> **Nguyên tắc số liệu.** Mọi con số trong báo cáo được đọc lại từ CSV/JSON trên đĩa tại hai thư
> mục trên; không có số nào lấy từ bộ nhớ notebook hay ước lượng. Báo cáo không chạy thêm thực
> nghiệm nào. Các chỗ thiếu bằng chứng được ghi rõ là "chưa đạt" thay vì suy diễn.

---

## 0. Tóm tắt điều hành

**Bài toán.** Dự đoán xác suất người học trả lời đúng ở lần tương tác kế tiếp, đồng thời mô tả
trạng thái nắm vững theo từng khái niệm và giải thích dự đoán bằng lịch sử làm bài.

**Dữ liệu.** ASSISTments 2009 (4.217 người học thô, 525.534 dòng, 123 khái niệm) và một tập con
có kiểm soát của Junyi Academy (3.000 người, 251.598 lượt, 200 khái niệm). Cả hai chia theo người
học, seed 42, tỉ lệ 0,6/0,2/0,2; đồ thị chỉ xây từ tập train.

**Kết quả chính — Cấp 1 (ASSISTments 2009, 5 seed 42/123/456/789/1011):**

| Mô hình | Đồ thị | AUC mean ± std | AUC seed 42 |
|---|---|---:|---:|
| DKT | — | **0,7862 ± 0,0006** | **0,7866** |
| GKT-MHA | động (attention) | 0,7668 ± 0,0095 | 0,7782 |
| GKT-Transition | tĩnh | 0,7609 ± 0,0096 | 0,7427 |
| GKT-Shuffled | tĩnh (xáo cạnh) | 0,7579 ± 0,0039 | 0,7542 |
| GKT-Identity | tĩnh (ma trận đơn vị) | 0,7578 ± 0,0056 | 0,7549 |
| GKT-Zero | không cạnh | 0,7576 ± 0,0055 | 0,7549 |
| GKT-ShuffledRow | tĩnh (xáo theo hàng) | 0,7576 ± 0,0127 | 0,7651 |

**Kết quả chính — Cấp 2 (seed 42):**

| Mô hình | ASSISTments 2009 AUC | Junyi (tập con) AUC |
|---|---:|---:|
| **GRKT-train (KDD 2024)** | **0,7972** | **0,7896** |
| DKT (train lại trong harness C2) | 0,7859 | 0,7501 |
| GRKT-zero (ablation) | 0,7840 | — |
| GKT-Zero (train lại, C2) | 0,7644 | 0,7225 |
| GKT-Transition (train lại, C2) | 0,7536 | 0,7202 |

- **Cấp 1 — dự đoán:** DKT thắng **mọi** biến thể GKT, chênh +0,0194 đến +0,0286 AUC, cả 6 so
  sánh đều có ý nghĩa (p = 0,0002–0,014; 5 seed ghép cặp). Thời gian: DKT 31–37 s/seed, GKT tĩnh
  1.379–3.789 s, MHA 2.009–4.345 s.
- **Cấp 1 — ablation lúc train:** không biến thể đồ thị nào khác Zero có ý nghĩa
  (Transition − Zero = +0,0033, p = 0,546; Transition − Shuffled = +0,0030, p = 0,495;
  ShuffledRow − Zero ≈ 0, p = 0,998) → **chưa chứng minh được vai trò cấu trúc của đồ thị**.
- **Cấp 1 — ablation lúc suy luận:** đặt `A := 0` làm Transition −0,0109 và ShuffledRow −0,0109
  AUC, nhưng Shuffled chỉ −0,0031 và MHA −0,0029; Identity/Zero đúng bằng 0. Mô hình **có đọc**
  ma trận kề, nhưng độ lớn này đo mức phụ thuộc vào `A`, không đo chất lượng `A`.
- **Cấp 2:** GRKT vượt mọi đường nền trên cả hai tập (+0,0113 AUC so với DKT trên ASSISTments;
  +0,0395 trên Junyi). Bỏ đồ thị làm giảm 0,0132 AUC trên ASSISTments (một seed). GRKT đạt
  0,7983 ± 0,0009 AUC qua 3 seed trên ASSISTments.
- **Quỹ đạo tri thức:** độ ổn định cao (Spearman 0,971–0,994) và có lan truyền qua đồ thị
  (`abs_delta_retrieval` 0,033 so với 0 khi bỏ cạnh), nhưng **`precedence_gap` âm trên cả hai
  tập** → tiêu chí nhất quán tiên quyết **không đạt**. Tín hiệu "quên" trên Junyi gần như bằng 0
  vì 88,1 % khoảng thời gian bằng 0 (giới hạn dữ liệu, không phải kết quả mô hình).

**Ba điều phải nói ngay khi bảo vệ:** (1) Junyi chỉ là tập con 4,1 % người học, không đại diện
toàn bộ Junyi; (2) bằng chứng lợi ích đồ thị ở Cấp 2 chỉ từ một seed, chưa phải kết luận nhân
quả; (3) Cấp 1 đã có đủ ablation nhưng kết quả **phủ định** — đồ thị không giúp mô hình tốt hơn,
và bằng chứng case study phải sinh lại trước khi dùng.

---

## 1. Giới thiệu và phạm vi

### 1.1 Mục tiêu

Đồ án gồm ba nhiệm vụ nối tiếp: dự đoán (đúng/sai), mô tả năng lực (mastery theo khái niệm và
thời gian), và giải thích (bằng chứng lịch sử + khái niệm liên quan). Báo cáo này tổng hợp kết
quả thực nghiệm của **Cấp độ 1** (baseline không đồ thị + mô hình đồ thị + ablation + 5 đúng/5
sai) và **Cấp độ 2** (một phương pháp graph-aware công bố từ 2024, hai tập dữ liệu, tiêu chí hợp
lý của quỹ đạo, trực quan hoá và một phân tích nâng cao).

### 1.2 Phạm vi

- **Trong phạm vi:** ASSISTments 2009, tập con Junyi; DKT, GKT (tĩnh và MHA), GRKT (KDD 2024);
  đánh giá AUC/Accuracy/NLL, đa seed, ablation đồ thị, calibration, quỹ đạo mastery, dashboard
  người học, 10 case giải thích.
- **Ngoài phạm vi:** Cấp độ 3 (quiz tự thu thập, 10 case đồ thị nâng cao) và Cấp độ 4 (mở rộng
  tự chọn). Báo cáo không tuyên bố kết quả cho hai cấp này.

### 1.3 Hai lần chạy Cấp 1 — nguồn số liệu

Repo có **hai bộ artifact Cấp 1** từ hai lần chạy khác nhau:

1. `results/level1_out/` — lần chạy **2026-11-09** (notebook
   `2026-11-08 level1-full+ablation-extend`, commit `639c908`): **7 mô hình × 5 seed**, đủ
   ablation lúc train và lúc suy luận, dashboard, calibration. **Đây là nguồn chính và duy nhất
   của báo cáo.**
2. `outputs/level1/` — lần chạy cũ (3 seed, 5 biến thể GKT tĩnh, không có MHA). Chỉ còn giá trị
   lịch sử; **không dùng số từ đây** trong báo cáo.

Hai bộ không được trộn vào cùng một bảng. Số DKT giữa hai lần chạy gần như trùng nhau
(0,7866 so với 0,7860 trung bình 3 seed), nhưng bộ mô hình GKT khác nhau. Lần chạy mới đã thay
thế hoàn toàn nhu cầu dùng `outputs/level1/` cho ablation.

---

## 2. Dữ liệu

### 2.1 ASSISTments 2009

| Thuộc tính | Giá trị |
|---|---|
| Tệp | `GKT/data/skill_builder_data.csv` |
| SHA-256 | `f4b8791ae5968b884b77ab3b08adec48130503ec57f3113158c9a5af88af3374` |
| Kích thước | 79,35 MB · 525.534 dòng |
| Người học thô | 4.217 |
| Khái niệm | 123 |
| Câu hỏi | 26.688 |
| Tỉ lệ làm đúng | 0,6795 |
| Dòng thiếu mã khái niệm | 66.326 |
| Người học < 3 tương tác | 269 |

Pipeline (`GKT/processing.py:92-95`) bỏ trước 66.326 dòng thiếu `skill_id`, rồi bỏ người còn dưới
3 tương tác, sau đó cắt độ dài tối đa 200. Kết quả dùng cho mô hình: **3.884 người học / 237.943
tương tác** (khớp giữa hai cấp). Con số 3.884 thấp hơn 4.217 − 269 = 3.948 vì một số người chỉ
còn dưới 3 dòng sau khi bỏ các dòng thiếu khái niệm. Mỗi câu hỏi gắn **một** khái niệm, nên đồ thị chuyển tiếp giữa các khái niệm
rất thưa: chỉ **884 cạnh** sau khi bỏ self-loop (mật độ 0,0589).

Hệ quả khoa học: ASSISTments 2009 là môi trường **khó** để chứng minh lợi ích của cấu trúc đồ thị,
vì phần lớn bước chuyển là "cùng khái niệm" hoặc không có cạnh. Báo cáo phải diễn giải kết quả
Cấp 1 trong bối cảnh này.

### 2.2 Junyi Academy (tập con có kiểm soát)

| Thuộc tính | Giá trị |
|---|---|
| Nguồn | `Log_Problem.csv` (Junyi Academy public dataset) |
| Người học thô / lượt thô | 72.630 / 16.217.311 |
| Tập con dùng | **3.000 người / 251.598 lượt** (4,1 % người; 1,55 % lượt) |
| Khái niệm | 200 (chọn theo tần suất **toàn bộ log**) |
| Câu hỏi | 3.888 |
| Điều kiện | `min_interactions=10`, tối đa 500 dòng/người, lấy mẫu người học seed 42 |
| Split | 1.800 / 600 / 600 người; 152.579 / 49.575 / 49.444 lượt |

Hai giới hạn phải khai báo:

1. **Tập con, không phải toàn bộ Junyi.** Mọi số Junyi chỉ có giá trị trên 3.000 người được lấy
   mẫu; không được viết "GRKT đạt 0,7896 trên Junyi".
2. **Chọn 200 khái niệm là transductive.** Tần suất khái niệm được tính trên toàn bộ log trước
   khi chia train/val/test. Đồ thị và nhãn vẫn chỉ lấy từ train, nhưng không gian khái niệm đã
   "nhìn thấy" toàn bộ dữ liệu. Đây là giới hạn của thiết kế dữ liệu, không phải rò rỉ nhãn.

Ngoài ra, meta ghi `rows=340.601` trong khi thống kê dùng 251.598 tương tác. Đã đối chiếu lại:
340.601 chính là số dòng CSV chuẩn hoá **sau** khi cắt tối đa 500 dòng/người và lọc
`min_interactions=10` (3.000 người, không ai dưới 10 tương tác). Chênh lệch 89.003 dòng đến từ
bước cắt `max_seq_len=200` trong `level2/grkt_data.py`: 538 người có chuỗi dài hơn 200 bị cắt, và
`Σ min(số_bước, 200) = 251.598` khớp đúng số tương tác dùng cho mô hình. Hai số không mâu thuẫn,
nhưng phải ghi rõ "340.601 dòng chuẩn hoá" và "251.598 tương tác sau cắt 200" khi trích dẫn.

**Giới hạn thời gian của Junyi:** timestamp được làm tròn 15 phút, `dt_p50 = 0`, `dt_p95 = 240`
phút, `dt_zero_frac = 0,881`. Phần lớn Δt bằng 0 hoặc bị chặn trần 240 phút, nên mọi kết luận
về "quên" trên Junyi là **giới hạn dữ liệu**.

### 2.3 Chia dữ liệu và chống rò rỉ

Cả hai cấp dùng cùng hàm chia theo người học, seed 42, tỉ lệ 0,6/0,2/0,2. Kết quả:

| Tập | ASSIST C1 | ASSIST C2 | Junyi C2 |
|---|---:|---:|---:|
| Train | 2.330 người / 142.299 lượt | 2.330 / 142.299 | 1.800 / 152.579 |
| Validation | 776 / 47.508 | 776 / 47.508 | 600 / 49.575 |
| Test | 778 / 48.136 | 778 / 48.136 | 600 / 49.444 |
| Giao ba tập | 0 | 0 | 0 |

Các kiểm tra chống rò rỉ đã chạy và ghi trong `leak_report.txt`:

- **Tách theo người học:** không người học nào xuất hiện ở hai tập (assert + đối chiếu).
- **Đồ thị chỉ từ train:** ASSIST 884 cạnh; khối lượng chuyển tiếp chỉ quan sát ở val+test là
  189/25.554 = **0,74 %** và **không** được đưa vào đồ thị; 10.110/10.299 (98,2 %) chuyển tiếp
  của người ngoài train rơi trên cạnh có trong train. Với GRKT, **trước ngưỡng hoá** đồ thị dựng
  từ train có 14.024 cạnh (rel+pre), dựng từ val chỉ còn 11.081; sau ngưỡng 0,6 còn
  2.720 + 2.642 = **5.362 cạnh** → đồ thị phụ thuộc tập huấn luyện.
- **Không dùng nhãn tương lai:** đặc trưng tại bước `t` chỉ chứa kết quả đến `t`; nhãn là bước
  `t+1`. GRKT chỉ dùng `knows[i+1]` và `times[i+1]` (danh tính và thời điểm câu kế tiếp, **không**
  dùng kết quả) cho pha học/quên.
- **Test loader không shuffle**, nên vị trí case khớp dữ liệu.

Về `same_split=null` trong `diagnostics/level1_linkage.json`: phiên Kaggle đọc
`outputs/level1/split.json` dưới dạng **git-lfs pointer** nên không so được nội dung. Kiểm tra
lại trên working tree hiện tại (file đã có nội dung thật), ba danh sách `train_users` /
`val_users` / `test_users` của `results/level1_out/split.json`,
`outputs/level1/split.json` và `results/level2_moderate/assist2009/split.json` **trùng khít từng
phần tử** (2.330 / 776 / 778). Vậy "cùng split theo người" giờ đã có bằng chứng nội dung file,
không chỉ bằng số đếm. Khi nộp nên dùng `results/level2_moderate/assist2009/split.json` làm file
split tham chiếu.

---

## 3. Phương pháp

### 3.1 DKT — baseline không đồ thị

DKT dùng LSTM trên chuỗi đặc trưng `(khái niệm, đúng/sai)` đã dịch một bước, đầu ra là xác suất
đúng cho câu kế tiếp. Cấu hình Cấp 1: ẩn 32, embedding 32, 39.899 tham số, 20 epoch, early stop
theo validation AUC. DKT là mốc so sánh bắt buộc vì nó **không** dùng bất kỳ cấu trúc khái niệm
nào.

### 3.2 GKT — đồ thị tĩnh và biến thể MHA

GKT duy trì trạng thái ẩn theo từng khái niệm và có hai pha:

- **Aggregate:** lan truyền trạng thái của các khái niệm lân cận qua ma trận kề `A` (Transition
  hoặc Zero), kèm embedding câu hỏi.
- **Update:** cập nhật trạng thái của khái niệm vừa làm bằng GRU.

Các biến thể được dùng trong báo cáo (đều là một khái niệm/câu hỏi cho ASSISTments 2009):

- `GKT-Transition`: ma trận kề đếm chuyển tiếp khái niệm, chỉ xây từ train, chuẩn hoá hàng
  (884 cạnh, mật độ 0,0589).
- `GKT-Zero`: bỏ toàn bộ cạnh (`A = 0`) — ablation "không đồ thị".
- `GKT-Identity`: ma trận đơn vị (123 cạnh, toàn đường chéo) — ablation theo yêu cầu "ma trận
  đơn vị". Vì `_agg_neighbors` ghi đè vị trí nút bằng `self_features`, cạnh tự nối không thêm
  thông tin, nên Identity trùng Zero về mặt cơ chế.
- `GKT-Shuffled`: giữ nguyên số cạnh nhưng xáo đích toàn cục (814 cạnh khác 0).
- `GKT-ShuffledRow`: xáo đích trong từng hàng, giữ out-degree của mỗi khái niệm (884 cạnh) —
  kiểm tra liệu "đúng hàng, sai đích" có quan trọng hơn "đúng cấu trúc".
- `GKT-MHA`: thay ma trận kề tĩnh bằng **attention đa đầu** trên embedding khái niệm
  (`GKT/models.py:253-292`). Mô hình **không** có `A` tĩnh.

Ablation có **hai loại, không thay thế nhau**:

1. **Lúc train** (`Transition` vs `Zero`/`Identity`/`Shuffled`/`ShuffledRow`): huấn luyện lại
   từ đầu với từng ma trận kề. Đây là ablation trả lời trực tiếp yêu cầu §5/§7.
2. **Lúc suy luận** (`level1/explain.py::graph_off`): trên checkpoint đã train, đặt `A := 0` nếu
   `graph` là `nn.Parameter`; với MHA thì hook `graph_model` trả đầu ra bằng 0 — tức bỏ toàn bộ
   cạnh mà attention suy ra, chỉ còn self-update. Loại này đo "mô hình phụ thuộc bao nhiêu vào
   adjacency đang có", không đo "adjacency có đúng không".

Cần phân biệt rõ: **GKT-MHA của Cấp 1 không phải GKT-Transition/Zero của Cấp 2.** Chúng là hai
kiến trúc khác nhau, train ở hai lần chạy khác nhau; không được gọi chung là "baseline GKT".
Đường nền Cấp 2 dùng `hid_dim = emb_dim = 32` (kế thừa Cấp 1, `level2/baselines.py:36-37`,
`:537-538`), trong khi GRKT dùng `d_hidden = 128` — một bất đối xứng dung lượng cần ghi nhớ khi
so sánh. Cấp 2 **train lại** đường nền trong harness riêng (`use_committed_ckpts=false` trong
`run_config.json`; `baselines/*.json` ghi `epochs: 20`), không nạp checkpoint Cấp 1. Vì vậy DKT
Cấp 2 = 0,7859 khác DKT Cấp 1 = 0,7866 (seed 42) / 0,7862 ± 0,0006 (5 seed) dù cùng split và
cùng cấu hình — đây là khác biệt lần train, không phải khác dữ liệu. Tương tự, GKT-Transition
Cấp 2 = 0,7536 và GKT-Zero = 0,7644 (1 seed, train lại) khác Cấp 1 = 0,7609 ± 0,0096 và
0,7576 ± 0,0055 (5 seed).

### 3.3 GRKT — phương pháp graph-aware 2024

GRKT (KDD 2024) tách quá trình theo dõi tri thức thành bốn pha:

1. **Retrieval:** với câu hỏi hiện tại, tính "nhu cầu khái niệm" và lan truyền trạng thái trên ba
   quan hệ: `rel` (đồng xuất hiện), `pre` (tiên quyết), `sub` (phụ thuộc, bằng `preᵀ`).
2. **Dự đoán:** gộp trạng thái các khái niệm liên quan thành mastery, trừ một độ lệch học được,
   rồi qua sigmoid. Điểm quan trọng: dự đoán được tính **trước** khi dùng kết quả hiện tại.
3. **Strengthening:** nếu đúng, gain lan sang khái niệm lân cận; nếu sai, loss lan tương tự.
4. **Learning–forgetting:** dùng Δt giữa hai lần làm để mô hình hoá "học ngấm" và quên theo hàm
   mũ; tốc độ học/quên của từng khái niệm được suy ra từ cấu trúc đồ thị.

Đồ thị của GRKT:

- `rel_map[i,j]` = tỉ lệ hai khái niệm cùng đúng hoặc cùng sai, chỉ đếm trên train.
- `pre_map[i,j]` = trong các cặp có kết quả lật, tỉ lệ lật theo hướng `i sai → j đúng`; đây là
  heuristic đếm, không phải quan hệ nhân quả.
- Ngưỡng `graph_thresh = 0,6`, `k_hop = 1`, ba quan hệ `rel`/`pre`/`sub`.

Cấu hình Cấp 2: `d_hidden = 128`, `k_hidden = 16`, `alpha = 0,01`, `lr = 1e-3`, batch 32, tối đa
20 epoch, patience 5, Δt chặn trần 240 phút. ASSISTments có 2.720 cạnh `rel` và 2.642 cạnh `pre`;
Junyi có 17.866 và 10.173.

### 3.4 Chỉ số và kiểm định

- **Dự đoán:** AUC, Accuracy, NLL/BCE; calibration mở rộng bằng Brier, ECE, ACE và reliability
  diagram. Vì nhãn dịch một bước, mỗi người học mất một dự đoán ở bước đầu: ASSISTments còn
  47.358 dự đoán từ 48.136 tương tác test (chênh đúng 778 = số người test), Junyi còn 48.844 từ
  49.444 (chênh đúng 600).
- **Độ ổn định:** trung bình ± độ lệch chuẩn qua seed. Cấp 1 có 5 seed cho **7 mô hình** (35 run)
  và kiểm định ghép cặp theo seed (paired t-test, bootstrap CI, Cohen's dz). Cấp 2 có 3 seed cho
  GRKT trên ASSISTments, nhưng baseline chỉ 1 seed nên **không** có kiểm định ghép cặp
  GRKT–baseline.
- **Quỹ đạo:** 6 tiêu chí "có thể sai thật" — `precedence`, `precedence_gap`,
  `stability_spearman`, `stability_changed_frac`, `forget_slope`, `pearson_dt_dm` — tách khỏi 3
  "kiểm chứng cài đặt" luôn bằng 1 do cấu trúc mô hình (`consistency_strengthen`,
  `consistency_spread`, `locality`).

---

## 4. Kết quả Cấp độ 1

### 4.1 Kết quả chính (seed 42)

| Mô hình | Đồ thị | Test AUC | Test Acc | Test NLL | Best Val AUC | Tham số | Thời gian |
|---|---|---:|---:|---:|---:|---:|---:|
| DKT | — | **0,7866** | **0,7340** | **0,5272** | 0,7922 | 39.899 | 35,1 s |
| GKT-MHA | động (MHA) | 0,7782 | 0,7306 | 0,5374 | 0,7836 | 38.236 | 4.072,8 s |
| GKT-ShuffledRow | tĩnh (xáo theo hàng) | 0,7651 | 0,7194 | 0,5462 | 0,7756 | 34.140 | 3.789,0 s |
| GKT-Identity | tĩnh (đơn vị) | 0,7549 | 0,7082 | 0,5570 | 0,7658 | 34.140 | 2.755,5 s |
| GKT-Zero | không cạnh | 0,7549 | 0,7082 | 0,5570 | 0,7658 | 34.140 | 2.912,4 s |
| GKT-Shuffled | tĩnh (xáo cạnh) | 0,7542 | 0,6980 | 0,5675 | 0,7671 | 34.140 | 3.409,4 s |
| GKT-Transition | tĩnh | 0,7427 | 0,7012 | 0,5740 | 0,7529 | 34.140 | 1.379,0 s |

Ở seed 42, DKT hơn mô hình GKT mạnh nhất (MHA) **0,0084 AUC** và hơn GKT-Transition
**0,0439 AUC**. GKT-Transition là mô hình tệ nhất bảng; ShuffledRow (cùng 884 cạnh nhưng xáo
đích trong từng hàng) lại tốt nhất trong nhóm tĩnh. Chênh lệch giữa các biến thể tĩnh lớn hơn
chênh lệch với DKT, cho thấy phương sai giữa các seed quan trọng — phải đọc bảng 5 seed bên dưới.

Ghi chú: cột NLL ở bảng chính là **trung bình loss theo batch** (`level1/pipeline.py::_epoch_loop`),
không phải BCE toàn cục. Vì vậy NLL ở đây lệch ~0,002 so với `calibration.csv` và
`graph_inference_ablation.csv` (xem §4.5).

### 4.2 Đa seed và kiểm định ghép cặp

| Mô hình | Số seed | AUC mean ± std | Acc mean ± std | NLL mean ± std |
|---|---:|---|---|---|
| DKT | 5 | **0,7862 ± 0,0006** | 0,7343 ± 0,0013 | **0,5273 ± 0,0005** |
| GKT-MHA | 5 | 0,7668 ± 0,0095 | 0,7213 ± 0,0087 | 0,5505 ± 0,0088 |
| GKT-Transition | 5 | 0,7609 ± 0,0096 | 0,7127 ± 0,0077 | 0,5567 ± 0,0105 |
| GKT-Shuffled | 5 | 0,7579 ± 0,0039 | 0,7094 ± 0,0064 | 0,5594 ± 0,0063 |
| GKT-Identity | 5 | 0,7578 ± 0,0056 | 0,7099 ± 0,0077 | 0,5546 ± 0,0068 |
| GKT-Zero | 5 | 0,7576 ± 0,0055 | 0,7098 ± 0,0077 | 0,5562 ± 0,0073 |
| GKT-ShuffledRow | 5 | 0,7576 ± 0,0127 | 0,7113 ± 0,0121 | 0,5555 ± 0,0119 |

| So sánh (AUC) | Chênh lệch | CI95 | t | p | Cohen's dz |
|---|---:|---|---:|---:|---:|
| DKT − Transition | +0,02536 | [+0,01158, +0,03914] | 5,110 | 0,0069 | 2,285 |
| DKT − Identity | +0,02838 | [+0,02034, +0,03641] | 9,807 | 0,0006 | 4,386 |
| DKT − Zero | +0,02862 | [+0,02084, +0,03640] | 10,216 | 0,0005 | 4,569 |
| DKT − Shuffled | +0,02832 | [+0,02267, +0,03397] | 13,913 | 0,0002 | 6,222 |
| DKT − MHA | +0,01943 | [+0,00649, +0,03237] | 4,169 | 0,0140 | 1,864 |
| DKT − ShuffledRow | +0,02863 | [+0,01054, +0,04672] | 4,395 | 0,0117 | 1,965 |

Kết luận thống kê: với 5 seed, DKT tốt hơn **mọi** biến thể GKT ở AUC (p = 0,0002–0,014; cỡ hiệu
ứng |dz| = 1,86–6,22) và ở NLL (p ≤ 0,01). Khoảng cách nhỏ nhất là với MHA (+0,0194), lớn nhất
là với ShuffledRow/Zero (+0,0286). Phương sai của các GKT (std 0,004–0,013) lớn gấp 7–23 lần DKT
(0,0006), nên từng seed có thể dao động mạnh: Transition seed 42 = 0,7427 nhưng seed 456 =
0,7695; ShuffledRow seed 1011 = 0,7333. Báo cáo chỉ dùng trung bình 5 seed cho kết luận, không
chọn seed có lợi.

Kiểm định giữa các biến thể đồ thị — **không so sánh nào có ý nghĩa**:

| So sánh | ΔAUC | p |
|---|---:|---:|
| Transition − Zero | +0,00326 | 0,5462 |
| Transition − Identity | +0,00302 | 0,5705 |
| Transition − Shuffled | +0,00296 | 0,4952 |
| Transition − ShuffledRow | +0,00328 | 0,7403 |
| Shuffled − Zero | +0,00030 | 0,8752 |
| ShuffledRow − Zero | −0,00001 | 0,9984 |
| Identity − Zero | +0,00024 | 0,3050 |

Đây là bằng chứng trực tiếp cho câu "đồ thị có giúp gì không": **không**. Transition không hơn
Zero, xáo cạnh không làm giảm hiệu năng, và ma trận đơn vị trùng Zero (xem §4.3).

### 4.3 Ablation lúc train — đã đủ, kết quả phủ định

`results/level1_out/` train **5 biến thể tĩnh** (`Transition`, `Identity`, `Zero`, `Shuffled`,
`ShuffledRow`) × 5 seed, cộng MHA. Bảng thống kê đồ thị:

| Đồ thị | Số cạnh khác 0 | Mật độ | Đường chéo khác 0 | Ghi chú |
|---|---:|---:|---:|---|
| Transition | 884 | 0,0589 | 0 | đồ thị chuyển tiếp từ train |
| ShuffledRow | 884 | 0,0589 | 0 | xáo đích trong từng hàng, giữ out-degree |
| Shuffled | 814 | 0,0542 | 0 | xáo đích toàn cục |
| Identity | 123 | 0,0082 | 123 | ma trận đơn vị |
| Zero | 0 | 0 | 0 | không cạnh |
| Dense | 15.006 | 1,0 | 0 | không train trong lần này |

Quy về AUC 5 seed (bảng đầy đủ ở §4.2):

| So sánh | ΔAUC | p | Kết luận |
|---|---:|---:|---|
| Transition − Zero | +0,00326 | 0,5462 | không ý nghĩa |
| Transition − Identity | +0,00302 | 0,5705 | không ý nghĩa |
| Transition − Shuffled | +0,00296 | 0,4952 | không ý nghĩa |
| Transition − ShuffledRow | +0,00328 | 0,7403 | không ý nghĩa |
| Shuffled − Zero | +0,00030 | 0,8752 | không ý nghĩa |
| ShuffledRow − Zero | −0,00001 | 0,9984 | không ý nghĩa |
| Identity − Zero | +0,00024 | 0,3050 | không ý nghĩa |

Đọc trung thực:

- **Không có biến thể đồ thị nào hơn "không cạnh" có ý nghĩa.** Transition (đồ thị đúng) không
  hơn Zero; xáo cạnh cũng không làm giảm hiệu năng.
- **Identity ≡ Zero** (seed 42 trùng đến ~1e-8): `_agg_neighbors` ghi đè vị trí nút bằng
  `self_features` (`GKT/models.py`), nên cạnh tự nối của ma trận đơn vị không thêm thông tin.
  Ở các seed có resume, cặp này lệch rất nhỏ (seed 456: 0,76195 vs 0,76090; seed 1011: 0,75941
  vs 0,75925) do số học lần resume, không phải kiến trúc.
- **ShuffledRow ≈ Transition** (+0,0033, p = 0,740) dù đích của mọi cạnh đã bị xáo trong từng
  hàng: cấu trúc *đích* của cạnh chuyển tiếp không tạo khác biệt đo được.

**Kết luận Cấp 1 về vai trò cấu trúc đồ thị:** chưa chứng minh được, và đây là kết quả phủ định
có kiểm định (5 seed, 7 so sánh ghép cặp), không phải "thiếu ablation". Nguyên nhân hợp lý:
ASSISTments 2009 có một khái niệm/câu hỏi và đồ thị chuyển tiếp rất thưa (884 cạnh), nên GKT
gần như quy về cơ chế self-update. Bằng chứng cho lợi ích đồ thị được chuyển sang Cấp 2, nơi
GRKT có hai loại quan hệ và ablation `train` vs `zero` trong cùng một mô hình.

### 4.4 Ablation lúc suy luận — mô hình có đọc `A`

Trên checkpoint seed 42, đặt `A := 0` (với MHA: hook `graph_model` trả 0) rồi đánh giá lại
(`tables/graph_inference_ablation.csv`):

| Mô hình | Loại | AUC có đồ thị | AUC khi `A := 0` | ΔAUC | ΔNLL | mean·\|Δp\| | max·\|Δp\| |
|---|---|---:|---:|---:|---:|---:|---:|
| GKT-Transition | tĩnh | 0,7427 | 0,7317 | **−0,0109** | +0,0080 | 0,0259 | 0,4933 |
| GKT-ShuffledRow | tĩnh | 0,7651 | 0,7542 | **−0,0109** | +0,0080 | 0,0248 | 0,6868 |
| GKT-Shuffled | tĩnh | 0,7542 | 0,7511 | −0,0031 | +0,0019 | 0,0075 | 0,4913 |
| GKT-MHA | động | 0,7454 | 0,7425 | −0,0029 | +0,0021 | 0,0026 | 0,1806 |
| GKT-Identity | tĩnh | 0,7549 | 0,7549 | 0 | 0 | 0 | 0 |
| GKT-Zero | tĩnh | 0,7549 | 0,7549 | 0 | 0 | 0 | 0 |

Cách đọc đúng:

- Mô hình **có khai thác** ma trận kề: bỏ `A` làm giảm AUC và tăng NLL ở 4/6 biến thể, với
  Transition/ShuffledRow giảm ~0,011 AUC.
- Nhưng **độ lớn gần như giống nhau giữa Transition và ShuffledRow** — hai đồ thị có cùng số cạnh
  và cùng out-degree nhưng đích khác hoàn toàn. Vì vậy Δ này đo **mức phụ thuộc vào adjacency**,
  không đo **chất lượng adjacency**. Kết hợp với §4.3: mô hình đọc `A`, nhưng `A` đúng không tốt
  hơn `A` xáo hay `A = 0`.
- Identity/Zero đúng bằng 0 là kiểm chứng cài đặt (không có cạnh để bỏ).
- **Cảnh báo số liệu:** dòng MHA dùng checkpoint cho AUC 0,7454, không phải 0,7782 của bảng chính
  (§4.5). Dòng này tự nhất quán trong nội bộ nhưng không phải trọng số của bảng §4.1.

### 4.5 Calibration

`tables/calibration.csv`, n = 47.358, tỉ lệ đúng thực 0,6263:

| Mô hình | AUC | NLL (toàn cục) | Brier | ECE | ACE | mean_pred |
|---|---:|---:|---:|---:|---:|---:|
| DKT | 0,7866 | 0,5249 | 0,1765 | 0,0154 | 0,0185 | 0,6348 |
| GKT-ShuffledRow | 0,7651 | 0,5440 | **0,1841** | **0,0105** | 0,0117 | 0,6189 |
| GKT-Identity / Zero | 0,7549 | 0,5546 | 0,1886 | 0,0223 | 0,0274 | 0,6103 |
| GKT-Shuffled | 0,7542 | 0,5652 | 0,1933 | 0,0706 | 0,0636 | 0,5559 |
| GKT-Transition | 0,7427 | 0,5725 | 0,1951 | 0,0507 | 0,0592 | 0,6758 |
| GKT-MHA | **0,7454** | 0,5645 | 0,1927 | 0,0442 | 0,0323 | 0,6705 |

- **Hai mô hình hiệu chỉnh tốt nhất:** DKT (ECE 0,0154) và ShuffledRow (ECE 0,0105). DKT cũng có
  Brier tốt nhất (0,1765) và mean_pred 0,6348 sát tỉ lệ đúng thực 0,6263.
- **Shuffled và Transition hiệu chỉnh kém:** ECE 0,0706 và 0,0507; mean_pred 0,5559 và 0,6758
  lệch khỏi 0,6263 theo hai hướng ngược nhau.
- **Cảnh báo MHA:** AUC ở bảng calibration là **0,7454**, không khớp 0,7782 của bảng chính, dù
  cùng n và cùng tỉ lệ nhãn. Sáu mô hình còn lại khớp tuyệt đối giữa hai bảng, nên đây là vấn đề
  riêng của checkpoint MHA. Dữ kiện: best val AUC của MHA rơi ở epoch 18 (0,7836), còn epoch 20
  chỉ 0,7622; file `.pt` lại chỉ là `state_dict` trần, không có `epoch`/`best_state`, nên không
  xác nhận được nguyên nhân từ artifact. Báo cáo **không dùng** calibration MHA để kết luận.
- **Hai định nghĩa NLL:** cột NLL ở §4.1 là trung bình loss theo batch; bảng này là BCE toàn cục
  trên toàn bộ 47.358 dự đoán. Chênh lệch ~0,002 và không đổi thứ hạng (ví dụ DKT 0,52719 so với
  0,52494). Khi trích dẫn phải ghi rõ đang dùng định nghĩa nào.

### 4.6 Case study 5 đúng + 5 sai — đủ số lượng, bằng chứng chưa dùng được

10 case, mỗi bucket 5 **khái niệm khác nhau**, tất cả do **GKT-Transition** dự đoán nên danh sách
"graph neighbors" đúng là đầu vào của mô hình:

- Đúng: Venn Diagram; Multiplication and Division Positive Decimals; Table; Conversion of Fraction
  Decimals Percents; Probability of Two Distinct Events.
- Sai: Venn Diagram; Conversion of Fraction Decimals Percents; Percent Of; Addition and Subtraction
  Integers; Median.

Tuy nhiên **toàn bộ phần bằng chứng của artifact này không dùng được**:

| Chỉ số | Kết quả |
|---|---|
| `evidence_ok=True` | **0/10** |
| Phản chứng lật một bước (`max_flip_abs_dp`) | **rỗng ở cả 10** |
| Ảnh hưởng bỏ đồ thị (`graph_dp`) | **rỗng ở cả 10** |
| Occlusion mạnh nhất (`max_occ_abs_dp`) | 0,3485–1,0023 nhưng **bị lệch hằng số**, không dùng được |
| Norm trạng thái tại khái niệm đích / hạng | 1,0366–1,8917 / hạng 3–17 trên 123 |

Nguyên nhân đo được: p lưu lúc chọn case (0,9532 / 0,7000 / 0,9000 / 0,4000 / 0,8000 và
0,9531 / 0,7000 / 0,9000 / 0,6000 / 0,8000) không tái lập được khi chạy lại một hàng (0,5917 /
0,4501 / 0,5243 / 0,4479 / 0,4887 / 0,5848 / 0,4642 / 0,6242 / 0,5168 / 0,5557). Đã kiểm tra
`pos → user_id` khớp `split.json` cho cả 10 case, nên lỗi không nằm ở map người học. Hai khả năng
còn lại: model dùng để quét ứng viên khác model nạp từ checkpoint seed 42, hoặc forward theo batch
khác forward một hàng. Artifact không đủ để phân biệt; cần thêm assert ngay khi tạo record.

Vì `evidence_ok=False`, `annotate_evidence` bỏ qua phản chứng và ablation đồ thị; `cases.md` chỉ
còn lịch sử cùng kỹ năng và template giải thích. Occlusion trong `cases.json` cũng không dùng
được: `delta = p_occ − rec["y_prob"]` lấy p đã lưu làm mốc trong khi `p_occ` đến từ lần chạy lại,
nên mọi Δ bị lệch một hằng số (ví dụ case Venn Diagram: bỏ t58 cho p = 0,3618, Δ lưu −0,5915,
nhưng so với baseline chạy lại 0,5917 thì Δ thật chỉ ≈ −0,230).

Ghi chú: các p trùng mốc 0,4/0,6/0,7/0,8/0,9 **không phải** số bịa — `select_stratified` lấy ứng
viên tự tin nhất trong từng ngăn `|p − 0,5|`, nên p rơi sát biên ngăn. Vấn đề là không tái lập
được, không phải làm tròn.

**Việc phải làm:** sinh lại 10 case trực tiếp từ `collect_cases` (p lấy đúng từ lượt forward đã
dùng để chọn), thêm assert `|p_recheck − p| ≤ 1e-6`, rồi mới tính phản chứng/occlusion/ablation
đồ thị. Chừng nào chưa làm, báo cáo không trích dẫn bất kỳ Δ nào từ `cases.json`.

### 4.7 Dashboard người học

`results/level1_out/` có 25 người học trong `tables/learner_view_summary.csv` và hình minh hoạ
`figures/learner_86709.png`; notebook có dropdown chọn người học. Ví dụ user 86709: 6 bước, độ
chính xác 0,833, xác suất trung bình **0,7377**, khái niệm trội 82 (Addition and Subtraction
Fractions), norm trạng thái trung bình ở khái niệm trội **2,2294** và norm trạng thái trung bình
cuối **0,9857**; bước đầu tiên dự đoán p = 0,6296. Đây là dashboard **minh hoạ**, không phải mẫu
thống kê.

---

## 5. Kết quả Cấp độ 2 — dự đoán

### 5.1 ASSISTments 2009

| Mô hình | AUC | Acc | NLL | Val AUC | n_test | Thời gian |
|---|---:|---:|---:|---:|---:|---:|
| **GRKT-train** | **0,7972** | **0,7399** | **0,5217** | 0,8035 | 47.358 | 2.838 s |
| DKT | 0,7859 | 0,7357 | 0,5278 | 0,7926 | 47.358 | không ghi |
| GRKT-zero | 0,7840 | 0,7317 | 0,5311 | 0,7924 | 47.358 | 2.583 s |
| GKT-Zero | 0,7644 | 0,7199 | 0,5475 | 0,7758 | 47.358 | không ghi |
| GKT-Transition | 0,7536 | 0,7091 | 0,5595 | 0,7626 | 47.358 | không ghi |

- GRKT-train − DKT = **+0,0113 AUC**; GRKT-train − GKT-Transition = +0,0436.
- **GRKT-zero (0,7840) thấp hơn DKT (0,7859) 0,0019 AUC.** Nói cách khác, nếu bỏ đồ thị thì
  GRKT không còn vượt DKT; toàn bộ phần vượt DKT (+0,0113) đến từ chênh lệch `train − zero`
  (+0,0132). Đây là lý do ablation `train` vs `zero` là bằng chứng quan trọng nhất của Cấp 2,
  đồng thời là cảnh báo rằng lợi ích không thể quy cho "kiến trúc GRKT nói chung".
- 3 seed GRKT: **0,7983 ± 0,0009 AUC**, 0,7424 ± 0,0023 acc, 0,5170 ± 0,0037 NLL.

### 5.2 Junyi (tập con)

| Mô hình | AUC | Acc | NLL | Val AUC | n_test | Thời gian |
|---|---:|---:|---:|---:|---:|---:|
| **GRKT-train** | **0,7896** | **0,8017** | **0,4378** | 0,7938 | 48.844 | 1.984 s |
| DKT | 0,7501 | 0,7864 | 0,4623 | 0,7520 | 48.844 | không ghi |
| GKT-Zero | 0,7225 | 0,7821 | 0,4798 | 0,7280 | 48.844 | không ghi |
| GKT-Transition | 0,7202 | 0,7800 | 0,4790 | 0,7208 | 48.844 | không ghi |

- GRKT − DKT = **+0,0395 AUC**; GRKT − GKT-Zero = +0,0671.
- Chênh lệch lớn hơn trên ASSISTments, nhưng đây là tập con và chỉ 1 seed GRKT. Không được suy
  rộng ra toàn bộ Junyi.

### 5.3 Ablation đồ thị `train` vs `zero`

| Tập | AUC train | AUC zero | ΔAUC | NLL train | NLL zero |
|---|---:|---:|---:|---:|---:|
| ASSISTments | 0,7972 | 0,7840 | **+0,0132** | 0,5217 | 0,5311 |
| Junyi | 0,7896 | chưa chạy | — | 0,4378 | — |

Đây là bằng chứng gợi ý rằng đồ thị `rel`/`pre` đóng góp **+0,0132 AUC** trên ASSISTments. Nhưng
chỉ có **một seed** và **một tập**, nên báo cáo chỉ được nói "có lợi ích đo được", không được
nói "đồ thị là nguyên nhân". Muốn kết luận nhân quả cần chạy `zero` cho 3 seed và cho Junyi.

### 5.4 Calibration

| Tập | Mô hình | Brier | ECE |
|---|---|---:|---:|
| ASSIST | GKT-Transition | 0,1893 | **0,0097** |
| ASSIST | DKT | 0,1768 | 0,0134 |
| ASSIST | GKT-Zero | 0,1848 | 0,0163 |
| ASSIST | GRKT-zero | 0,1788 | 0,0291 |
| ASSIST | **GRKT-train** | **0,1743** | **0,0379** |
| Junyi | GKT-Transition | 0,1566 | **0,0113** |
| Junyi | **GRKT-train** | **0,1404** | 0,0127 |
| Junyi | DKT | 0,1504 | 0,0171 |
| Junyi | GKT-Zero | 0,1564 | 0,0310 |

GRKT cho Brier tốt nhất ở cả hai tập, nhưng trên ASSISTments ECE của GRKT **tệ nhất** (0,0379 so
với 0,0097 của GKT-Transition). Diễn giải: GRKT dự đoán "sắc" hơn (sai số bình phương thấp) nhưng
lệch hiệu chỉnh hơn. Đây là trade-off, không phải "GRKT tốt hơn mọi mặt". Trên Junyi, GRKT vừa
Brier tốt nhất vừa ECE gần tốt nhất (0,0127 so với 0,0113).

---

## 6. Cấp độ 2 — quỹ đạo tri thức

### 6.1 Định nghĩa các tiêu chí

| Tiêu chí | Câu hỏi | Hướng tốt |
|---|---|---|
| `precedence` | Trên các bước làm đúng khái niệm `d`, mastery của khái niệm tiên quyết `c` có ≥ mastery của `d`? | Cao |
| `precedence_gap` | `precedence` trừ đi giá trị khi hoán vị đích các cạnh (giữ bậc) | **> 0** mới là bằng chứng |
| `stability_spearman` | Lật kết quả bước áp chót, hạng mastery toàn bộ khái niệm thay đổi bao nhiêu? | Gần 1 |
| `stability_changed_frac` | Tỉ lệ khái niệm có mastery thay đổi khi lật một bước | Thấp là ổn định |
| `forget_slope` | Hồi quy mastery theo Δt khi không luyện | Âm |
| `pearson_dt_dm` | Tương quan Δt với thay đổi mastery | Âm |

Ba tiêu chí `consistency_strengthen`, `consistency_spread`, `locality` luôn bằng 1 do cấu trúc
mô hình (phép biến đổi dương, chuẩn hoá hàng); chúng chỉ là **kiểm chứng cài đặt**, không phải
bằng chứng khoa học.

### 6.2 Kết quả

150 người học mỗi tập:

| Tập / đồ thị | `precedence` | `precedence_gap` | Spearman | changed_frac | `forget_slope` | `pearson_dt_dm` |
|---|---:|---:|---:|---:|---:|---:|
| ASSIST / train | 0,3418 | **−0,1108** | 0,9712 | 0,4459 | −4,63e−4 | −0,6104 |
| ASSIST / zero | 0,2015 | **−0,0776** | 0,9936 | 0,0081 | −2,38e−4 | −0,2290 |
| Junyi / train | 0,3651 | **−0,1014** | 0,9861 | 0,5120 | +1,01e−5 | −0,0002 |

Cơ chế lan truyền (`abs_delta_retrieval`): ASSIST/train 0,0335 so với 0,0 ở zero; Junyi/train
0,0209. Đây là bằng chứng định lượng rằng đồ thị **có** lan truyền thay đổi sang khái niệm khác
khi dự đoán, không chỉ là tham số trang trí.

### 6.3 Cách đọc — không được đảo dấu

1. **Nhất quán tiên quyết không đạt.** `precedence_gap` âm ở cả ba cấu hình. Nghĩa là đồ thị
   `pre_map` thật **không** giúp mastery khái niệm tiên quyết cao hơn so với một đồ thị hoán vị
   cùng bậc. Đây là kết quả phủ định và phải viết đúng. Hai nguyên nhân khả dĩ: (a) `pre_map` là
   heuristic đếm flip, không phải quan hệ tiên quyết thật; (b) mastery là trạng thái ẩn nhiều
   kênh, không được ràng buộc thứ tự.
2. **Độ ổn định cao nhưng độ nhạy cũng cao.** Spearman 0,971–0,994 cho thấy thứ hạng mastery ít
   đổi; nhưng `changed_frac` 0,446 (train) so với 0,008 (zero) cho thấy lật một bước làm thay đổi
   gần một nửa số khái niệm. Đồ thị lan truyền rộng — có thể là nhạy cảm quá mức, cần phân tích
   thêm trước khi gọi là "ổn định".
3. **Quên trên Junyi là giới hạn dữ liệu.** `forget_slope` +1,0e−5 và `pearson_dt_dm` −0,0002,
   trong khi `dt_zero_frac = 0,881`: phần lớn Δt bằng 0 do timestamp làm tròn 15 phút. Không được
   trình bày "GRKT học được quên trên Junyi". Trên ASSIST, tín hiệu âm nhẹ (−4,6e−4) và tương
   quan −0,61 là có thật nhưng nhỏ.
4. **Không dùng ba chỉ số luôn bằng 1 làm bằng chứng.** Chúng chỉ xác nhận cài đặt đúng.

### 6.4 Trực quan hoá và dashboard

- Heatmap mastery theo khái niệm × thời gian: `assist2009/trajectories/heatmap_train.png`,
  `heatmap_zero.png`, `junyi/trajectories/heatmap_train.png`; dữ liệu nền ở `mastery_*.csv`.
- Dashboard Cấp 2: 60 dòng người học–đồ thị (`dashboard/learner_dashboard_summary.csv`) = 20
  người × 3 cấu hình (assist train, assist zero, junyi train), tức **40 người duy nhất**; mỗi
  dòng có xác suất từng bước, nhãn, mastery trước/sau lan truyền, mastery láng giềng, Brier/AUC
  cá nhân và `alignment` ("khớp S−1").
- Ví dụ user 86709 (ASSIST): 6 bước, acc 0,833, mean_pred 0,851, Brier 0,098, mastery 0,777 →
  0,890, mastery láng giềng trung bình 0,502. Ví dụ user 88793: 41 bước, acc 0,171,
  mean_pred 0,363, mastery 0,522 → 0,427. Hai ví dụ cho thấy dashboard phản ánh đúng khác biệt
  cá nhân; không dùng để suy rộng.

---

## 7. Liên kết hai cấp và thảo luận

### 7.1 Cùng split, khác harness

| Kiểm tra | Kết quả |
|---|---|
| Split ASSIST C1 vs C2 | 2.330/776/778 và 142.299/47.508/48.136 — **trùng khít** |
| DKT C1 vs C2 | 0,7866 (C1 seed 42) / 0,7862 ± 0,0006 (C1 5 seed) vs 0,7859 (C2) → cùng split, cùng kiến trúc, khác lần train |
| GKT tĩnh C1 vs C2 | Transition 0,7427 (seed 42) / 0,7609 ± 0,0096 (5 seed) vs 0,7536 (C2); Zero 0,7549 / 0,7576 ± 0,0055 vs 0,7644 (C2) → **cùng kiến trúc nhưng khác harness**, chênh nằm trong/ngoài 1 std |
| GKT-MHA | chỉ có ở C1 (0,7782 seed 42 / 0,7668 ± 0,0095); C2 không chạy MHA |
| Cấu hình ẩn | C1: 32; C2 baseline: 32; C2 GRKT: 128 → **bất đối xứng dung lượng** |

Hệ quả của bất đối xứng: GRKT có nhiều tham số hơn baseline (128 so với 32), nên chênh lệch AUC
không thể quy hoàn toàn cho kiến trúc hay đồ thị. Ablation `train` vs `zero` giữ nguyên dung
lượng GRKT ở cả hai nhánh, nên **so sánh trong ablation là công bằng hơn** so sánh GRKT với
baseline; đó là lý do báo cáo dùng ablation làm bằng chứng chính cho vai trò đồ thị.

Vì vậy báo cáo dùng hai loại so sánh:

1. **Trong cùng cấp:** Cấp 1 so DKT với 6 biến thể GKT (5 seed + paired test); Cấp 2 so GRKT với
   DKT và GKT tĩnh (1 seed, so điểm).
2. **Giữa hai cấp:** chỉ so **DKT** trên cùng split (0,7866/0,7862 vs 0,7859) như kiểm tra tính
   nhất quán của pipeline. Không so GKT-MHA với GRKT như thể cùng một baseline.

Bảng xếp hạng Cấp 1 trên test split ASSISTments, trung bình 5 seed (chỉ để định vị; mọi khoảng
cách giữa các biến thể GKT đều **không** có ý nghĩa thống kê, xem §4.2):

| Hạng | Mô hình | AUC (mean ± std) |
|---|---|---:|
| 1 | DKT | **0,7862 ± 0,0006** |
| 2 | GKT-MHA | 0,7668 ± 0,0095 |
| 3 | GKT-Transition | 0,7609 ± 0,0096 |
| 4 | GKT-Shuffled | 0,7579 ± 0,0039 |
| 5 | GKT-Identity | 0,7578 ± 0,0056 |
| 6 | GKT-Zero | 0,7576 ± 0,0055 |
| 7 | GKT-ShuffledRow | 0,7576 ± 0,0127 |

Bảng Cấp 2 (seed 42, harness riêng, có GRKT):

| Hạng | Mô hình | AUC |
|---|---|---:|
| 1 | GRKT-train | 0,7972 |
| 2 | DKT | 0,7859 |
| 3 | GRKT-zero | 0,7840 |
| 4 | GKT-Zero | 0,7644 |
| 5 | GKT-Transition | 0,7536 |

Hai bảng **không** được gộp thành một bảng xếp hạng duy nhất: khác harness, khác số seed, và các
baseline GKT đã được train lại ở Cấp 2.

### 7.2 Những gì đã chứng minh được

- Pipeline chia dữ liệu, chống rò rỉ và đánh giá **nhất quán** giữa hai cấp: DKT lệch 0,0007 AUC
  giữa hai lần chạy độc lập trên cùng split, và C1 qua 5 seed chỉ dao động ±0,0006.
- **DKT vượt mọi biến thể GKT ở Cấp 1** với kiểm định paired trên 5 seed: ΔAUC +0,0194 (so MHA)
  đến +0,0286 (so Zero), t = 4,17–13,91, p = 0,0002–0,014. Đây là kết quả **phủ định** cho GKT
  trong harness này và phải được viết đúng như vậy.
- **Ablation đồ thị lúc train đã đủ và cho kết quả null:** Transition − Zero = +0,0033
  (p = 0,546), Transition − Shuffled = +0,0030 (p = 0,495), ShuffledRow − Zero = −0,00001
  (p = 0,998), Identity − Zero = +0,0002 (p = 0,305). Cấu trúc đồ thị **không** tạo khác biệt đo
  được ở Cấp 1.
- **Ablation lúc suy luận chứng minh mô hình *có đọc* `A`:** đặt `A := 0` lúc eval làm AUC giảm
  0,0109 (Transition và ShuffledRow), 0,0031 (Shuffled), 0,0029 (MHA), và đúng 0 với Identity/Zero
  (hai cấu hình này không có cạnh để bỏ). Độ lớn này đo **mức phụ thuộc**, không đo **chất lượng**
  cạnh — Transition và ShuffledRow (đồ thị ngẫu nhiên theo hàng) giảm gần bằng nhau.
- **GRKT vượt DKT trên cả hai tập** ở seed 42 (+0,0113 và +0,0395 AUC), và đạt
  0,7983 ± 0,0009 qua 3 seed trên ASSISTments.
- **Bỏ đồ thị làm giảm 0,0132 AUC** trên ASSISTments (một seed) và làm `abs_delta_retrieval` về 0
  → với GRKT, đồ thị có tác dụng cơ chế, không chỉ là tham số.
- **Calibration của GRKT tốt nhất theo Brier** ở cả hai tập, nhưng không tốt nhất theo ECE trên
  ASSISTments.
- Quỹ đạo mastery **ổn định về thứ hạng** (Spearman > 0,97) và **có lan truyền** sang khái niệm
  lân cận.

### 7.3 Những gì chưa chứng minh được

- **Vai trò cấu trúc đồ thị ở Cấp 1.** Đã đo, và kết quả là **null**: không biến thể nào khác
  Zero có ý nghĩa. Nguyên nhân khả dĩ: GKT tĩnh chỉ lan truyền một bước (`k_hop=1`), `A` được
  chuẩn hoá hàng nên thông tin tự thân bị ghi đè trong `_agg_neighbors`, và đồ thị Transition dày
  5,9 % khiến Shuffled gần như tương đương.
- **Lợi ích đồ thị ở Cấp 2 là nhân quả.** Chỉ 1 seed, 1 tập; chưa có kiểm định.
- **Nhất quán tiên quyết.** `precedence_gap` âm ở cả hai tập → không đạt.
- **Quên trên Junyi.** Tín hiệu gần 0 và bị chi phối bởi Δt = 0.
- **So sánh GRKT với baseline theo seed.** Baseline chỉ 1 seed, không có paired test.
- **Tổng quát hoá ra ngoài tập con Junyi.** 4,1 % người học, 1,55 % lượt, chọn khái niệm
  transductive.
- **Case study Cấp 1.** 0/10 case có `evidence_ok=True`; không trích dẫn Δ nào từ `cases.json`.
- **Calibration của GKT-MHA.** Bảng chính 0,7782 nhưng file calibration cùng checkpoint cho
  0,7454; chưa xác định được nguyên nhân từ artifact.

### 7.4 Liên hệ với bài báo

GRKT được tái lập theo mô tả KDD 2024 với hai quan hệ `rel`/`pre` và bốn pha. Kết quả định tính
khớp bài báo ở chỗ GRKT vượt DKT và GKT trên dữ liệu có đồ thị khái niệm; nhưng báo cáo này
**không** tái lập đúng con số của bài báo vì: (a) dùng tập con Junyi thay vì toàn bộ; (b) chia
theo người học 0,6/0,2/0,2 thay vì giao thức gốc; (c) giới hạn 20 epoch và early stopping;
(d) đồ thị `pre` là heuristic đếm flip. Vì vậy chỉ nên so **xu hướng**, không so số tuyệt đối.

---

## 8. Hạn chế

1. **Junyi là tập con 3.000 người** (4,1 %) và 200 khái niệm chọn transductive; không đại diện
   toàn bộ Junyi.
2. **Đa seed không đều:** Cấp 1 có 5 seed cho **cả 7 mô hình** (35 run); Cấp 2 chỉ GRKT trên
   ASSISTments có 3 seed, baseline và Junyi chỉ 1 seed.
3. **Bất đối xứng dung lượng:** GRKT dùng 128 ẩn, baseline Cấp 2 dùng 32. Chênh lệch GRKT–baseline
   không chỉ do kiến trúc/đồ thị; chỉ ablation `train` vs `zero` là so sánh công bằng.
4. **Ablation đồ thị Cấp 1 đã đủ nhưng kết quả null**; độ phủ 99,5 % chỉ nói về đồ thị
   Transition, không nói mô hình có dùng nó.
5. **Case study Cấp 1 không dùng được:** 0/10 case có `evidence_ok=True`, p chọn case không khớp
   p lúc chạy lại (ví dụ 0,9532 → 0,5917), cột phản chứng/đồ thị trống; occlusion bị lệch hằng số
   nên Δ trong `cases.json` không phải Δ thật.
6. **Calibration Cấp 1 không khớp bảng chính với GKT-MHA** (0,7454 so với 0,7782). Nguyên nhân
   chưa xác định được từ artifact (file `.pt` là `state_dict` trần).
7. **`seconds` của baseline Cấp 2 = 0** dù đã train 11–20 epoch; không thể báo cáo chi phí đầy đủ.
8. **`precedence_gap` âm** → tiêu chí tiên quyết không đạt; `changed_frac` cao gợi ý nhạy cảm
   quá mức.
9. **Quên trên Junyi bị giới hạn dữ liệu** (88,1 % Δt = 0; trần 240 phút).
10. **Không có kiểm định thống kê cho GRKT vs baseline**; chênh lệch AUC chỉ là so điểm.
11. **Hai định nghĩa NLL:** bảng chính tính trung bình loss theo batch, calibration/ablation tính
    BCE toàn cục; lệch hệ thống ~0,002 nên không được so chéo hai bảng.
12. **`GKT-Identity` và `GKT-Zero` cho kết quả trùng khít** (0,754883 cả hai ở seed 42): trong
    `_agg_neighbors` vị trí tự thân bị ghi đè bởi `self_features`, nên ma trận đơn vị tương đương
    không cạnh. Đây là thuộc tính cài đặt, không phải hai thí nghiệm độc lập.
13. **Baseline GKT được train lại ở Cấp 2** nên số không trùng Cấp 1 (Transition 0,7536 vs
    0,7609 ± 0,0096; Zero 0,7644 vs 0,7576 ± 0,0055). Không được trộn số giữa hai cấp.
14. **3/35 run Cấp 1 được resume từ checkpoint** (Identity seed 1011, Zero seed 456, MHA seed
    789); fingerprint khớp nên hợp lệ, nhưng là nguồn khác biệt tiềm ẩn.
15. **Không có bảng phân công, slide và video demo** trong repo tại thời điểm tổng hợp.

---

## 9. Đối chiếu `YEU_CAU.md`

### 9.1 Cấp độ 1 (§5)

| Yêu cầu | Bằng chứng | Trạng thái |
|---|---|---|
| ≥ 1 tập công khai | ASSISTments 2009 đầy đủ, hash `f4b8791a…af3374`, 525.534 dòng | Đạt |
| Baseline không đồ thị | DKT 39.899 tham số, 0,7862 ± 0,0006 | Đạt |
| Mô hình GCN/GAT/GraphSAGE **hoặc GKT** | 6 biến thể GKT (Transition, Identity, Zero, Shuffled, ShuffledRow + MHA); `Dense` có đồ thị nhưng không train | Đạt hình thức; cần giải thích MHA thay `A` tĩnh bằng attention |
| Ablation bỏ cạnh / ma trận đơn vị | Lúc train: 5 biến thể × 5 seed; lúc suy luận: 6 mô hình. Kết quả **null** ở train, có phụ thuộc ở suy luận | **Đạt về thực hiện**; kết luận vai trò cấu trúc là phủ định |
| 5 đúng + 5 sai kèm bằng chứng | 10 case, 5 khái niệm khác nhau mỗi bucket | **Chưa đạt về bằng chứng** (0/10 `evidence_ok`, occlusion lệch) |
| §6 chống rò rỉ | split rời, đồ thị train-only, nhãn dịch 1 bước, `leak_report.txt` (chỉ 0,74 % khối lượng ở val+test) | Đạt |
| §7 AUC/ACC/NLL | `tables/level1_metrics.csv` | Đạt |
| §7 ≥ 3 seed mean ± std | 5 seed × 7 mô hình + paired t-test + bootstrap CI + dz | Đạt |
| §7 vai trò đồ thị | Ablation train null (p 0,31–1,00); ablation suy luận −0,0109 AUC cho Transition/ShuffledRow | **Đo đủ**; không chứng minh được lợi ích cấu trúc |
| §7 calibration | Brier/ECE/ACE + reliability 7 mô hình; MHA lệch bảng chính | Đạt nhưng trừ MHA |
| §8 dashboard | 25 người học + hình + dropdown | Đạt |
| §8 báo cáo/slide/bảng phân công | Báo cáo này; slide + bảng phân công chưa có | Một phần |

### 9.2 Cấp độ 2 (§5, §7)

| Yêu cầu | Bằng chứng | Trạng thái |
|---|---|---|
| Phương pháp graph-aware ≥ 2024 | GRKT (KDD 2024), checkpoint + `grkt_api.py` | Đạt |
| ≥ 2 tập công khai, cùng quy trình/chỉ số | ASSISTments + Junyi (tập con), cùng split seed 42, cùng AUC/ACC/NLL | Đạt (Junyi là tập con) |
| Trực quan mastery theo khái niệm × thời gian | heatmap + `mastery_*.csv` + dashboard 60 dòng (40 người) | Đạt |
| ≥ 2 tiêu chí quỹ đạo | 6 tiêu chí thật; 3 kiểm chứng cài đặt tách riêng | Đạt về đo lường; tiên quyết không đạt |
| §6.2 đồ thị từ train | `graphs_train_only.npz` + `leak_report.txt` | Đạt |
| §6.4 split rời theo người | giao = 0 ở cả hai tập | Đạt |
| §7 AUC/ACC/NLL | `level2_all_datasets.csv` | Đạt |
| §7 ≥ 3 seed | GRKT ASSIST 3 seed; Junyi/baseline 1 seed | Một phần |
| §7 ablation đồ thị | `train` vs `zero` trên ASSIST | Đạt một phần (1 seed) |
| §7 phân tích nâng cao | calibration Brier/ECE/ACE + reliability | Đạt |
| §8 dashboard | 60 dòng (20 người × 3 cấu hình) + JSON + CSV từng bước | Đạt |

### 9.3 §8 — sản phẩm nộp

| Sản phẩm | Trạng thái |
|---|---|
| Mã nguồn + requirements + hướng dẫn chạy | Có (`level1/`, `level2/`, `GKT/`, `GRKT/`, `requirements.txt`) |
| Split cố định / mã tạo split | Có (`split.json` cả hai cấp; đã đối chiếu trùng khít từng phần tử; file track bằng LFS) |
| Báo cáo 12–18 trang | **Báo cáo này** |
| Slide + bảng phân công | **Chưa có** — cần nhóm bổ sung |
| Notebook/dashboard chọn người học | Có (C1 và C2) |
| 5 đúng + 5 sai | Có (C1); C2 có dashboard từng bước nhưng chưa có 5+5 riêng |
| Demo + video dự phòng | **Chưa có** |

---

## 10. Tái lập

### 10.1 Cấu hình đã khoá

Cấp 1 — các khoá chính rút gọn từ `results/level1_out/run_config.json` (file gốc bọc trong
`"locked"`, kèm `data_sha256` và `graph_types`; ngoài ra còn `time_budget_min`):

```json
{
  "model_names": ["DKT", "GKT-Transition", "GKT-Identity", "GKT-Zero",
                  "GKT-Shuffled", "GKT-MHA", "GKT-ShuffledRow"],
  "epochs": 20, "batch_size": 32, "max_seq_len": 200,
  "emb_dim": 32, "hid_dim": 32, "lr": 0.001, "patience": 5,
  "seeds": [42, 123, 456, 789, 1011], "split_seed": 42
}
```

`graph_types` trong file gốc chỉ liệt kê `Transition, Dense, Identity, Zero, Shuffled`;
`GKT-ShuffledRow` được thêm vào `model_names` sau nên không xuất hiện trong `graph_types` — không
ảnh hưởng lần chạy (bảng `graph_stats.csv` có đủ 6 đồ thị).

Cấp 2 — các khoá chính rút gọn từ `results/level2_moderate/run_config.json` (file gốc bọc trong
`"locked"`, ngoài ra còn `dataset_caps` và `time_budget_min`):

```json
{
  "datasets": ["assist2009", "junyi"],
  "variants_by_dataset": {"assist2009": ["train", "zero"], "junyi": ["train"]},
  "epochs": 20, "baseline_epochs": 20, "batch_size": 32, "max_seq_len": 200,
  "d_hidden": 128, "k_hidden": 16, "k_hop": 1,
  "graph_thresh": 0.6, "alpha": 0.01, "lr": 0.001, "patience": 5,
  "seeds": [42, 123, 456], "split_seed": 42,
  "max_users": 3000, "multi_seed_datasets": ["assist2009"]
}
```

### 10.2 Cách chạy lại

1. Mở `kaggle_level1_full.ipynb` trên Kaggle (T4), attach `skill_builder_data.csv`, chạy toàn bộ.
   Có thể đặt `EVAL_ONLY=True` để chỉ đánh giá lại từ checkpoint đã có.
2. Mở `kaggle_level2_full.ipynb`, attach dữ liệu Junyi thô và output Cấp 1; chạy với
   `EVAL_ONLY=False` cho lần đầu, các lần sau đặt `EVAL_ONLY=True` và attach
   `level2_moderate_output.zip`.
3. Cấu hình bị khoá bởi `run_config.json`; đổi cấu hình giữa các phiên sẽ bị từ chối. Không xoá
   `ckpt_*.pt` nếu muốn resume.

### 10.3 Thời gian đo được

- Cấp 1: 35 run qua 7 phiên, tổng **87.894,8 s ≈ 24,42 h** (ngân sách 540 phút/phiên). Theo mô
  hình, mỗi mô hình 5 seed:

| Mô hình | min (s) | max (s) | tổng (s) | resume |
|---|---:|---:|---:|---|
| DKT | 31,1 | 37,0 | 166,6 | — |
| GKT-Transition | 1.379,0 | 3.588,7 | 13.355,5 | — |
| GKT-Identity | 2.279,4 | 3.406,8 | 14.749,7 | seed 1011 |
| GKT-Zero | 2.261,1 | 3.742,6 | 15.764,7 | seed 456 |
| GKT-Shuffled | 1.892,1 | 3.707,5 | 14.692,2 | — |
| GKT-MHA | 2.009,2 | 4.344,9 | 15.208,9 | seed 789 |
| GKT-ShuffledRow | 1.521,9 | 3.789,0 | 13.957,2 | — |

- Cấp 2: một phiên 378 phút (09:48–16:06 UTC), ngân sách 540 phút → còn 162 phút.
  GRKT ASSIST/train 2.838 s (12 epoch, ~4,0 phút/epoch); GRKT ASSIST/zero 2.583 s (11 epoch);
  GRKT Junyi 1.984 s (11 epoch, ~3,0 phút/epoch). Tổng GRKT ≈ 2,06 giờ.
- Baseline Cấp 2 đã train 11–20 epoch nhưng trường `seconds` không được ghi (0,0).

---

## 11. Kết luận và việc còn lại

### 11.1 Kết luận

1. **Pipeline đúng và tái lập được.** Cùng split, cùng seed, cùng chỉ số; DKT giữa hai lần chạy
   lệch 0,0007 AUC và qua 5 seed Cấp 1 chỉ dao động ±0,0006. Chống rò rỉ được kiểm tra bằng số
   (giao = 0; đồ thị chỉ từ train; nhãn dịch 1 bước; 0,74 % khối lượng đồ thị chỉ xuất hiện ở
   val+test).
2. **GRKT là mô hình tốt nhất trong các mô hình đã chạy ở Cấp 2** trên cả hai tập: 0,7972 AUC
   (ASSISTments) và 0,7896 (Junyi tập con), vượt DKT +0,0113 và +0,0395. Trên ASSISTments, GRKT
   đạt 0,7983 ± 0,0009 qua 3 seed.
3. **Với GRKT, đồ thị có tác dụng cơ chế nhưng bằng chứng nhân quả còn yếu.** Bỏ cạnh làm giảm
   0,0132 AUC và đưa `abs_delta_retrieval` về 0, nhưng chỉ 1 seed/1 tập.
4. **Cấp 1 đo đủ và cho kết quả phủ định về vai trò cấu trúc đồ thị.** DKT vượt *mọi* biến thể GKT
   với ý nghĩa thống kê (ΔAUC +0,0194…+0,0286, p ≤ 0,014). Ablation lúc train null
   (p 0,31–1,00); ablation lúc suy luận cho thấy mô hình **có đọc** `A` (−0,0109 AUC khi zero hoá
   với Transition/ShuffledRow) nhưng độ lớn phản ánh mức phụ thuộc, không phản ánh chất lượng cạnh
   — Transition và ShuffledRow giảm gần bằng nhau.
5. **Case study Cấp 1 chưa dùng được làm bằng chứng:** 0/10 case tái lập được, occlusion bị lệch
   hằng số; báo cáo không trích Δ nào từ `cases.json`.
6. **Quỹ đạo:** ổn định thứ hạng tốt (Spearman > 0,97), có lan truyền, nhưng nhất quán tiên quyết
   **không đạt** (gap âm) và tín hiệu quên trên Junyi là giới hạn dữ liệu.
7. **Junyi chỉ là tập con**, mọi kết luận chỉ có giá trị trên 3.000 người được lấy mẫu.

### 11.2 Việc còn lại (ưu tiên)

| # | Việc | Lý do | Chi phí ước tính |
|---|---|---|---|
| 0 | Sinh lại 10 case từ đúng lượt forward đã dùng để chọn (assert `\|p_recheck−p\| ≤ 1e-6`), rồi mới tính phản chứng/occlusion/ablation | Bằng chứng case hiện 0/10 hợp lệ; đây là mục §7 nhiệm vụ 3 | ~15–30 phút eval |
| 1 | Xác minh checkpoint `GKT-MHA` seed 42 (lưu kèm `epoch`/best-val trong payload) rồi tính lại calibration | Bảng chính 0,7782 vs calibration 0,7454 | ~15 phút eval |
| 2 | Chạy GRKT `zero` cho 3 seed (và Junyi nếu còn ngân sách) | Biến ΔAUC +0,0132 thành bằng chứng qua seed | ~3–6 giờ GPU |
| 3 | Chạy baseline Cấp 2 3 seed trên ASSISTments để có paired test GRKT–DKT | Hiện chỉ so điểm | ~1–3 giờ GPU |
| 4 | Điều tra `precedence_gap` âm (định nghĩa `pre_map`, ràng buộc thứ tự) | Tiêu chí tiên quyết đang không đạt | phân tích, không cần train |
| 5 | Bổ sung 5 đúng + 5 sai riêng cho Cấp 2 (GRKT) | §7 nhiệm vụ 3; C1 chưa dùng được | ~30–60 phút |
| 6 | Viết slide 20 phút + bảng phân công + video demo ≤ 5 phút | Yêu cầu §8 | không tính |

Không còn việc train thêm cho ablation Cấp 1: 5 biến thể × 5 seed đã chạy xong trong artifact
hiện tại.

---

## Phụ lục A — Bản đồ artifact

**Cấp 1 (`results/level1_out/`)**

| Tệp | Nội dung |
|---|---|
| `run_config.json`, `run_manifest.json`, `run_sessions.json` | cấu hình khoá, môi trường, 7 phiên / 35 run / 3 resume |
| `tables/dataset_stats.csv`, `tables/split_summary.csv`, `split.json` | 525.534 dòng, 4.217 người, 123 skill; split 2.330/776/778 |
| `tables/level1_metrics.csv` | bảng chính seed 42, 7 mô hình |
| `tables/runs_all_seeds.csv` | 35 run, 5 seed × 7 mô hình |
| `tables/multiseed_metrics.csv` | mean ± std 5 seed |
| `tables/paired_tests.csv` | paired t-test, bootstrap CI, dz (DKT vs 6 biến thể + giữa các biến thể) |
| `tables/graph_stats.csv`, `graph_degree_by_skill.csv`, `graph_coverage.csv` | thống kê 6 đồ thị + độ phủ |
| `tables/graph_inference_ablation.csv` | **đã có số** — ΔAUC/ΔNLL/mean\|Δp\| khi zero hoá `A` lúc eval |
| `tables/calibration.csv`, `calibration_bins.csv` | Brier/ECE/ACE + ngăn, 7 mô hình |
| `tables/case_evidence_summary.csv`, `cases.md`, `cases.json` | 10 case; **0/10 `evidence_ok`**, không dùng làm bằng chứng |
| `tables/learner_view_summary.csv`, `learner_view.json` | dashboard 25 người |
| `leak_report.txt` | kiểm tra rò rỉ (884 cạnh train, 0,74 % khối lượng val+test-only) |
| `checkpoints/` | 35 file `.pt` (7 mô hình × 5 seed; `state_dict` trần, không có metadata epoch) |
| `figures/*.png` | EDA, learning curves, so sánh, reliability, ablation, case, learner |

**Cấp 2 (`results/level2_moderate/`)**

| Tệp | Nội dung |
|---|---|
| `run_config.json`, `run_provenance.json`, `run_sessions.json` | cấu hình, lịch sử train, phiên |
| `level2_all_datasets.csv` | bảng gộp 9 dòng kết quả |
| `assist2009/level2_metrics.csv`, `junyi/level2_metrics.csv` | bảng theo tập |
| `assist2009/multiseed_metrics.csv` | GRKT 3 seed |
| `assist2009/trajectory_metrics.csv`, `junyi/trajectory_metrics.csv` | 6 tiêu chí quỹ đạo |
| `assist2009/calibration.csv`, `junyi/calibration.csv` | Brier/ECE/ACE |
| `assist2009/trajectories/`, `junyi/trajectories/` | heatmap, mastery CSV, traj NPZ |
| `dashboard/learner_dashboard_summary.csv`, `learner_dashboard_steps.csv`, `learner_dashboard.json` | dashboard 60 dòng người–đồ thị (40 người) + từng bước |
| `diagnostics/level1_linkage.json`, `legacy_ckpt_eval.csv` | liên kết C1, checkpoint cũ |
| `tables/checkpoint_status.csv` | fingerprint 5 checkpoint |
| `*/leak_report.txt` | kiểm tra rò rỉ từng tập |
| `*/graphs_train_only.npz`, `*/split.json` | đồ thị train-only, split |
| `figures/*.png` | EDA, learning curves, so sánh, dashboard |

## Phụ lục B — Thuật ngữ

| Thuật ngữ | Nghĩa trong báo cáo |
|---|---|
| KC / khái niệm | Knowledge component — đơn vị kiến thức gắn với câu hỏi |
| DKT | Deep Knowledge Tracing (LSTM, không đồ thị) |
| GKT | Graph Knowledge Tracing (trạng thái theo khái niệm + lan truyền đồ thị) |
| GKT-MHA | Biến thể GKT thay đồ thị tĩnh bằng multi-head attention |
| GRKT | Graph-based Knowledge Tracing with Retrieval, Strengthening, Learning–Forgetting (KDD 2024) |
| `rel_map` / `pre_map` | Đồ thị đồng xuất hiện / tiên quyết, đếm từ train |
| Ablation `zero` | Bỏ toàn bộ cạnh, giữ nguyên phần còn lại của mô hình |
| AUC | Diện tích dưới đường ROC, đo khả năng xếp hạng |
| NLL / BCE | Negative log-likelihood / binary cross-entropy |
| Brier | Sai số bình phương trung bình của xác suất |
| ECE / ACE | Expected / Adaptive Calibration Error |
| `precedence_gap` | Chênh lệch nhất quán tiên quyết so với đồ thị hoán vị |
| Transductive | Bước chọn khái niệm dùng thống kê toàn bộ log trước khi chia |

## Phụ lục C — Bảng phân công (mẫu, nhóm điền)

| Thành viên | MSSV | Phụ trách | Bằng chứng đóng góp |
|---|---|---|---|
| (điền) | | Dữ liệu, tiền xử lý, đồ thị | commit / notebook |
| (điền) | | Cấp 1: DKT, 6 biến thể GKT, ablation, case study | commit / notebook |
| (điền) | | Cấp 2: GRKT, quỹ đạo, calibration | commit / notebook |
| (điền) | | Báo cáo, slide, dashboard, demo | commit / slide |

> Ghi chú: bảng này để trống có chủ ý — báo cáo không tự gán tên hay tỉ lệ đóng góp khi chưa có
> xác nhận của nhóm.

---

*Hết báo cáo. Mọi số liệu có thể đối chiếu trực tiếp trong `results/level1_out/` và
`results/level2_moderate/`; các phát hiện và việc còn lại được theo dõi chi tiết trong
`GAP_LEVEL1.md` §6 và `GAP_LEVEL2.md` §10.*
