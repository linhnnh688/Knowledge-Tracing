# Ôn tập bảo vệ — dữ liệu thật, đồ thị, kết quả, rò rỉ, điểm yếu

> Bộ tài liệu gồm 3 file, nên đọc theo thứ tự:
>
> 1. **File này**: theo dõi một người học thật qua toàn bộ pipeline, đọc từng con số trong
>    `results/`, chứng minh không rò rỉ, và các điểm yếu thầy có thể chỉ ra.
> 2. [BAO_CAO_KIEN_TRUC.md](BAO_CAO_KIEN_TRUC.md): kiến trúc, công thức, khác bản gốc thế nào.
> 3. [HUONG_DAN_DOC_CODE.md](HUONG_DAN_DOC_CODE.md): từng hàm làm gì, ứng với phương trình nào.
>
> Mọi số trong file này đều đo lại từ `GKT/data/skill_builder_data.csv` và `results/` của
> commit `868838c` (branch `fix/assist2009-leak-batchnorm`). Kết quả trong `results/` đã
> **chạy lại trên dữ liệu sạch** (Cấp 1: commit `edef1b3`, Cấp 2: commit `8a875ea`).

---

## Mục lục

1. [Kịch bản trình bày 3 phút](#1-kịch-bản-trình-bày-3-phút)
2. [Một người học thật đi qua pipeline](#2-một-người-học-thật-đi-qua-pipeline)
3. [Đồ thị bằng ví dụ thật](#3-đồ-thị-bằng-ví-dụ-thật)
4. [Rò rỉ dữ liệu: 5 loại, bằng chứng, chỗ trong code](#4-rò-rỉ-dữ-liệu)
5. [Kết quả Cấp 1 và cách đọc](#5-kết-quả-cấp-1-và-cách-đọc)
6. [Kết quả Cấp 2 và cách đọc](#6-kết-quả-cấp-2-và-cách-đọc)
7. [Điểm yếu đã biết trong `results/` (đừng để bị bất ngờ)](#7-điểm-yếu-đã-biết-trong-results)
8. [Thầy chỉ vào X thì trả lời gì](#8-thầy-chỉ-vào-x-thì-trả-lời-gì)

---

## 1. Kịch bản trình bày 3 phút

1. **Bài toán:** cho lịch sử (khái niệm, đúng/sai) của một học sinh, dự đoán xác suất làm
   đúng câu kế tiếp. Dữ liệu chính ASSISTments 2009; Cấp 2 thêm tập con Junyi.
2. **Cấp 1:** DKT (LSTM, không đồ thị) và GKT (mỗi khái niệm một trạng thái, lan truyền 1-hop
   trên đồ thị khái niệm). Ablation: Transition (đồ thị thật) / Zero (không cạnh) / Shuffled
   (xáo cạnh). Kết quả: **DKT thắng GKT**; Transition > Zero có ý nghĩa, nhưng **Transition ≈
   Shuffled** → chưa chứng minh được *cấu trúc* cạnh có ích.
3. **Cấp 2:** GRKT (KDD 2024) với 3 đồ thị rel/pre/sub, dự đoán kiểu IRT, học/quên theo thời
   gian. **GRKT-train vượt DKT** (0,7691 vs 0,7608) và **GRKT-train vượt GRKT-zero +0,0143 AUC
   trên 3 seed (p = 0,0009)**. So sánh train vs zero mới cô lập được vai trò đồ thị.
4. **Chống rò rỉ:** chia theo người học, đồ thị chỉ từ train, nhãn dịch 1 bước, và **sửa lỗi
   thứ tự/lượt trùng của file ASSIST2009** (trước khi sửa: 70,6% bước dự đoán có tương lai
   trong lịch sử). Leak report có số để kiểm.
5. **Hạn chế thật:** mất 11 khái niệm do dedup, Junyi chỉ là tập con 1 seed, tiêu chí precedence
   chưa đạt trên ASSIST2009, pha học/quên của GRKT gần như không hoạt động trên ASSIST2009.

---

## 2. Một người học thật đi qua pipeline

Người học `user_id = 64525` (thuộc **tập val**). Trong file có 561 lượt sau làm sạch → chỉ
200 lượt đầu được dùng.

### 2.1. File gốc: xếp theo kỹ năng, không theo thời gian

8 dòng đầu của người này **theo thứ tự trong file**:

| order_id | skill_id | skill_name | correct |
|---|---|---|---|
| 33022537 | 1 | Box and Whisker | 1 |
| 33022709 | 1 | Box and Whisker | 1 |
| **28186893** | 2 | Circle Graph | 1 |
| 28187093 | 2 | Circle Graph | 1 |
| 32413158 | 2 | Circle Graph | 1 |
| … | | | |

`order_id` nhảy từ 33.022.709 **xuống** 28.186.893: file gom theo `skill_id` trước. Nếu
`groupby('user_id')` giữ thứ tự file (như mã gốc), lượt 33.022.537 (làm sau) nằm *trước* lượt
28.186.893 (làm trước) trong lịch sử → mô hình thấy tương lai. 2.893 / 4.163 người học có lỗi
này.

### 2.2. Một lượt làm bài gắn nhiều kỹ năng = nhiều dòng

`user_id = 14`, `order_id = 21617623` (cùng một lần làm câu `problem_id = 93383`):

| order_id | problem_id | correct | ms_first_response | skill_id | skill_name |
|---|---|---|---|---|---|
| 21617623 | 93383 | 0 | 26271 | 2 | Circle Graph |
| 21617623 | 93383 | 0 | 26271 | 37 | *(trống)* |
| 21617623 | 93383 | 0 | 26271 | 70 | Percent Of |

Ba dòng, **cùng một nhãn**. Không gộp thì bước 2 có nhãn = nhãn bước 1 (bản sao) → mô hình
"đoán" đúng chỉ bằng cách chép nhãn bước trước.

### 2.3. `prepare_interactions` ([GKT/processing.py](../GKT/processing.py))

| Bước | Số dòng |
|---|---|
| File gốc | 525.534 (4.217 người, 123 kỹ năng) |
| Bỏ dòng không có `skill_id` | 459.208 |
| Sắp theo `order_id` trong từng người, giữ 1 dòng mỗi `(user_id, order_id)` | bỏ **176.103** dòng = **121.207** dòng lặp y hệt (cùng người, lượt, kỹ năng) + **54.896** dòng kỹ năng phụ |
| Bỏ người < 3 lượt | **282.619** dòng, **3.852** người, **112** kỹ năng |

11 kỹ năng biến mất vì chỉ xuất hiện như kỹ năng phụ: Mode, Ordering Real Numbers, Fraction
Of, Multiplication Whole Numbers, Intercept, Linear Equations, Slope, Computation with Real
Numbers, Solving Inequalities và 2 kỹ năng không có tên (id 37, 102).

Người 64525 sau khi làm sạch (8 lượt đầu, đã đúng thứ tự thời gian):

| order_id | skill_id | skill (0..111) | correct | x = 2·skill + correct | problem_id | ms_first_response |
|---|---|---|---|---|---|---|
| 21441630 | 278 | 71 | 0 | 142 | 54003 | 20797 |
| 21441970 | 278 | 71 | 1 | 143 | 53991 | 13797 |
| 21442097 | 278 | 71 | 0 | 142 | 54071 | 14172 |
| 21442513 | 278 | 71 | 1 | 143 | 54015 | 48813 |
| 21442851 | 278 | 71 | 1 | 143 | 53987 | 22187 |
| 21443065 | 280 | 73 | 1 | 147 | 53781 | 27672 |
| 21443359 | 279 | 72 | 1 | 145 | 54193 | 5390 |
| 21443474 | 280 | 73 | 1 | 147 | 53795 | 13937 |

- skill 71 = *Addition and Subtraction Positive Decimals*, 72 = *Multiplication and Division
  Integers*, 73 = *Addition and Subtraction Fractions*.
- `skill = factorize(skill_id, sort=True)`: id gốc 278 → chỉ số 71.
- `x = 2·skill + correct`: làm **sai** skill 71 → token 142, làm **đúng** → 143. Đầu vào
  one-hot 2C = 224 chiều.

### 2.4. Thành tensor cho DKT/GKT

`KTDataset` lưu cho mỗi người học 3 list: `features` (x), `questions` (skill), `answers`
(correct). `pad_collate` đệm `-1` tới độ dài chuỗi dài nhất trong batch:

```
features  = [142, 143, 142, 143, 143, 147, 145, 147, ...,  -1, -1]
questions = [ 71,  71,  71,  71,  71,  73,  72,  73, ...,  -1, -1]
answers   = [  0,   1,   0,   1,   1,   1,   1,   1, ...,  -1, -1]
```

Mô hình trả `pred[B, T-1]`. **`pred[t]` là xác suất đúng của lượt t+1**, tính từ trạng thái
sau khi đã thấy lượt 0..t và *danh tính* khái niệm `questions[t+1]`:

| t | đầu vào đã thấy | pred[t] dự đoán | so với nhãn |
|---|---|---|---|
| 0 | (71, sai) | P(đúng skill 71 ở lượt 1) | answers[1] = 1 |
| 4 | …, (71, đúng) | P(đúng skill 73 ở lượt 5) | answers[5] = 1 |

`KTLoss` so `pred` với `answers[:, 1:]` và bỏ vị trí `-1`. Lượt 0 không bao giờ là mục tiêu
(chưa có lịch sử).

### 2.5. Thành `UserSequence` cho GRKT ([level2/grkt_data.py](../level2/grkt_data.py))

GRKT dùng chỉ số **1-based** (0 = padding) và thêm câu hỏi + thời gian:

```
probs  = [1621, 1618, 1638, 1624, 1617, 1495, 1716, 1509]   # factorize(problem_id) + 1
knows  = [[72], [72], [72], [72], [72], [74], [73], [74]]   # skill + 1
corrs  = [0, 1, 0, 1, 1, 1, 1, 1]
Δt     = [0, 0.347, 0.230, 0.236, 0.814, 0.370, 0.461, 0.090]  # phút = ms_first_response lượt TRƯỚC / 60000
times  = [0, 0, 1, 1, 2, 2, 2, 3]                           # cộng dồn (chặn 240/khoảng), làm tròn SAU khi cộng
```

`_assert_align` kiểm tra `skills0` và `corrs` trùng **từng phần tử** với chuỗi Cấp 1 của đúng
người này; lệch là dừng. Vì vậy DKT, GKT, GRKT dự đoán đúng cùng 39.385 nhãn test.

> Nhận xét quan trọng: Δt trên ASSIST2009 chỉ là thời gian làm bài (vài chục giây), nên `times`
> tăng rất chậm (8 lượt = 3 phút). Đó là lý do pha học/quên của GRKT gần như không làm gì trên
> tập này (§6.3: biên độ học/quên 0,0002 so với củng cố 0,063).

### 2.6. Chia tập

`random_split` với `Generator().manual_seed(42)`, **mỗi phần tử là một người học** → 2.311 /
770 / 771 người (198.344 tương tác sau cắt 200, tỉ lệ đúng 63,7%). Danh sách người học được ghi
vào `results/level1_out/split.json`.

---

## 3. Đồ thị bằng ví dụ thật

### 3.1. Transition (GKT, Cấp 1)

Đếm trên 114.653 cặp lượt liên tiếp của **người học train**; **76,7%** cặp là cùng kỹ năng
(đường chéo, bị bỏ). Còn 2.665 cạnh khác 0 (mật độ 21,4%).

Hàng của skill 71 *Addition and Subtraction Positive Decimals* (đã chuẩn hoá hàng):

| → khái niệm đích | số lần n(71→j) | A[71, j] |
|---|---|---|
| 73 Addition and Subtraction Fractions | 232 | 0,393 |
| 72 Multiplication and Division Integers | 81 | 0,137 |
| 34 Addition Whole Numbers | 48 | 0,081 |
| 69 Multiplication and Division Positive Decimals | 40 | 0,068 |
| 70 Addition and Subtraction Integers | 30 | 0,051 |

Đọc: "sau khi luyện cộng trừ số thập phân, 39% lần chuyển sang kỹ năng khác là sang cộng trừ
phân số". Đây là **thứ tự trong chương trình/bài tập giáo viên giao**, không phải quan hệ
tiên quyết đã kiểm chứng. Người 64525 ở §2 đi đúng 71 → 73 → 72 → 73.

### 3.2. rel / pre (GRKT, Cấp 2)

Ô khác 0 trước khi nhị phân hoá: rel 7.194, pre 6.819 (tổng 14.013 = dòng "Số cạnh đồ thị từ
train" trong leak report). Sau ngưỡng 0,6: **rel 3.246 cạnh, pre 2.609 cạnh**.

Ví dụ hàng của skill 89 *Equation Solving Two or Fewer Steps*:

| | khái niệm cột | giá trị |
|---|---|---|
| pre | 4 Scatter Plot | 0,749 |
| pre | 15 Interior Angles Triangle | 0,836 |
| pre | 13 Probability of a Single Event | 0,755 |
| rel | 4 Scatter Plot | 0,683 |
| rel | 6 Table | 0,627 |

Cách đọc `pre[x, y] = 0,836`: trong các cặp người học làm hai khái niệm với kết quả **khác
nhau**, 83,6% là "x **sai**, y **đúng**". Trên toàn bộ 2.609 cạnh pre, **87%** cạnh có cột y
dễ hơn hàng x (tỉ lệ đúng cao hơn). ⇒ `pre[x, y] = 1` thực chất nghĩa là **y dễ hơn / là nền
của x**.

**Phải tự nói trước khi bị hỏi:** cạnh pre là thống kê về **độ khó tương đối**, không phải
quan hệ tiên quyết sư phạm. "Giải phương trình ← Góc trong tam giác" không có ý nghĩa giáo
khoa; nó chỉ nói tam giác dễ hơn phương trình với cùng nhóm học sinh.

---

## 4. Rò rỉ dữ liệu

| # | Loại rò rỉ | Phòng thế nào | Bằng chứng (số) | Code |
|---|---|---|---|---|
| 1 | Cùng một người học ở cả train và test | `random_split` trên dataset mà **mỗi phần tử là một người** | leak report: `train∩val=0 train∩test=0 val∩test=0` | `GKT/processing.py` `load_dataset`; `level2/grkt_data.py` `leak_report` |
| 2 | Đồ thị dựng bằng dữ liệu test | Transition chỉ đếm `train_dataset.indices`; rel/pre chỉ đếm chuỗi train | dựng lại từ val cho 11.415 ô ≠ 14.013 từ train → đồ thị phụ thuộc tập dựng | `build_transition_graph`, `build_relevance_maps` |
| 3 | Lịch sử chứa lượt **tương lai** | sắp theo `order_id` trong từng người | trước sửa 70,6% bước dự đoán; nay `order_id không tăng = 0` trên 194.492 bước | `prepare_interactions`, `sequence_order_check` |
| 4 | Nhãn mục tiêu là **bản sao** của bước trước | giữ 1 dòng mỗi `(user_id, order_id)` | trước sửa 8,9% bước; nay `lượt trùng = 0` | như trên |
| 5 | Dùng nhãn của bước đang dự đoán | GKT/DKT: `pred[t]` chỉ dùng `x_0..x_t` + danh tính `q_{t+1}`; GRKT: pha học/quên chỉ dùng `knows[i+1]`, `times[i+1]`, **không** dùng `corrs[i+1]` | đọc code: `_get_next_pred(yt, questions[:, i+1])`; GRKT dự đoán bước i trước khi cộng `corr_1` | `GKT/models.py`, `GRKT/model/GRKT.py` |

Còn một rò rỉ nhỏ **đã khai báo**: Junyi chọn 200 khái niệm phổ biến nhất trên **toàn bộ** log
(transductive). Nó chỉ quyết định khái niệm nào được giữ, không đưa nhãn test vào mô hình.

### 4.1. Đọc leak report Cấp 2 từng dòng (`results/level2_moderate/assist2009/leak_report.txt`)

| Dòng | Ý nghĩa |
|---|---|
| `Người học train/val/test: 2311/770/771 (split seed=42)` | đúng split Cấp 1 |
| `Giám sát giao nhau: 0 0 0` | không ai ở hai tập |
| `Số tương tác: train 116964, val 41224, test 40156` | sau cắt 200 |
| `n_knows=112 (Cấp độ 1 + 1 cho padding)` | 111 khái niệm + hàng 0 padding của GRKT; Cấp 1 có 112 khái niệm 0..111 nên **GRKT có 113 hàng** — dòng chữ in ra hơi gây nhầm, xem §7 |
| `Số cạnh đồ thị từ train: 14013` / `nếu dựng từ val: 11415` | ô khác 0 của rel + pre trước ngưỡng |
| `biến thể train: rel 3246 pre 2609` / `zero: 0 0` | cạnh sau ngưỡng 0,6 |
| `sắp lại theo order_id: 2893 người` | số người có thứ tự file ≠ thời gian |
| `gộp lượt trùng: bỏ 176103 dòng` | 121.207 lặp y hệt + 54.896 kỹ năng phụ |
| `194492 bước dự đoán, order_id không tăng=0, lượt trùng=0` | chứng minh #3, #4 |

Từ commit `868838c`, leak report Cấp 2 có thêm dòng **"Thứ tự trong chuỗi GRKT"** đếm trực tiếp
trên chuỗi GRKT (trước đó chỉ đếm trên chuỗi Cấp 1 rồi suy ra). Dòng này chỉ xuất hiện sau
lần chạy lại notebook Cấp 2 kế tiếp.

---

## 5. Kết quả Cấp 1 và cách đọc

Nguồn: `results/level1_out/report_summary.md`, `tables/`. Split cố định seed 42; seed
42/123/456 chỉ đổi khởi tạo + thứ tự batch.

### 5.1. Bảng chính (3 seed, mean ± std)

| Mô hình | AUC | Acc | NLL | Tham số | Thời gian/seed |
|---|---|---|---|---|---|
| **DKT** | **0,7608 ± 0,0005** | 0,7228 | 0,5500 | 36.720 | 39 s |
| GKT-Transition | 0,7358 ± 0,0010 | 0,7081 | 0,5720 | 33.073 | 2.668 s |
| GKT-Shuffled | 0,7290 ± 0,0045 | 0,7069 | 0,5764 | 33.073 | 3.547 s |
| GKT-Zero | 0,7183 ± 0,0027 | 0,6952 | 0,5833 | 33.073 | 2.665 s |

### 5.2. Kiểm định ghép cặp (paired t-test trên 3 seed)

| So sánh | ΔAUC | CI 95% | p | Kết luận |
|---|---|---|---|---|
| DKT − GKT-Transition | +0,0250 | [+0,0212; +0,0288] | 0,001 | DKT tốt hơn, có ý nghĩa |
| GKT-Transition − GKT-Zero | +0,0175 | [+0,0101; +0,0249] | 0,009 | có cạnh tốt hơn không cạnh |
| GKT-Transition − GKT-Shuffled | +0,0068 | [−0,0083; +0,0219] | 0,193 | **không** có ý nghĩa |
| GKT-Shuffled − GKT-Zero | +0,0108 | [−0,0113; +0,0328] | 0,171 | không có ý nghĩa |

**Kết luận đúng:** có cạnh giúp GKT, nhưng cạnh *thật* chưa hơn cạnh *xáo* một cách có ý
nghĩa → chưa chứng minh được giá trị của **cấu trúc** đồ thị. Với n = 3 seed, độ mạnh kiểm định
thấp; nói "chưa chứng minh", không nói "không có".

### 5.3. Tắt đồ thị lúc suy luận (A := 0 trên mô hình đã train)

| Mô hình | AUC có đồ thị | AUC khi A:=0 | ΔAUC | mean \|Δp\| |
|---|---|---|---|---|
| GKT-Transition | 0,7346 | 0,7056 | −0,0291 | 0,055 |
| GKT-Shuffled | 0,7338 | 0,7115 | −0,0223 | 0,044 |
| GKT-Zero | 0,7152 | 0,7152 | 0 | 0 (kiểm tra cài đặt: đúng phải bằng 0) |

Mô hình đã học **dựa vào** đồ thị (tắt đi thì giảm AUC), cả với đồ thị xáo.

### 5.4. Vì sao DKT thắng GKT? (câu hỏi chắc chắn gặp)

1. 76,7% bước chuyển là cùng kỹ năng → thông tin "khái niệm A → B" ít.
2. GKT chỉ lan truyền từ **một** khái niệm vừa làm, 1 hop, trạng thái 32 chiều mỗi khái niệm;
   DKT có LSTM nhìn cả chuỗi.
3. GKT-Zero vẫn cập nhật mọi khái niệm qua GRU (BAO_CAO §4.4) nên đồ thị chỉ thêm một phần.
4. Bài báo GKT báo GKT > DKT (0,721 vs 0,709) trên tiền xử lý khác (lọc skill ≥ 10 lần, 1.000
   người); không so trực tiếp được.

### 5.5. Hiệu chỉnh

DKT ECE 0,011; GKT-Transition 0,024 (dự đoán cao hơn thực tế 2,3 điểm phần trăm); GKT-Shuffled
0,033. Tức GKT hơi **quá tự tin về phía đúng**.

### 5.6. Case study (5 đúng + 5 sai, GKT-Transition)

`evidence_ok = True` ở cả 10 ca (chạy lại tái tạo đúng p, sai số ≤ 0,02) — điều này chỉ có
được **sau khi sửa BatchNorm**. Ví dụ ca sai rank 1: người 78863, bước 107, *Addition Whole
Numbers*, mô hình đoán p = 0,967 nhưng học sinh làm sai; lật một lượt trong lịch sử đổi p tối
đa 0,08; tắt đồ thị đổi p +0,09. Xem §7 về cột `max_occ_abs_dp`.

---

## 6. Kết quả Cấp 2 và cách đọc

Nguồn: `results/level2_moderate/`. Baseline DKT/GKT được **train lại** trên đúng bundle của
Cấp 2 (không dùng checkpoint Cấp 1), nên số seed 42 lệch nhẹ so với Cấp 1 (DKT 0,7608 vs
0,7614) do GPU không tất định tuyệt đối.

### 6.1. ASSIST2009 (seed 42, cùng 39.385 nhãn test)

| Mô hình | AUC | Acc | NLL |
|---|---|---|---|
| **GRKT-train** | **0,7691** | 0,7259 | 0,547 |
| DKT | 0,7608 | 0,7237 | 0,549 |
| GRKT-zero | 0,7538 | 0,7131 | 0,563 |
| GKT-Transition | 0,7325 | 0,7081 | 0,572 |
| GKT-Zero | 0,7233 | 0,7000 | 0,583 |

### 6.2. Đa seed GRKT (seed 42/123/456)

| | AUC mean ± std | NLL mean |
|---|---|---|
| GRKT-train | 0,7694 ± 0,0006 | 0,5489 |
| GRKT-zero | 0,7551 ± 0,0011 | 0,5606 |

Paired (`multiseed_paired.csv`): **ΔAUC = +0,0143**, CI 95% [+0,0124; +0,0161], p = 0,0009;
từng seed +0,0150 / +0,0136 / +0,0141. ΔNLL = −0,0117, p = 0,027.

**Kết luận đúng:** với GRKT, đồ thị train-only giúp ổn định khoảng +0,014 AUC — đây là bằng
chứng vai trò đồ thị **sạch nhất** của đồ án, vì train và zero giống hệt nhau trừ tập cạnh.
Nhưng: chưa có biến thể `shuffled` cho GRKT, nên chưa tách được "có cạnh" và "cạnh đúng".

GRKT − DKT = +0,0083 **không** phải bằng chứng cho đồ thị: GRKT còn có độ khó từng câu hỏi
(17.737 câu, embedding chiếm 83% trong 2,7 triệu tham số). GRKT-zero (0,7538) đã gần DKT.

### 6.3. Quỹ đạo mastery (`trajectory_metrics.csv`, 150 người học test, 6.963 bước)

| Tiêu chí | train | zero | Đọc thế nào |
|---|---|---|---|
| `consistency_strengthen` | 1,0 | 1,0 | **kiểm chứng cài đặt**, đúng theo cấu trúc (ReLU + trọng số dương), không phải bằng chứng |
| `locality` | 1,0 | 1,0 | như trên (k_hop = 1) |
| `precedence` | 0,510 | 0,354 | tỉ lệ cạnh có mastery(nền) ≥ mastery(phụ thuộc) khi học sinh vừa làm đúng khái niệm phụ thuộc |
| `precedence_gap` | **−0,007** | −0,036 | trừ đi mức khi xáo cạnh; ≈ 0 → **chưa đạt** trên ASSIST2009 |
| `stability_spearman` | 0,983 | 0,993 | lật bước áp chót, thứ hạng mastery cuối gần như giữ nguyên → ổn định |
| `stability_changed_frac` | 0,527 | 0,009 | train: lật 1 bước đổi mastery ~53% khái niệm (lan qua đồ thị); zero: 0,009 ≈ **1/112** = chỉ đúng khái niệm đó |
| `forget_slope` | −4,2·10⁻⁴ | −4,2·10⁻⁴ | âm nhưng rất nhỏ |
| biên độ củng cố / học-quên / truy xuất | 0,063 / **0,0002** / 0,025 | 0,077 / 0,0002 / 0 | pha học/quên gần như tắt (Δt nhỏ, §2.5) |

Junyi (chỉ train): `precedence_gap = +0,019` (đúng chiều, nhỏ); `forget_slope = +1·10⁻⁵` ≈ 0 →
**không thấy quên** (report tự động ghi "(âm = …)" ngay cạnh số dương, đừng đọc nhầm).

### 6.4. Hiệu chỉnh Cấp 2 (ASSIST2009)

Brier thấp nhất: GRKT-train 0,1831. ECE thấp nhất: DKT 0,0065 (GRKT-train 0,0117). GRKT-zero
và GKT-Zero có ECE cao nhất (0,033; 0,037).

### 6.5. Junyi (tập con 3.000 / 72.630 người, 1 seed, chỉ biến thể train)

GRKT-train 0,7896 > DKT 0,7501 > GKT-Zero 0,7225 ≈ GKT-Transition 0,7202. Không có GRKT-zero
nên **không** kết luận được vai trò đồ thị trên Junyi.

---

## 7. Điểm yếu đã biết trong `results/`

Những chỗ thầy nhìn vào bảng là thấy. Nên chủ động nói trước.

| # | Chỗ | Vấn đề | Trả lời |
|---|---|---|---|
| 1 | `report_summary.md` Cấp 1 §9, cột `max_occ_abs_dp` = 2,21; 2,30… | Δp của phép che lượt **> 1**, vô lý với xác suất | Lỗi đã biết của cell §7 notebook: che lượt ngay trước bước dự đoán đặt `questions = -1`, `GKT._predict` không sigmoid hàng đó nên trả **logit**. Các cột khác (lật kết quả, tắt đồ thị) không bị ảnh hưởng. Sửa: xoá hẳn lượt thay vì đặt −1; chỉ cần chạy lại phần case study, không cần train lại. |
| 2 | Cấp 1 case 5: `54:nan` | thiếu tên kỹ năng | 12 skill trong file gốc không có `skill_name` (id 37, 76, 91, 94, 96, 97, 99, 101, 102, 104, 105, 367); 10 skill trong số đó còn lại sau dedup. Không phải lỗi mã |
| 3 | Cấp 1 §11 bảng đối chiếu ghi "5 seed" | thực tế chạy 3 seed (42, 123, 456) | chữ cố định trong notebook; bảng số đúng là 3 |
| 4 | Cấp 2 §10 hạn chế 5 "ΔAUC chỉ từ một seed" | lỗi thời — đã có 3 seed (§6.2) | report sinh từ mẫu chữ cũ |
| 5 | Cấp 2 §6 "split.json C1: missing", "Tập người học trùng: None" | notebook Cấp 2 không đính kèm output Cấp 1 nên không so file | Cấp 2 *tự dựng lại* bundle cùng hàm, cùng seed, và `_assert_align` so từng người học/nhãn; số 2311/770/771 và 39.385 nhãn test trùng Cấp 1 |
| 6 | leak report "n_knows=112 (Cấp độ 1 + 1 cho padding)" | thực tế `concept_num` = 112 khái niệm, GRKT thêm hàng 0 → ma trận 113×113 (`graphs_train_only.npz`) | chữ in ra gây nhầm, số không sai |
| 7 | Junyi `forget_slope` dương | không quan sát được quên | nói thẳng: trên tập con này không thấy hiệu ứng quên |
| 8 | `precedence_gap` ASSIST2009 âm nhẹ | tiêu chí thứ tự chưa đạt | báo cáo đúng như vậy; pre là thống kê độ khó (§3.2) |
| 9 | Mất 11 khái niệm | dedup giữ `skill_id` nhỏ nhất | cách giữ đủ (gộp vào `knows` của GRKT) cần train lại; GKT/DKT không nhận nhiều khái niệm một bước |
| 10 | GRKT `k_hop = 1`, bài báo L = 2 | lệch cấu hình bài báo | chọn để vừa VRAM/thời gian Kaggle |

---

## 8. Thầy chỉ vào X thì trả lời gì

Bổ sung cho bộ câu hỏi ở [BAO_CAO_KIEN_TRUC.md §14](BAO_CAO_KIEN_TRUC.md#14-bộ-câu-hỏi-vấn-đáp-dự-kiến).

**"Dòng này trong CSV là gì?"** Một lần học sinh trả lời một câu, *cho một kỹ năng*. Câu nhiều
kỹ năng = nhiều dòng cùng `order_id` (§2.2). Cột dùng: `user_id`, `order_id` (thứ tự thời gian),
`skill_id`, `correct`; GRKT thêm `problem_id`, `ms_first_response`.

**"Tại sao `x = 2·skill + correct`?"** Để một số nguyên mã hoá cả khái niệm và kết quả; one-hot
2C chiều. Sai skill 71 = 142, đúng = 143.

**"Lượt đầu tiên có được dự đoán không?"** Không. Chuỗi dài T cho T−1 dự đoán; nhãn so với
`answers[:, 1:]`.

**"Cắt 200 thì mất gì?"** 323 người dài hơn 200 lượt bị bỏ phần sau (không cắt thành nhiều
cửa sổ). Trung bình 51,5, trung vị 23 lượt.

**"Tại sao đồ thị Shuffled có 2.146 cạnh mà Transition 2.665?"** Hoán vị toàn cục chỉ số cột
làm một số ô trùng nhau bị gộp. `ShuffledRow` (xáo trong từng hàng) giữ đúng 2.665.

**"Tại sao Identity ≡ Zero?"** Vị trí khái niệm đang làm luôn bị ghi đè bằng `f_self`, nên
đường chéo A không bao giờ được dùng.

**"Số 14.013 trong leak report là gì?"** Ô khác 0 của rel + pre (đã lọc cặp ít dữ liệu) trước
ngưỡng 0,6. Dựng từ val ra 11.415 → đồ thị phụ thuộc tập dựng, nên phải dựng từ train.

**"Vì sao GRKT-zero vẫn hơn GKT?"** Vì GRKT có độ khó từng câu hỏi (IRT: năng lực − độ khó) và
nhiều tham số hơn ~80 lần. Đó là lý do so GRKT với DKT không nói gì về đồ thị.

**"p = 0,0009 với 3 seed có đáng tin?"** Ba chênh lệch rất đều (+0,0150/+0,0136/+0,0141) nên
t lớn. Nhưng chỉ đo phương sai **khởi tạo**, không đo phương sai **split** (split cố định);
kết luận giới hạn trong split này.

**"Nếu chênh lệch giữa các seed giống hệt nhau thì p bao nhiêu?"** Từ commit `868838c`,
`_paired_vs_zero` ghi **NaN** cho p và CI thay vì một p cực nhỏ giả tạo (độ lệch chuẩn 0).

**"Δt của ASSIST2009 lấy từ đâu?"** Không có timestamp; dùng `ms_first_response` của lượt trước
(phút), cộng dồn, chặn 240. Nó chỉ phản ánh thời gian làm bài, nên pha quên gần như không
hoạt động (§6.3).

**"Kết quả có tái lập được không?"** Split seed 42 + `split.json`; mỗi run có fingerprint (dữ
liệu, `prep_version`, split, đồ thị, siêu tham số). Từ commit `868838c`, fingerprint của
checkpoint GRKT và baseline Cấp 2 cũng có `prep_version`. GPU không tất định tuyệt đối và GRKT
lấy mẫu Gumbel cả lúc eval, nên lệch ở chữ số thứ 3–4.

**"Code nào nhóm viết?"** `level1/`, `level2/`, hai notebook, và các sửa đổi trong `GKT/`,
`GRKT/` (BAO_CAO §6, §9). Công thức mô hình là của tác giả gốc.
