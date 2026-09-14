# Ý tưởng và kế hoạch — Project 02: Knowledge Tracing trên đồ thị

Tài liệu này đề xuất hướng làm đồ án *Theo dõi tri thức người học bằng học máy trên đồ thị* (nhánh A: đúng/sai), bám theo 3 nhiệm vụ và các cấp độ điểm trong đề.

---

## 1. Bài toán cần giải

**Knowledge Tracing (KT):** tại bước \(t\), mô hình nhìn chuỗi tương tác trước đó của người học \((q_1, r_1), \ldots, (q_{t-1}, r_{t-1})\) cùng câu hỏi hiện tại \(q_t\), rồi:

1. **Dự đoán** xác suất trả lời đúng \(P(r_t = 1 \mid \text{lịch sử}, q_t)\).
2. **Mô tả năng lực:** vector trạng thái tri thức theo từng khái niệm (knowledge concept, KC) và quỹ đạo của nó theo thời gian.
3. **Giải thích (xAI):** chỉ ra lần làm bài nào và cạnh/khái niệm nào trên đồ thị đã đẩy mô hình tới kết luận đó.

Câu hỏi và khái niệm **không độc lập**: một câu hỏi gắn một hoặc nhiều KC (Q-matrix); các KC có quan hệ tiên quyết hoặc đồng xuất hiện. GNN lan truyền tín hiệu trên đồ thị KC; mô hình thời gian (GRU/LSTM/attention) cập nhật trạng thái người học sau mỗi lần tương tác.

```
lịch sử làm bài
        │
        ▼
 Q-matrix + đồ thị khái niệm
        │
        ▼
 GCN / GAT / GraphSAGE  (biểu diễn KC)
        │
        ▼
 GRU / LSTM / GRKT      (cập nhật trạng thái người học)
        │
        ├──► P(đúng | câu hỏi tiếp theo)     [Nhiệm vụ 1]
        ├──► mastery(KC, t) theo thời gian   [Nhiệm vụ 2]
        └──► bằng chứng: tương tác + cạnh    [Nhiệm vụ 3]
```

---

## 2. Lựa chọn phạm vi

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Nhánh | **A. Đúng/sai** | Bắt buộc để khởi động; dữ liệu ASSISTments/Junyi sẵn; không cần nội dung phương án |
| Mục tiêu điểm | **Cấp 1 + 2 + 3** (trần 10) | Cấp 4 (+1) chỉ làm nếu còn thời gian |
| Dữ liệu công khai | **ASSISTments 2009** và **Junyi Academy** | ASSIST09 dễ chạy cấp 1; Junyi có quan hệ tiên quyết — đúng tinh thần đồ thị; GRKT đã báo cáo trên cả hai |
| Phương pháp 2024+ (cấp 2) | **GRKT** (KDD 2024) | Có mã nguồn; trạng thái mastery tường minh; 3 giai đoạn retrieval / strengthening / learning-forgetting; giải thích tự nhiên hơn PSI-KT |
| Phương án dự phòng cấp 2 | PSI-KT (ICLR 2024) | Nếu GRKT khó tái lập; mạnh về đồ thị tiên quyết và diễn giải Bayesian |
| Baseline không đồ thị | **DKT** (bắt buộc tối thiểu) + **simpleKT** nếu kịp | DKT dễ implement; simpleKT là baseline ICLR 2023 mạnh hơn |
| Mô hình đồ thị cấp 1 | **GAT + GRU** (tự implement) và/hoặc **GKT** | GAT cho attention trên cạnh → bằng chứng xAI; GKT là mô hình KT-đồ thị kinh điển |

**Không làm (trừ khi cấp 4):** nhánh B (dự đoán phương án), EdNet đầy đủ, hệ thống LMS/web hoàn chỉnh.

---

## 3. Ý tưởng giải pháp

### 3.1. Dữ liệu và chống rò rỉ

Mỗi dòng tương tác tối thiểu: `user_id`, `question_id`, `skill_id`(s), `correct` ∈ {0,1}, `order`/`timestamp`.

Quy trình bám **pyKT** (`pykt-toolkit`):

- Cắt chuỗi theo độ dài tối đa (ví dụ 200), bỏ người học quá ít tương tác (ví dụ < 5).
- **Split theo người học** (user-wise): một user chỉ thuộc một trong train / val / test. Lưu `split.json` + seed.
- Đồ thị đồng xuất hiện / chuyển tiếp **chỉ xây từ tập train**.
- Đặc trưng tại bước \(t\) **không** dùng nhãn \(r_t\) hay tương lai.
- Mọi mô hình dùng **cùng một file split**. Chạy ≥ 3 seed, báo cáo mean ± std.

### 3.2. Xây dựng đồ thị khái niệm

- **Đỉnh:** mỗi knowledge concept / skill.
- **Cạnh (ưu tiên theo nguồn):**
  1. Junyi: cạnh tiên quyết do dữ liệu cung cấp (có hướng).
  2. ASSIST09: cạnh **đồng xuất hiện** trên câu hỏi đa-KC, hoặc **chuyển tiếp** (user làm KC A rồi KC B) từ train, ngưỡng tần suất.
- **Trọng số cạnh:** tần suất chuẩn hóa, hoặc 1/0 sau khi lọc.
- **Ablation bắt buộc (cấp 1):** (i) bỏ hết cạnh, (ii) ma trận kề = đơn vị (identity), (iii) tùy chọn: xáo trộn cạnh (edge shuffle). Mục tiêu: chứng minh cấu trúc đồ thị **thật sự** đóng góp, không phải chỉ tăng tham số.

Q-matrix \(Q \in \{0,1\}^{|questions| \times |KC|}\): nếu câu hỏi có nhiều skill thì lấy trung bình / attention trên các KC liên quan khi dự đoán.

### 3.3. Cấp độ 1 — Dự đoán bước tiếp theo

**Baseline không đồ thị — DKT**

- Embedding cặp \((q, r)\) hoặc \((kc, r)\).
- LSTM/GRU trên chuỗi; đầu ra: \(P(\text{đúng} \mid q_t)\).
- Loss: Binary Cross-Entropy trên các bước có nhãn.

**Mô hình đồ thị — GAT-KT (đề xuất chính)**

1. Khởi tạo embedding đỉnh KC: \(h_c^{(0)}\).
2. \(L\) lớp GAT: mỗi KC nhận thông tin từ láng giềng (tiên quyết / đồng xuất hiện). Attention \(\alpha_{ij}\) là tín hiệu giải thích cấp 1.
3. Trạng thái người học \(s_t\): GRU nhận \([h_{q_t}; e_{r_t}]\) (biểu diễn câu hỏi = tổng/attention các KC của câu đó, cộng embedding đúng/sai).
4. Điểm dự đoán: \(\hat{y}_t = \sigma(s_{t-1}^\top W h_{q_t} + b)\) — **dùng \(s_{t-1}\)** chứ không phải \(s_t\) để không rò nhãn hiện tại.

Có thể thêm GKT nếu thời gian cho phép (cập nhật trực tiếp “ô” KC trên đồ thị sau mỗi lần trả lời).

### 3.4. Cấp độ 2 — Mô tả năng lực + GRKT

DKT ẩn trạng thái trong hidden vector nên **khó đọc mastery theo KC**. GRKT giải quyết đúng yêu cầu nhiệm vụ 2:

Ba giai đoạn trên đồ thị KC (tóm tắt):

1. **Knowledge retrieval:** truy xuất tri thức liên quan câu hỏi hiện tại, có chuyển giao sang KC láng giềng (transfer of learning).
2. **Memory strengthening:** củng cố những KC vừa được kích hoạt (testing effect).
3. **Learning / forgetting:** tăng mastery nếu làm đúng, giảm theo thời gian/khoảng cách nếu không ôn (Ebbinghaus-style).

Đầu ra hữu ích cho đồ án:

- Vector mastery \(m_{u,c,t} \in [0,1]\) cho từng user–concept–thời điểm → vẽ quỹ đạo.
- Dự đoán \(\hat{y}_t\) vẫn đánh giá AUC/ACC như cấp 1.
- So sánh quỹ đạo GAT-KT vs DKT vs GRKT trên cùng user.

**Hai tiêu chí hợp lý của quỹ đạo (bắt buộc đo, chọn ≥ 2):**

| Tiêu chí | Cách đo (gợi ý) |
|---|---|
| **Hướng nhất quán** | Khi user trả lời đúng câu gắn KC \(c\), \(\Delta m_c \ge -\varepsilon\) (không tụt mastery một cách bất hợp lý). Tỷ lệ vi phạm trên tập test. |
| **Tính cục bộ** | Một lần trả lời KC \(c\) không làm \(\lvert\Delta m_{c'}\rvert\) lớn với \(c'\) không kề và không cùng câu hỏi. Trung bình \(\lvert\Delta m\rvert\) ngoài 1-hop vs trong 1-hop. |
| **Nhất quán tiên quyết** (Junyi) | Mastery KC con không vượt xa KC tiên quyết một cách có hệ thống (nếu đồ thị có hướng). |
| **Ổn định** | Nhiễu nhỏ lịch sử (đổi 1 nhãn xa) → thay đổi mastery tại bước cuối nhỏ (Lipschitz / correlation). |

Khuyến nghị đo **hướng nhất quán** + **tính cục bộ** trên cả hai dataset; thêm tiên quyết trên Junyi như phân tích nâng cao cấp 2.

### 3.5. Cấp độ 3 — Bằng chứng / xAI + quiz nhỏ

Với **mỗi** trong ≥ 10 case (5 đúng + 5 sai ở cấp 1; đủ 10 case có bằng chứng đồ thị ở cấp 3), ghi:

1. User id, bước \(t\), câu hỏi, KC, nhãn thật, \(\hat{y}\).
2. Top-k lần tương tác trước đó có ảnh hưởng lớn (attention GRU/GAT, hoặc gradient × input / occlusion: che một bước rồi đo \(\Delta\hat{y}\)).
3. Top-k cạnh/KC trên đồ thị: \(\alpha_{ij}\) GAT, hoặc KC mà GRKT vừa retrieval/strengthen.
4. Một câu giải thích tiếng Việt: *“Mô hình dự đoán đúng vì user vừa làm đúng 3 câu cùng KC X và KC Y là tiên quyết của X.”*

**Quiz tự thu thập (gợi ý):**

- Chủ đề hẹp (ví dụ: phương trình bậc nhất / đạo hàm cơ bản / SQL SELECT).
- 4–6 khái niệm có quan hệ tiên quyết rõ (vẽ tay đồ thị).
- 20–30 câu hỏi, Q-matrix do nhóm gán.
- 10–20 người, ≥ ~400 tương tác tổng.
- Fine-tune GRKT đã train trên ASSIST/Junyi **hoặc** đánh giá zero-shot transfer rồi mới fine-tune; split theo user hoặc 5-fold theo user.

Không cần web app: một notebook Streamlit/Gradio đủ: chọn user → lịch sử, \(\hat{y}\), heatmap mastery, highlight bằng chứng.

### 3.6. Cấp độ 4 (tuỳ chọn, +1)

Giả thuyết đề xuất nếu còn bandwidth:

> **Time-decayed graph propagation:** tín hiệu từ lần làm bài cách xa nên suy giảm khi lan trên đồ thị KC (cạnh × \(\exp(-\Delta t / \tau)\)), giúp quỹ đạo ổn định hơn và AUC không giảm.

Bắt buộc: giả thuyết + ablation (có/không decay) + 2 dataset công khai. Hướng khác hợp lệ: GNNExplainer + fidelity, học cấu trúc cạnh, ước lượng bất định.

---

## 4. Đánh giá

Mọi mô hình, cùng split, ≥ 3 seed.

**Nhiệm vụ 1**

- AUC, Accuracy, BCE/NLL.
- Ablation đồ thị (no-edge, identity, shuffle).
- Tuỳ chọn: Brier / ECE (calibration) — đủ cho “phân tích nâng cao” cấp 2.

**Nhiệm vụ 2**

- Heatmap / line chart mastery theo KC × thời gian cho vài user điển hình (mạnh, yếu, dao động).
- Bảng 2+ chỉ số quỹ đạo (mục 3.4), so sánh DKT vs GAT-KT vs GRKT.

**Nhiệm vụ 3**

- 5 đúng + 5 sai (mọi nhóm); cấp 3: 10 case có cạnh đồ thị.
- Occlusion / attention rank làm bằng chứng định lượng nhẹ.

---

## 5. Kiến trúc thư mục đề xuất

```
Knowledge-Tracing/
├── KE_HOACH.md
├── README.md                 # hướng dẫn chạy lại
├── requirements.txt
├── configs/                  # seed, hyperparam, đường dẫn split
├── data/
│   ├── raw/                  # không commit dữ liệu lớn
│   ├── processed/
│   └── splits/               # train_users.json, ...
├── src/
│   ├── preprocess.py         # pyKT-style
│   ├── graph.py              # Q-matrix, adjacency từ train
│   ├── datasets.py
│   ├── models/
│   │   ├── dkt.py
│   │   ├── gat_kt.py
│   │   └── grkt/             # port / wrapper mã gốc
│   ├── train.py
│   ├── evaluate.py
│   ├── metrics_trajectory.py
│   └── explain.py            # attention, occlusion
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_case_study.ipynb   # 5+5 / 10 case
│   └── 03_dashboard.ipynb
├── experiments/              # log AUC, ablation
└── quiz/                     # đề, Q-matrix, dữ liệu nhóm
```

Stack gợi ý: Python, PyTorch, PyTorch Geometric, pandas, scikit-learn, matplotlib/plotly; dashboard: Streamlit hoặc notebook.

---

## 6. Kế hoạch thực hiện

Giả định nhóm ≤ 4 người, khoảng **8–10 tuần**. Điều chỉnh mốc theo lịch nộp Moodle.

### Giai đoạn 0 — Khởi động (tuần 1)

- [ ] Đọc DKT, GKT, GRKT (arxiv 2406.12896), skim pyKT.
- [ ] Chốt split protocol và bảng phân công.
- [ ] Tải ASSIST09 + Junyi; chạy thống kê: số user, câu, KC, độ dài chuỗi, độ lệch nhãn.
- [ ] Môi trường conda/venv + `requirements.txt`.

### Giai đoạn 1 — Cấp độ 1 (tuần 2–4) → trần 7.0

- [ ] Pipeline preprocess + user-wise split cố định.
- [ ] Xây đồ thị ASSIST (co-occurrence) và Junyi (prerequisite).
- [ ] Train DKT trên ASSIST09; kiểm tra không leak (sanity: shuffle nhãn → AUC ~ 0.5).
- [ ] Implement GAT-KT; train ASSIST09 rồi Junyi.
- [ ] Ablation no-edge / identity; bảng AUC/ACC/BCE, 3 seed.
- [ ] Chọn 5 case đúng + 5 case sai; viết evidence lịch sử + KC.

**Cột mốc:** notebook chạy được 1 user, in \(P(\text{đúng})\) và so sánh DKT vs GAT-KT.

### Giai đoạn 2 — Cấp độ 2 (tuần 5–7) → trần 8.5

- [ ] Tái lập GRKT (code: https://github.com/JJCui96/GRKT); map vào split của nhóm (không dùng split mặc định khác nếu muốn so sánh công bằng — hoặc báo cáo cả hai và giải thích).
- [ ] Đánh giá GRKT trên **cả hai** dataset, cùng chỉ số với cấp 1.
- [ ] Export mastery \(m_{u,c,t}\); vẽ quỹ đạo.
- [ ] Implement 2 metric quỹ đạo; so sánh 3 mô hình.
- [ ] Một phân tích nâng cao: ECE **hoặc** user lịch sử ngắn **hoặc** so sánh quỹ đạo có/không đồ thị.

### Giai đoạn 3 — Cấp độ 3 (tuần 7–9) → trần 10

- [ ] `explain.py`: occlusion + GAT attention + (nếu có) trọng số retrieval GRKT.
- [ ] 10 case đầy đủ bằng chứng đồ thị.
- [ ] Thiết kế quiz 4–6 KC, 20–30 câu; thu thập ≥ 400 tương tác.
- [ ] Fine-tune / đánh giá chuyển giao GRKT trên quiz.
- [ ] Dashboard: user → lịch sử, xác suất, mastery, evidence.

### Giai đoạn 4 — Báo cáo, demo, buffer (tuần 9–10)

- [ ] Báo cáo 12–18 trang: dữ liệu, chống leak, mô hình, bảng, ablation, case study, hạn chế.
- [ ] Slide ≤ 20 phút + video dự phòng ≤ 5 phút.
- [ ] Bảng đóng góp + khai báo AI/mã nguồn mở.
- [ ] Chạy lại 1 experiment nhỏ từ README (để vấn đáp).
- [ ] Nếu dư sức: cấp 4 time-decay + ablation 2 dataset.

### Phân công gợi ý (4 người)

| Thành viên | Trọng tâm | Phải giải thích được khi vấn đáp |
|---|---|---|
| A | Preprocess, split, đồ thị, chống leak | pyKT, vì sao graph chỉ từ train |
| B | DKT + GAT-KT + ablation | công thức dự đoán, chỗ không dùng \(r_t\) |
| C | GRKT + metric quỹ đạo | 3 giai đoạn, mastery |
| D | xAI, quiz, dashboard, báo cáo/slide | 10 case, quy trình thu thập |

Mọi người vẫn train/eval được mô hình của mình trên cùng data để tránh “chỉ một người hiểu code”.

---

## 7. Rủi ro và cách xử lý

| Rủi ro | Xử lý |
|---|---|
| GRKT khó chạy / OOM | Giảm max-seq, subsample ASSIST; fallback PSI-KT hoặc DGEKT |
| AUC GAT ≈ DKT | Kiểm tra leak; tăng chất lượng cạnh (Junyi pred); báo cáo trung thực + nhấn quỹ đạo/xAI |
| Quiz không đủ 400 tương tác | Thêm người / cho làm 2 phiên; không bịa nhãn |
| Tranh luận split pyKT vs paper GRKT | Giữ **một** split nhóm cho mọi so sánh chính; appendix chạy config gốc GRKT |
| Đồ thị ASSIST quá dày/nhiễu | Ngưỡng cạnh; so sánh vài cách tạo cạnh (mở rộng nhiệm vụ 1) |

---

## 8. Sản phẩm nộp (checklist đề)

- [ ] Mã preprocess, graph, train, eval + environment + README chạy lại
- [ ] File split hoặc seed + mã tạo split; đoạn văn chống rò rỉ
- [ ] Báo cáo 12–18 trang, slide, bảng phân công
- [ ] Notebook/dashboard: user, \(P(\text{đúng})\), lịch sử, mastery, evidence
- [ ] 5 đúng + 5 sai chi tiết; cấp 3: 10 case đồ thị
- [ ] Demo live + video ≤ 5 phút

---

## 9. Tài liệu neo

- Đề: `Project02_Knowledge_Tracing.pdf`
- pyKT: https://github.com/pykt-team/pykt-toolkit
- DKT: Piech et al., NeurIPS 2015
- GKT: Nakagawa et al., WI 2019
- simpleKT: Liu et al., ICLR 2023
- **GRKT:** Cui et al., KDD 2024 — https://arxiv.org/abs/2406.12896 — https://github.com/JJCui96/GRKT
- PSI-KT (dự phòng): ICLR 2024 — https://github.com/mlcolab/psi-kt

---

## 10. Kết luận ngắn

Làm **nhánh đúng/sai**, hai dataset **ASSIST09 + Junyi**, mô hình **DKT vs GAT-KT vs GRKT**, đồ thị KC xây đúng từ train, đo AUC kèm **ablation cạnh** và **hai chỉ số quỹ đạo**, rồi đóng gói bằng **10 case xAI** và **quiz nhỏ**. Đường đi này khớp lần lượt cấp 1 → 2 → 3 của đề, tận dụng đúng điểm mạnh của học máy đồ thị: không chỉ đoán đúng/sai mà còn nói được người học đang yếu khái niệm nào và vì sao mô hình nghĩ vậy.
