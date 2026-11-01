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
| --- | --- | --- |
| Nhánh | **A. Đúng/sai** | Bắt buộc để khởi động; dữ liệu ASSISTments/Junyi sẵn; không cần nội dung phương án |
| Mục tiêu điểm | **Cấp 1 + 2 + 3** (trần 10) | Cấp 4 (+1) chỉ làm nếu còn thời gian |
| Dữ liệu công khai | **ASSISTments 2009** và **Junyi Academy** | ASSIST09 đã hoàn thành cấp 1; Junyi có quan hệ tiên quyết — đúng tinh thần đồ thị; GRKT đã báo cáo trên cả hai |
| Phương pháp 2024+ (cấp 2) | **GRKT** (KDD 2024) | Có mã nguồn; trạng thái mastery tường minh; 3 giai đoạn retrieval / strengthening / learning-forgetting; giải thích tự nhiên hơn PSI-KT |
| Phương án dự phòng cấp 2 | PSI-KT (ICLR 2024) | Nếu GRKT khó tái lập; mạnh về đồ thị tiên quyết và diễn giải Bayesian |
| Baseline không đồ thị | **DKT** (chuẩn đối sánh bắt buộc) | DKT (LSTM) nhận chuỗi tương tác, đối sánh trực tiếp với mô hình đồ thị |
| Mô hình đồ thị cấp 1 | **GKT** (Nakagawa et al., WI 2019) | Mô hình KT-đồ thị kinh điển, chuẩn hóa các dạng đồ thị: Transition, Dense, Identity, Zero |

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

### 3.3. Cấp độ 1 — Dự đoán bước tiếp theo (Đã hoàn thành thực nghiệm)

**Baseline không đồ thị — DKT (Deep Knowledge Tracing - Piech et al., NeurIPS 2015)**
- Nhận chuỗi tương tác khái niệm kèm kết quả \((c_1, r_1), \ldots, (c_t, r_t)\).
- LSTM cập nhật vector ẩn trạng thái người học qua thời gian; lớp Linear chiếu ra xác suất làm đúng khái niệm tiếp theo \(P(r_{t} = 1 \mid c_t)\).
- Loss: NLLLoss / BCE trên các bước có nhãn thực tế.

**Mô hình đồ thị — GKT (Graph-based Knowledge Tracing - Nakagawa et al., WI 2019)**
- Mỗi KC \(c\) có một vector trạng thái ẩn riêng \(h_c^t \in \mathbb{R}^{d}\) trong ma trận trạng thái người học.
- **Lan truyền đồ thị (GNN):** Khi người học tương tác với KC \(c\), hàm \(\mathcal{F}_{\text{self}}\) và \(\mathcal{F}_{\text{neighbor}}\) lan truyền thông tin sang các KC láng giềng kề cận trên ma trận kề \(A\).
- **Cập nhật thời gian:** Sử dụng Erase-Add Gate kết hợp GRUCell để cập nhật trạng thái của tất cả các KC, phản ánh mức độ thành thạo theo thời gian.
- **Khảo sát cấu trúc đồ thị & Ablation:**
  1. `Transition Graph`: Đồ thị chuyển tiếp xác suất trích xuất strictly từ tập train (chống rò rỉ).
  2. `Dense Graph`: Đồ thị liên thông toàn bộ (fully-connected graph baseline).
  3. `Identity Graph (Ablation)`: Ma trận đơn vị (không lan truyền sang láng giềng).
  4. `Zero Graph (Ablation)`: Ma trận 0 (ngắt hoàn toàn liên kết đồ thị).
- **Đánh giá đa seed:** Chạy 3 seed ngẫu nhiên (`[42, 123, 456]`) báo cáo \(\text{AUC} \pm \text{Std}\) và \(\text{Acc} \pm \text{Std}\).
- **Giải thích xAI (Nhiệm vụ 3 mức cơ bản):** Trích xuất 5 case đúng + 5 case sai kèm lịch sử tương tác và liên kết láng giềng đồ thị (`cases.md`).

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
| --- | --- |
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

## 5. Kiến trúc thư mục thực tế

```
Knowledge-Tracing/
├── KE_HOACH.md                   # Kế hoạch, nhật ký tiến độ, khai báo sửa mã nguồn mở
├── YEU_CAU.md                    # Đề bài viết lại + phụ lục đối chiếu yêu cầu ↔ triển khai
├── GAP_LEVEL1.md                 # Rà soát khoảng trống Cấp độ 1 so với yêu cầu
├── README.md                     # Hướng dẫn chạy lại và mô tả sản phẩm
├── kaggle_level1.ipynb           # Notebook điều phối Cấp độ 1 (Kaggle GPU)
├── kaggle_gkt.ipynb              # Notebook thử nghiệm GKT độc lập
├── kaggle_level2.ipynb           # Notebook Cấp độ 2: GRKT + quỹ đạo mastery + calibration
├── GKT/                          # Mã nguồn GKT & DKT của Cấp độ 1
│   ├── data/
│   │   ├── assistment_test15.csv # Tập thử nghiệm nhanh (smoke test)
│   │   └── skill_builder_data.csv# ASSISTments 2009 đầy đủ (>525k dòng)
│   ├── models.py                 # Kiến trúc GKT, DKT
│   ├── processing.py             # Nạp dữ liệu, sinh đồ thị Transition/Dense/Identity/Zero
│   ├── metrics.py                # KTLoss, ROC-AUC, Accuracy
│   ├── layers.py, utils.py       # MLP / EraseAddGate / dataloader
│   └── train.py, command.sh      # Entry point gốc của upstream
├── GRKT/                         # GRKT upstream (Cui et al., KDD 2024) đã package-hoá
│   ├── model/GRKT.py             # Kiến trúc gốc + hook ghi m_pre/m_tilde/m_strong
│   ├── BaseModel.py, Training.py, metric.py, utils.py
│   ├── Runner.py                 # Không dùng (k-fold trên cửa sổ ⇒ rò rỉ người học)
│   └── data/assist09/            # .bin gốc — KHÔNG dùng (đồ thị dựng trên toàn bộ dữ liệu)
├── level1/                       # Package điều phối Cấp độ 1
│   ├── pipeline.py               # load_kt_data, save_split, train_one, evaluate, run_experiment
│   ├── explain.py                # collect_cases (5 đúng + 5 sai), format_cases
│   ├── dashboard.py              # chọn người học → xác suất, lịch sử, trạng thái tri thức
│   ├── paths.py                  # Quản lý đường dẫn tự động
│   └── gikt.py, gikt_data.py     # GIKT (đã tắt trong pipeline)
├── level2/                       # Package điều phối Cấp độ 2
│   ├── dataset_spec.py           # spec assist2009 / assist2009_smoke / junyi + chuẩn hoá CSV
│   ├── grkt_data.py              # chuỗi GRKT, đối chiếu Cấp độ 1, rel/pre từ train, biến thể đồ thị
│   ├── grkt_api.py               # make_args / build_grkt / train_grkt / evaluate_grkt
│   ├── baselines.py              # DKT / GKT trên cùng split (nạp checkpoint Cấp độ 1)
│   ├── trajectory.py             # m(u,c,t), heatmap/curves, tiêu chí thật + kiểm chứng cài đặt
│   ├── calibration.py            # ECE / ACE / Brier / reliability
│   └── experiment.py             # run_dataset / run_level2 (B1–B8)
└── outputs/level1/               # Artifact Cấp độ 1 đã commit
    ├── split.json                # Con trỏ Git LFS (xem README)
    ├── level1_metrics.csv, metrics.json, multiseed_metrics.csv
    ├── cases.md, learner_view.json
    └── *.pt                      # DKT, GKT-{Transition,Dense,Identity,Zero,Shuffled}

# outputs/level2/ chỉ sinh ra khi chạy notebook (không commit).
```

---

## 6. Kế hoạch thực hiện & Nhật ký tiến độ thực tế

Giả định nhóm ≤ 4 người, khoảng **8–10 tuần**.

### Giai đoạn 0 — Khởi động (tuần 1) — [ĐÃ HOÀN THÀNH]

- [x] Đọc DKT, GKT, GRKT (arxiv 2406.12896), pyKT.
- [x] Chốt split protocol theo người học (User-wise split) và cấu trúc thư mục repo.
- [x] Tải và chuẩn bị dữ liệu ASSISTments 2009 (`skill_builder_data.csv` và `assistment_test15.csv`); hoàn tất EDA (phân phối tương tác, tỷ lệ đúng/sai).
- [x] Thiết lập môi trường chạy trên Kaggle GPU (Tesla T4 / P100) và Git auto-sync.

### Giai đoạn 1 — Cấp độ 1 (tuần 2–4) → trần 7.0 — [ĐÃ HOÀN THÀNH]

- [x] **Pipeline chống rò rỉ dữ liệu:** Tạo split cố định theo user (`split.json`), sinh đồ thị Transition chỉ từ tập train (`train_dataset.indices`).
- [x] **Xây dựng đồ thị:** Transition Graph (xác suất chuyển tiếp KC), Dense Graph (liên thông đầy đủ).
- [x] **Baseline không đồ thị:** Huấn luyện và đánh giá DKT (LSTM) trên cùng tập split với GKT.
- [x] **Mô hình đồ thị chính:** Huấn luyện GKT-Transition trên Kaggle GPU, đạt hiệu năng vượt trội so với DKT.
- [x] **Khảo sát cấu trúc đồ thị & Ablation:** Huấn luyện GKT-Dense, GKT-Identity (ma trận đơn vị), GKT-Zero (ma trận không) để chứng minh đóng góp của cấu trúc đồ thị.
- [x] **Đánh giá đa seed (Multi-seed):** Module chạy 3 seed ngẫu nhiên (`[42, 123, 456]`) tính Mean ± Std cho DKT và GKT-Transition (`multiseed_metrics.csv`).
- [x] **Giải thích xAI cơ bản (Nhiệm vụ 3):** Trích xuất tự động 5 case đúng + 5 case sai kèm lịch sử tương tác và liên kết láng giềng đồ thị (`cases.md`).
- [x] **Trực quan hóa & Xuất Artifacts:** Biểu đồ Learning Curves (Loss & AUC theo epoch), biểu đồ cột so sánh Test AUC/Acc, xuất `level1_metrics.csv` và checkpoint `.pt`.

**Cột mốc hoàn thành:** Notebook [kaggle_level1.ipynb](file:///d:/workspace/Knowledge-Tracing/kaggle_level1.ipynb) và [kaggle_gkt.ipynb](file:///d:/workspace/Knowledge-Tracing/kaggle_gkt.ipynb) đã chạy trơn tru, xuất toàn bộ kết quả Cấp độ 1.

### Giai đoạn 2 — Cấp độ 2 (tuần 5–7) → trần 8.5

Mã nguồn trong package `level2/` và notebook `kaggle_level2.ipynb` đã viết xong; **số liệu
cuối cùng chỉ tồn tại sau khi chạy Kaggle** (máy dev không có file Junyi thật 3,0 GB và GPU;
kiểm chứng local chỉ dùng dữ liệu tổng hợp + `assist2009_smoke`).

- [x] Tái lập GRKT (code vendored trong `GRKT/`, đã package-hoá). **Map vào split của nhóm**:
      bỏ `GRKT/Runner.py` (k-fold trên cửa sổ 100 tương tác ⇒ cùng một user nằm ở cả train
      lẫn test), mỗi user là một chuỗi, split lấy thẳng `level1.pipeline.save_split` (seed 42).
- [x] Đánh giá GRKT trên **cả hai** dataset (`assist2009` + `junyi`), cùng `load_dataset`
      tiền xử lý, cùng chỉ số AUC/ACC/NLL và cùng mặt nạ với cấp 1 (`level2/experiment.py`).
      ASSISTments chạy toàn bộ 4.217 người học; Junyi chạy **toàn bộ người học** (72.630 người,
      sau lọc ≥10 tương tác còn ít hơn; tối đa ≈16,2 triệu lượt) với 200 bài tập phổ biến nhất
      — K=1.326 buộc batch
      còn 4–8 và >10 giờ/epoch trên T4 (xem `grkt_activation_gib`). Chế độ nhanh dùng tập
      `assist2009_smoke` (chính là `GKT/data/assistment_test15.csv` của cấp 1); `MAX_USERS`
      chỉ áp dụng cho tập phải chuẩn hoá nên smoke test trên ASSISTments không bị âm thầm chạy
      full 4.217 người học.
- [x] Cấu hình báo cáo đặt `MAX_SEQ_LEN = 200` **đúng bằng cấp 1** để checkpoint DKT/GKT nạp
      lại được eval trên đúng cửa sổ nó đã học; cell §4 do đó so được AUC đường nền với
      0.7866 (dung sai 0.005). Đường nền phải train lại (Junyi) dùng `BASELINE_EPOCHS = 20`
      theo công thức cấp 1 và ghi trọng số vào `<save_dir>/baselines/`, không đè `outputs/level1`.
- [x] Chạy dài hơn một phiên Kaggle 12 giờ: `train_grkt` ghi `ckpt_GRKT-<graph>.pt` mỗi
      `CKPT_EVERY_MIN=10` phút và cuối epoch (trọng số + Adam + epoch + batch đang dở + thứ tự
      batch + best state); đường nền phải train lại (Junyi) ghi `baselines/ckpt_<tên>.pt` cuối
      mỗi epoch; `TIME_BUDGET_MIN` tính một lần cho cả lần chạy và dừng chủ động bằng
      `TrainingPaused` trước khi phiên bị cắt; phiên sau attach zip phiên trước (private Kaggle
      Dataset) + `RESUME_ZIP` + `RESUME_GRKT_CKPT=True` để chạy tiếp đúng epoch/batch còn lại.
      Fingerprint trong checkpoint chặn resume sai split/đồ thị/siêu tham số hoặc `batch_size`
      (`level2/grkt_api.py`, `level2/baselines.py`, notebook §0.3 và §8.4).
- [x] Export mastery \(m_{u,c,t}\): `trajectories/traj_<graph>.npz` + `mastery_<graph>.csv`
      (dạng dài), heatmap khái niệm × thời điểm và đường cong mastery theo bước.
- [x] Metric quỹ đạo (định nghĩa trong `level2/trajectory.py`):
      `consistency_strengthen` (kiểm chứng cài đặt), `consistency_spread` (tiêu chí thật:
      hướng lan truyền sang láng giềng 1-hop), `locality` (độ cục bộ của cập nhật),
      `forget_slope` (Δm giữa hai bước theo Δt tại KC không bị chạm) + `stage_attribution`.
      Mặc định chỉ đo trên đồ thị `train` (`RUN_GRAPH_ABLATIONS=False`); bảng so sánh giữa 4
      biến thể `train/dense/zero/shuffled` bật lên khi cần cột so sánh chéo.
- [x] Phân tích nâng cao đã chọn cho cấp 2: **quỹ đạo mastery + ≥2 tiêu chí hợp lý** ở trên
      (đề bài chỉ yêu cầu chọn MỘT). ECE + ACE + Brier + reliability diagram cho DKT, GKT và
      GRKT vẫn có sẵn trong `level2/calibration.py`, chạy bằng `RUN_CALIBRATION=True`;
      đa seed riêng cho GRKT chạy bằng `RUN_MULTI_SEED=True` (mặc định tắt vì 3 seed của
      DKT/GKT đã có ở cấp 1 trên đúng split này).
- [ ] Điền số liệu vào báo cáo sau khi chạy `kaggle_level2.ipynb` (`QUICK_RUN=False`).

#### Trạng thái 30/10/2026 — sửa lỗi phát sinh khi chạy Junyi

Lần chạy Kaggle đầu tiên dừng ở bước chuẩn hoá với `KeyError`: `JUNYI.columns` không chứa tên
cột thật của `Log_Problem.csv` (file **không có** cột `exercise`). Đã sửa và kiểm chứng bằng
dữ liệu tổng hợp 14 cột + chạy end-to-end `assist2009_smoke`; **chưa chạy được dữ liệu thật
3,0 GB trên máy dev** (không có file/GPU), nên số liệu Cấp độ 2 vẫn chờ lần chạy Kaggle.
Các sửa đổi dưới đây **phải được commit + push** trước khi chạy, vì notebook clone từ GitHub.

| Lỗi / rủi ro | Xử lý | Kiểm chứng |
| --- | --- | --- |
| `KeyError` khi chuẩn hoá Junyi | `JUNYI.columns` dùng tên thật: `uuid` / `ucid` / `upid` / `is_correct` / `timestamp_TW` / `total_sec_taken`; khái niệm ← `ucid` | File synthetic 14 cột: PASS |
| `total_sec_taken` là **giây**, không phải ms | `_to_duration_ms` đổi đơn vị theo tên cột (×1000), không đoán theo median | `ms_first_response` của synthetic ≈ 30.000 ms |
| Đọc 3,0 GB / 16,2 triệu dòng | Đọc header trước + `usecols` 6 cột; quét THEO MẢNH và mã hoá id (`_CodeMap`) ngay khi quét, giải mã sau khi đã lọc người học | Log `ánh xạ cột: ...` trong cell §2 |
| `reliability_plot` crash dưới NumPy 2 (`markersize` nhận mảng) | Điểm vẽ bằng `ax.scatter(..., s=...)` | `run_dataset` end-to-end |
| Cảnh báo `max_users` sai cả khi CSV đã có sẵn | Chỉ cảnh báo khi `has_canonical(...)` đúng; CSV sinh ra có `.meta.json` để chặn dùng lại với cấu hình khác | `level2/experiment.py`, `level2/dataset_spec.py` |
| Notebook §2 không chỉ ra cột thật khi schema đổi | In header file dò được + liệt kê mọi CSV trong `/kaggle/input` | Notebook đã cập nhật |
| Junyi `Δt = 0` vì `timestamp_TW` làm tròn 15 phút | Ghi cảnh báo ở §5: `forget_slope` / `mean_abs_delta_learn_forget` = 0 là **giới hạn dữ liệu**, không phải lỗi mô hình | README + notebook §5 |
| Phiên Kaggle 12 giờ kết thúc khi GRKT mới train dở (bản cũ chỉ lưu `GRKT-*.pt` khi xong) | Thêm checkpoint giữa epoch (`ckpt_GRKT-*.pt`, 10 phút/lần, gồm Adam + batch đang dở), ngân sách `TIME_BUDGET_MIN` dừng chủ động, và đường resume hai phiên: §8.5 zip → upload private Dataset → §0.3 giải nén → `RESUME_ZIP` + `RESUME_GRKT_CKPT=True`. Fingerprint chặn resume khác cấu hình | `py_compile` + 32 cell notebook compile sạch; chưa chạy GPU/24 GB data trên máy dev |

#### Khai báo chỉnh sửa mã nguồn mở GRKT (bắt buộc cho mục "khai báo AI/mã nguồn mở")

Bản gốc: Cui et al., KDD 2024 — <https://github.com/JJCui96/GRKT>. Bảy chỉnh sửa dưới đây
đều có comment `# SỬA so với bản gốc` tại chỗ sửa:

| # | File | Sửa gì | Vì sao |
| --- | --- | --- | --- |
| 1 | `GRKT/__init__.py`, `GRKT/model/__init__.py`, `GRKT/{BaseModel,Runner,model/GRKT}.py` | Đóng gói + import tương đối | `GKT/` (cấp 1) cũng có `utils.py`; để hai package cùng trên `sys.path` thì import top-level nạp nhầm file của nhau |
| 2 | `GRKT/model/GRKT.py::forward` | 3 hook ghi `m_pre` / `m_tilde` / `m_strong` khi `feed_dict["rec"]` được cấp phát | Nhiệm vụ 2 cần mastery của **mọi** KC theo thời điểm; `rec=None` khi train ⇒ chi phí chỉ là một phép thử `is not None` |
| 3 | `GRKT/model/GRKT.py::forward` | Tách biến `master_full` trước khi `gather` | Giữ vector mastery đầy đủ `[B, NK]` sau giai đoạn retrieval (bản gốc chỉ giữ phần của KC đang làm) |
| 4 | `GRKT/Training.py` | `eval('{0}.{0}')` → `MODEL_REGISTRY`, thêm `ALLOWED_DATASETS` + dựng đường dẫn bằng `Path` | Bỏ thực thi chuỗi thành mã; chặn vô tình nạp `assist09.bin` (đồ thị dựng trên toàn bộ dữ liệu ⇒ rò rỉ) |
| 5 | *không sửa model* — `level2/grkt_api.py` | Vòng lặp epoch riêng: snapshot `state_dict()` về CPU, early stop theo val AUC toàn cục | `Runner.fit` nhân bản bằng `deepcopy(model)` (giữ tensor trên GPU ⇒ hết VRAM) và dùng k-fold |
| 6 | *không sửa model* — `level2/grkt_data.py` | `rel_map`/`pre_map` đếm **chỉ từ train**; `args.thresh=0.5` với ma trận đã nhị phân | Chống rò rỉ §6.2; nhánh `if thresh == 0` của upstream gán toàn ma trận = 1 kể cả hàng/cột padding |
| 7 | *không sửa model* — `level2/grkt_data.py` | Trục thời gian = phút cộng dồn **theo từng người học**, chặn 240 phút/bước, làm tròn SAU khi cộng dồn | `exp(-(n+1)·Δt·k)` với Δt thô (order_id, µs) underflow ⇒ mô hình xoá sạch trạng thái sau một bước. Tính theo từng user để không nhét khoảng cách giữa hai người học khác nhau vào delta đầu tiên; làm tròn sau cumsum vì thời gian làm bài < 1 phút, nếu làm tròn từng delta thì mọi Δt = 0 và giai đoạn học/quên không còn tác dụng |

**Bằng chứng cài đặt chạy được trên Kaggle** (không cần đọc log dài): với biến thể `zero`,
`beta_*_tilde = 0` nên `h_tilde = h` — do đó `m_tilde == m_pre` **chính xác** và cột
`mean_abs_delta_retrieval` của `stage_attribution` phải bằng 0. Cell §5.1 in rõ phép kiểm tra
này; nếu nó > 0 thì đồ thị rỗng chưa thật sự rỗng.

**Tương thích môi trường Kaggle** (image hiện là Python 3.13 + pandas 3.x + numpy 2.x): cell
cài đặt chỉ pip-install gói thật sự thiếu qua `importlib.util.find_spec`, nên không hạ cấp
pandas/numpy; mọi hình vẽ đi qua matplotlib thuần vì pandas 3 đã bỏ accessor `Series.hist` và
`Series.barh`; `matplotlib.use("Agg")` không được gọi giữa phiên (làm hỏng `plt.show()` nội
bảng). Junyi được đọc với `usecols` giới hạn trong `JUNYI.columns` vì file gốc hàng chục triệu
dòng, và timestamp được đổi qua `parsed - pd.Timestamp(0, tz="UTC")` thay cho
`astype("int64")` (cách cũ trả số rất âm trên pandas 2 và raise trên pandas 3 khi có NaT).

**Không đổi**: kiến trúc mô hình, công thức loss (BCE), `get_feed_dict`/`feature2tensor`
(đệm batch), định nghĩa AUC/ACC trong `GRKT/metric.py`. Hai tham số lệch mặc định upstream
được ghi rõ trong docstring `level2/grkt_api.make_args`: `thresh` 0.6 → 0.5, `alpha` giữ
0.01 nhưng đi cùng trục phút.

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

#### Đề cương báo cáo 12–18 trang (mỗi mục kèm nguồn số liệu/file đã có sẵn)

| § | Nội dung | Độ dài | Lấy từ đâu |
|---|---|---|---|
| 1 | Bài toán đúng/sai, vì sao cần đồ thị, phạm vi nhóm chọn (cấp 1 + 2) | 1 tr. | `YEU_CAU.md` §1–3 |
| 2 | Dữ liệu: ASSISTments 2009 (+ Junyi cho cấp 2), EDA số lượt/người học, phân bố khái niệm | 1.5 tr. | cell EDA hai notebook, `skill_builder_data.csv` |
| 3 | Tiền xử lý & xây đồ thị: `skill_with_answer = 2·skill + correct`, chuỗi theo người, Transition graph **chỉ từ train** (`GKT/processing.py:172-197`), `rel_map`/`pre_map` của GRKT (`level2/grkt_data.build_relevance_maps`) | 2 tr. | `README.md` §Định vị pyKT |
| 4 | Chống rò rỉ: split theo người seed 42, label shift-1, đặc trưng không dùng \(r_t\), một split cho mọi mô hình + mọi seed, `ALLOWED_DATASETS` chặn `assist09.bin` | 1.5 tr. | `level1/pipeline.py`, `level2/experiment.py` |
| 5 | Mô hình: DKT, GKT (công thức aggregate/update), GRKT (retrieval/strengthening/learning-forgetting) + bảng 7 chỗ đã sửa so với upstream | 3 tr. | `GKT/models.py:18-317`, notebook level2 §1, `KE_HOACH.md` mục khai báo |
| 6 | Kết quả Cấp độ 1: bảng 1 seed + bảng 3 seed mean±std; ablation 5 biến thể; **ba kết luận trung thực** (DKT thắng; +0.0089 ≈ 1.8σ; Shuffled ≈ Transition ⇒ cấu trúc cạnh không chứng minh được bằng AUC) | 2.5 tr. | `README.md` § *Kết quả Cấp độ 1* (đúng số đã commit) |
| 7 | Kết quả Cấp độ 2: bảng DKT/GKT-Transition/GKT-Zero/GRKT trên **2 tập** (`assist2009`, `junyi`) với cùng split và cùng chỉ số; số liệu đối chiếu AUC đường nền với Cấp độ 1; **tiêu chí thật** của quỹ đạo `precedence_gap`, `stability_spearman`/`stability_changed_frac`, `forget_slope` (ba cột `consistency_*`/`locality` chỉ là kiểm chứng cài đặt, ≡ 1 theo cấu trúc); heatmap + đường mastery. **Bật cho assist2009**: biến thể `train` vs `zero` và 3 seed; Junyi 1 biến thể + 1 seed (giới hạn phải ghi rõ). ECE/Brier vẫn là phần tuỳ chọn (`RUN_CALIBRATION=False`) | 3 tr. | `[CHỜ]` `outputs/level2/*` sau lần chạy `QUICK_RUN=False` |
| 8 | xAI: 2–3 case trong `cases.md` kèm số phản chứng (đổi kết quả 1 bước → Δp) và ablation `A:=0` từng case; dashboard `learner_view.json` | 1.5 tr. | `outputs/level1/cases.md` (bản chạy lại sau commit `cbf8ce5`) |
| 9 | Hạn chế & hướng cải tiến (học cấu trúc cạnh, time-decay, fidelity) — nối sang cấp 3/4 | 1 tr. | `KE_HOACH.md` §3.5–3.6 |
| 10 | Phụ lục: cách chạy lại, môi trường Kaggle, bảng phân công, khai báo AI/mã nguồn mở | — | `README.md`, mục khai báo ở §Giai đoạn 2 |

### Phân công gợi ý (4 người)

| Thành viên | Trọng tâm | Phải giải thích được khi vấn đáp |
| --- | --- | --- |
| A | Preprocess, split, đồ thị, chống leak | pyKT, vì sao graph chỉ từ train |
| B | DKT + GAT-KT + ablation | công thức dự đoán, chỗ không dùng \(r_t\) |
| C | GRKT + metric quỹ đạo | 3 giai đoạn, mastery |
| D | xAI, quiz, dashboard, báo cáo/slide | 10 case, quy trình thu thập |

Mọi người vẫn train/eval được mô hình của mình trên cùng data để tránh “chỉ một người hiểu code”.

---

## 7. Rủi ro và cách xử lý

| Rủi ro | Xử lý |
| --- | --- |
| GRKT khó chạy / OOM | Giảm max-seq, subsample ASSIST; fallback PSI-KT hoặc DGEKT |
| AUC GAT ≈ DKT | Kiểm tra leak; tăng chất lượng cạnh (Junyi pred); báo cáo trung thực + nhấn quỹ đạo/xAI |
| Quiz không đủ 400 tương tác | Thêm người / cho làm 2 phiên; không bịa nhãn |
| Tranh luận split pyKT vs paper GRKT | Giữ **một** split nhóm cho mọi so sánh chính; appendix chạy config gốc GRKT |
| Đồ thị ASSIST quá dày/nhiễu | Ngưỡng cạnh; so sánh vài cách tạo cạnh (mở rộng nhiệm vụ 1) |

---

## 8. Sản phẩm nộp (checklist đề)

- [x] **Mã nguồn Cấp độ 1:** pipeline preprocess, graph construction, train, eval, mô hình DKT & GKT (`GKT/`, `level1/`).
- [x] **Chống rò rỉ dữ liệu:** File `split.json` chia tập theo user; đồ thị chỉ xây từ tập train; tài liệu giải trình chống rò rỉ.
- [x] **Nhiệm vụ 3 mức cơ bản (5 đúng + 5 sai):** Báo cáo chi tiết 10 case study có bằng chứng lịch sử và láng giềng đồ thị (`cases.md`).
- [x] **Notebook thực nghiệm Cấp độ 1:** `kaggle_level1.ipynb` và `kaggle_gkt.ipynb` chạy hoàn tất trên Kaggle GPU.
- [ ] **Cấp độ 2 (Mastery Trajectory):** Tái lập GRKT, xuất vector thành thạo theo thời gian và đo 2 tiêu chí quỹ đạo. *(mã + notebook hoàn tất; số liệu và artifact chờ lần chạy `QUICK_RUN=False` trên Kaggle)*
- [ ] **Cấp độ 3 (xAI mở rộng & Quiz):** 10 case đồ thị nâng cao, quiz thu thập thực tế, giao diện dashboard.
- [ ] **Báo cáo & Thuyết minh:** Báo cáo 12–18 trang, slide trình bày và video demo ≤ 5 phút.

---

## 9. Tài liệu neo

- Đề: `Project02_Knowledge_Tracing.pdf`
- pyKT: <https://github.com/pykt-team/pykt-toolkit>
- DKT: Piech et al., NeurIPS 2015
- GKT: Nakagawa et al., WI 2019
- simpleKT: Liu et al., ICLR 2023
- **GRKT:** Cui et al., KDD 2024 — <https://arxiv.org/abs/2406.12896> — <https://github.com/JJCui96/GRKT>
- PSI-KT (dự phòng): ICLR 2024 — <https://github.com/mlcolab/psi-kt>

---

## 10. Kết luận ngắn

Làm **nhánh đúng/sai**, hai dataset **ASSIST09 + Junyi**, mô hình **DKT vs GKT (Cấp độ 1) và GRKT (Cấp độ 2)**, đồ thị KC xây đúng từ train, đo AUC kèm **ablation đồ thị (Dense, Identity, Zero)** và **đánh giá đa seed**, rồi mở rộng bằng **quỹ đạo mastery, 10 case xAI** và **quiz nhỏ**. Đường đi này khớp lần lượt cấp 1 → 2 → 3 của đề, tận dụng đúng điểm mạnh của học máy đồ thị: không chỉ đoán đúng/sai mà còn mô tả được năng lực người học và giải thích được nguyên nhân đưa ra dự đoán.

