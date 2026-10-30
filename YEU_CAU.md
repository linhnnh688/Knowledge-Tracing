# Yêu cầu đồ án — Project 02: Theo dõi tri thức người học bằng học máy trên đồ thị

> Viết lại từ đề cương `Project02_Knowledge_Tracing.pdf` bằng ngôn ngữ dễ hiểu.
> Đề gốc là văn bản pháp lý của đồ án; nếu có mâu thuẫn, đề gốc vẫn quyết định.

---

## 1. Bài toán là gì?

**Knowledge Tracing (KT)** = ước lượng trình độ của người học *theo thời gian*, dựa trên chuỗi làm bài đã qua.

Mỗi lần làm bài, mô hình nhìn vào các câu **đã làm trước đó** (câu hỏi + đúng/sai) rồi:

1. Đoán xem lần tới người học sẽ **đúng hay sai**.
2. Mô tả người học **đang giỏi/yếu khái niệm nào**, và mức đó thay đổi thế nào.
3. Giải thích **vì sao** mô hình lại kết luận như vậy (xAI).

Đồ án chia thành 3 nhiệm vụ nối tiếp nhau — đi từ *đoán* → *mô tả* → *giải thích*:

| # | Nhiệm vụ | Câu hỏi cần trả lời |
| --- | ---------- | ---------------------- |
| 1 | Dự đoán câu trả lời cho lần tương tác tiếp theo (xác suất đúng, hoặc phương án sẽ chọn) | Người học sẽ làm đúng hay sai câu hỏi tiếp theo? |
| 2 | Mô tả năng lực: mức nắm vững từng khái niệm và sự thay đổi theo thời gian | Người học đang nắm vững khái niệm nào, yếu khái niệm nào? |
| 3 | Chỉ ra bằng chứng và giải thích (xAI) cho hai nhiệm vụ trên: lần làm bài nào, khái niệm nào trên đồ thị dẫn tới kết luận đó | Vì sao mô hình dự đoán và mô tả như vậy? |

---

## 2. Vì sao phải dùng đồ thị?

Câu hỏi và khái niệm **không độc lập**:

- Một câu hỏi kiểm tra một hoặc nhiều khái niệm.
- Các khái niệm có quan hệ **tiên quyết** (phải học A trước khi học B) hoặc thường **xuất hiện cùng nhau**.

Cấu trúc này vẽ thành đồ thị rất tự nhiên, trong đó:

- **GNN** (GCN/GAT/GraphSAGE) lan truyền thông tin giữa các khái niệm liên quan — người học trả lời đúng một câu thì mô hình có cơ sở cập nhật cả khái niệm lân cận.
- **Mô hình thời gian** (GRU/LSTM/attention) cập nhật trạng thái người học qua từng lần làm bài.

**Quy trình tổng thể:**

```
lịch sử làm bài
      → Q-matrix + đồ thị khái niệm
      → GCN / GAT / GraphSAGE
      → mô hình thời gian (GRU/LSTM)
      → dự đoán câu trả lời tiếp theo
      → trạng thái tri thức + bằng chứng
```

Đây chỉ là cấu trúc *gợi ý*, nhóm được điều chỉnh cho phù hợp với mô hình của mình.

---

## 3. Chọn nhánh bài toán

| Nhánh | Nội dung | Mức |
|-------|----------|-----|
| **A. Đúng/sai** | Dự đoán xác suất người học trả lời đúng câu tiếp theo. Nhánh **chuẩn, dễ triển khai, bắt buộc bắt đầu từ đây**. | Cấp độ 1–3 |
| **B. Phương án trả lời** *(mở rộng)* | Dự đoán phương án người học sẽ chọn, có thể mở rộng sang phát hiện hiểu lầm. Vẫn phải báo cáo kết quả đúng/sai suy ra từ phương án. Cần dữ liệu có nội dung phương án (Eedi hoặc EdNet). | Mở rộng |

**Không bắt buộc** xây dựng hệ thống quản lý học tập hoàn chỉnh hay giao diện web — một notebook hoặc dashboard nhỏ chạy được và minh họa kết quả là đủ.

---

## 4. Dữ liệu và xây dựng đồ thị

### 4.1. Dữ liệu đầu vào (tối thiểu)

- Mã người học, mã câu hỏi, kết quả đúng/sai, thứ tự hoặc thời điểm làm bài.
- Mã khái niệm/kiến thức gắn với từng câu hỏi. Nếu một câu có nhiều khái niệm → xây **ma trận câu hỏi – khái niệm (Q-matrix)**.
- Các trường như thời gian trả lời, phương án đã chọn, độ khó câu hỏi: **chỉ dùng khi** tập dữ liệu có sẵn và phù hợp với nhánh đã chọn.
- Ngoài ra nhóm có thể triển khai các dạng input khác.

### 4.2. Xây đồ thị

- **Đỉnh:** các khái niệm kiến thức.
- **Cạnh:** quan hệ tiên quyết do dữ liệu cung cấp, *hoặc* quan hệ đồng xuất hiện/chuyển tiếp **tính từ tập huấn luyện**.
- Chuỗi tương tác của từng người học vẫn giữ theo thứ tự thời gian — cấp độ 1 **không** cần chuyển toàn bộ dữ liệu thành đồ thị động hay đồ thị dị thể.
- Đồ thị câu hỏi–khái niệm, đồ thị dị thể (đỉnh là phương án trả lời / hiểu lầm) là hướng mở rộng.

### 4.3. Kiến trúc cấp độ 1

Dùng **GCN, GAT hoặc GraphSAGE** tạo biểu diễn cho các khái niệm, kết hợp với kết quả đúng/sai rồi đưa qua **GRU/LSTM** để dự đoán câu trả lời tiếp theo. Có thể dùng **GKT** hoặc kiến trúc đồ thị tương đương **nếu giải thích được cơ chế hoạt động**.

---

## 5. Các cấp độ thực hiện và điểm trần

| Cấp độ | Nhiệm vụ | Yêu cầu cụ thể | Điểm trần |
| -------- | ---------- | ----------------- | ----------- |
| **1** | Dự đoán câu trả lời tiếp theo | • Dùng ít nhất **1 tập dữ liệu công khai**.<br>• Một **baseline không đồ thị** (khuyến nghị DKT hoặc simpleKT) **và** một mô hình dùng GCN/GAT/GraphSAGE hoặc GKT; dự đoán đúng/sai bước tiếp theo.<br>• **Ablation**: bỏ cạnh hoặc thay đồ thị bằng ma trận đơn vị để chứng minh vai trò của cấu trúc đồ thị.<br>• Giải thích mức cơ bản: phân tích **5 trường hợp đoán đúng + 5 trường hợp đoán sai**, chỉ ra bằng chứng trong lịch sử làm bài (các lần tương tác và khái niệm liên quan). | **7.0** |
| **2** | Mô tả năng lực người học | • Trực quan hóa trạng thái tri thức ước lượng theo khái niệm và theo thời gian.<br>• Định nghĩa và đo ít nhất **2 tiêu chí tính hợp lý của quỹ đạo tri thức**.<br>• Triển khai hoặc tái lập **1 phương pháp graph-aware knowledge tracing công bố từ 2024 trở lại đây**, đánh giá trên ít nhất **2 tập dữ liệu công khai** với cùng quy trình chia dữ liệu và chỉ số. Chỉ cần 1 phương pháp mới. | **8.5** |
| **3** | Bằng chứng / giải thích (xAI) | • Với mỗi dự đoán hoặc mô tả năng lực được phân tích, chỉ ra bằng chứng cụ thể: các lần làm bài trong lịch sử và các khái niệm/cạnh trên đồ thị ảnh hưởng đến kết luận — **tối thiểu 10 trường hợp**.<br>• Xây một **quiz nhỏ** và áp dụng phương pháp mới của cấp độ 2 lên dữ liệu **do nhóm tự thu thập**. Quy mô gợi ý: 10–20 người tham gia, 20–30 câu hỏi, 4–6 khái niệm, tối thiểu khoảng **400 tương tác**. Được phép huấn luyện trước trên dữ liệu công khai rồi tinh chỉnh hoặc đánh giá chuyển giao. | **10** |
| **4** | Mở rộng tự chọn | Đề xuất một cải tiến của riêng nhóm, theo một trong các hướng:<br>**(a)** cải tiến thuật toán: học cấu trúc cạnh, truyền thông tin suy giảm theo thời gian, mô hình option-aware, ước lượng độ bất định, ràng buộc tiên quyết;<br>**(b)** giải thích nâng cao: phân tích attention, GNNExplainer, phản chứng dạng nếu–thì, kiểm tra định lượng kiểu fidelity;<br>**(c)** chẩn đoán hiểu lầm theo thời gian và đề xuất can thiệp.<br>Áp dụng công trình mới năm 2026 cũng được tính điểm cộng.<br>**Bắt buộc:** có giả thuyết, ablation và thực nghiệm trên ít nhất 2 tập công khai. | **+1.0** |

---

## 6. Chia dữ liệu và chống rò rỉ (BẮT BUỘC)

1. Chỉ dùng các tương tác **trước** thời điểm cần dự đoán — **tuyệt đối không** dùng nhãn của câu hỏi hiện tại hoặc tương lai để tạo đặc trưng.
2. Nếu cạnh đồ thị suy ra từ tần suất đồng xuất hiện hoặc chuyển tiếp → đồ thị phải xây **từ tập huấn luyện**, không dùng toàn bộ dữ liệu.
3. Mọi mô hình phải dùng **cùng một tệp chia** train/validation/test. Nộp tệp chia hoặc seed + mã tạo split để tái lập được.
4. Đánh giá người học mới phải **tách theo người học** — không để tương tác của cùng một người xuất hiện đồng thời ở train và test.
5. Với dữ liệu tự thu thập: nên tách theo người học hoặc 5-fold cross-validation theo người học (do số mẫu nhỏ).
6. Bám quy trình chuẩn hóa của **pyKT** — pyKT đã chứng minh tiền xử lý/đánh giá sai có thể rò rỉ nhãn và làm kết quả tăng giả tạo.

---

## 7. Đánh giá kết quả

### Nhiệm vụ 1 — Dự đoán

| Bắt buộc | Mở rộng (tuỳ chọn) |
| ---------- | -------------------- |
| Đúng/sai: **AUC, accuracy, BCE/NLL** | Brier score hoặc ECE (calibration) |
| Nhánh B (phương án): top-1 accuracy và macro-F1 | Recall cho nhóm hiểu lầm hiếm |
| Độ ổn định: trung bình + độ lệch chuẩn của ít nhất **3 seed** | Đánh giá lịch sử ngắn hoặc người học mới |
| Vai trò đồ thị: ablation bỏ cạnh, xáo trộn cạnh hoặc đồ thị đơn vị | So sánh các cách tạo cạnh khác nhau |

### Nhiệm vụ 2 — Mô tả năng lực (cấp độ 2 trở lên)

| Bắt buộc | Mở rộng |
|----------|---------|
| Trực quan hóa mức nắm vững các khái niệm theo thời gian | Phân tích quỹ đạo tri thức của một số người học điển hình |
| Định nghĩa và đo ít nhất **2** trong các tiêu chí: **tính nhất quán về hướng** (trả lời đúng không làm giảm bất hợp lý mức nắm vững khái niệm liên quan), **tính cục bộ** (một lần trả lời không thay đổi quá lớn các khái niệm không liên quan), **tính nhất quán tiên quyết**, **tính ổn định với thay đổi nhỏ trong lịch sử** | So sánh quỹ đạo giữa mô hình có đồ thị và không có đồ thị |

### Nhiệm vụ 3 — Bằng chứng / giải thích

| Bắt buộc | Mở rộng |
|----------|---------|
| Phân tích tối thiểu **5 đúng + 5 sai**, kèm bằng chứng lịch sử và khái niệm liên quan (mọi nhóm) | Phân tích attention, GNNExplainer, phản chứng dạng nếu–thì |
| Cấp độ 3: bằng chứng cụ thể trên đồ thị cho tối thiểu **10 trường hợp** | Kiểm tra định lượng kiểu fidelity (điểm cộng) |

**Lưu ý ở cấp độ 2:** chỉ cần chọn thêm **một** phân tích nâng cao phù hợp với mô hình (calibration, người học lịch sử ngắn, hoặc một kiểm tra đơn giản về tính hợp lý của trạng thái tri thức). **Không** bắt buộc làm đồng thời dự đoán nhiều bước, cold-start, calibration và mọi tiêu chí quỹ đạo.

---

## 8. Sản phẩm cần nộp

- [ ] Mã nguồn tiền xử lý, tạo đồ thị, huấn luyện và đánh giá, kèm file `requirements`/environment và hướng dẫn chạy lại.
- [ ] Tập chia dữ liệu cố định hoặc mã tạo split, mô tả rõ cách phòng tránh rò rỉ dữ liệu.
- [ ] Báo cáo kỹ thuật khoảng **12–18 trang**, slide trình bày và bảng phân công đóng góp từng thành viên.
- [ ] Notebook hoặc dashboard nhỏ: chọn một người học → xem xác suất dự đoán, lịch sử làm bài, trạng thái tri thức ước lượng và bằng chứng cho dự đoán.
- [ ] Phân tích chi tiết tối thiểu **5 trường hợp đoán đúng + 5 trường hợp đoán sai**.
- [ ] Demo chạy trực tiếp khi vấn đáp, chuẩn bị **video dự phòng không quá 5 phút**.

---

## 9. Quy định

- Nhóm tối đa **4 thành viên**.
- Nộp bài trên Moodle. Nếu file lớn quá dung lượng cho phép → upload lên cloud, chia sẻ link và nộp link lên Moodle.
- Trình bày tối đa **20 phút** (slide + demo chạy trực tiếp), sau đó giảng viên vấn đáp **từng thành viên**.
- Giảng viên có thể hỏi bất kỳ thành viên nào về dữ liệu, đồ thị, mô hình, mã nguồn, kết quả và rò rỉ dữ liệu. **Điểm cá nhân có thể khác điểm chung**, dựa trên chất lượng trả lời vấn đáp và minh chứng đóng góp (lịch sử commit, bảng phân công trong báo cáo).
- Nhóm chuẩn bị môi trường để **chạy lại một thực nghiệm nhỏ** hoặc giải thích một đoạn mã bất kỳ tại buổi chấm.
- Danh sách dữ liệu và bài báo trong đề chỉ là gợi ý — nhóm được khuyến khích tự tìm phương án tốt hơn, nhưng cần **trao đổi với giảng viên trước khi thay đổi lớn**.
- Được dùng mã nguồn mở và công cụ AI hỗ trợ, nhưng **phải khai báo trong báo cáo**, ghi rõ phần đã chỉnh sửa và tự giải thích được toàn bộ nội dung khi vấn đáp.
- Mọi hành vi gian lận hoặc không chứng minh được kết quả do nhóm thực hiện đều bị xử lý theo quy định của môn học.

---

## 10. Dữ liệu gợi ý

> Thống kê chi tiết có thể thay đổi theo phiên bản — nhóm cần tự kiểm chứng khi tải dữ liệu.

| Tài nguyên | Vai trò trong đồ án |
| ------------ | --------------------- |
| ASSISTments 2009/2015/2017 | Dễ nhất cho nhánh đúng/sai; chỉ có nhãn đúng/sai; phù hợp cấp độ 1–2 |
| Junyi Academy | Có quan hệ khái niệm/tiên quyết; phù hợp để đánh giá vai trò của đồ thị |
| Eedi (NeurIPS 2020 Challenge) | Trắc nghiệm toán 4 lựa chọn kèm cây khái niệm; phù hợp nhánh dự đoán phương án |
| Eedi – Mining Misconceptions in Mathematics (Kaggle 2024) | Nhãn hiểu lầm gắn với từng phương án sai; chỉ dùng bổ sung cho nhánh chẩn đoán hiểu lầm |
| EdNet-KT1 | Dữ liệu lớn (~100 triệu tương tác) kèm phương án người học chọn; chỉ nên dùng một mẫu con có kiểm soát |
| XES3G5M (NeurIPS 2023) | Bộ KT kèm nội dung câu hỏi và quan hệ khái niệm |
| pyKT (pykt-toolkit) | Thư viện chuẩn hóa tiền xử lý, chia dữ liệu, benchmark và chạy baseline KT |

---

## 11. Bài báo và mô hình gợi ý (không giới hạn)

- **Baseline không đồ thị:** BKT, IRT; DKT (NeurIPS 2015); simpleKT (ICLR 2023); sparseKT (SIGIR 2023).
- **Mô hình đồ thị cho cấp độ 1:** GKT (IEEE/WIC/ACM WI 2019); GIKT (ECML-PKDD 2020).
- **Phương pháp mới cho cấp độ 2:**
  - DGEKT (ACM TOIS 2024)
  - GRKT (KDD 2024) — truy hồi, củng cố, học, quên trên đồ thị khái niệm
  - PSI-KT (ICLR 2024) — động lực học người học kết hợp đồ thị tiên quyết, chú trọng diễn giải
  - Efficient Knowledge Tracing Leveraging Higher-Order Information in Integrated Graphs (arXiv 2025)
- **Công cụ và định vị:** pyKT (NeurIPS 2022, Datasets and Benchmarks); khảo sát graph-based knowledge tracing (Journal of Educational Data Mining 2025) — dùng để định vị, **không** chọn làm mô hình.

---

## 12. Phụ lục — Đối chiếu yêu cầu với triển khai hiện tại (30/10/2026)

> Mục 1–11 giữ nguyên nội dung đề. Bảng dưới chỉ ghi **nơi đáp ứng** và **trạng thái thực tế**
> trong repo. Số liệu Cấp độ 2 chỉ hình thành sau lần chạy `kaggle_level2.ipynb` với
> `QUICK_RUN=False` trên Kaggle (máy dev không có file Junyi 3,0 GB và GPU).

| Yêu cầu | Triển khai | Trạng thái |
|---|---|---|
| §4.1 dữ liệu tối thiểu (người học, câu hỏi, KC, đúng/sai, thời điểm) | `GKT/processing.py`; CSV chuẩn hoá 8 cột trong `level2/dataset_spec.py` | Đã có; ASSISTments 2009 dùng ngay |
| §4.2 đồ thị chỉ xây từ tập train | GKT Transition chỉ từ `train_dataset`; `level2/grkt_data.py::build_relevance_maps` + `leak_report` | Đã có |
| §5 cấp 1: baseline + mô hình đồ thị + ablation | DKT, GKT-{Transition,Dense,Identity,Zero,Shuffled} (`GKT/`, `level1/pipeline.py`) | Đã chạy — `outputs/level1/level1_metrics.csv` |
| §5 cấp 2: phương pháp graph-aware từ 2024 | GRKT (KDD 2024) vendored, huấn luyện qua `level2/grkt_api.py` | Đã có mã; số liệu chờ chạy |
| §5 cấp 2: ≥ 2 tập công khai, cùng quy trình và chỉ số | `assist2009` + `junyi`, cùng `load_kt_data`, cùng split seed 42, cùng AUC/ACC/NLL | Đã có mã; số liệu chờ chạy |
| §7 trực quan mastery theo KC × thời gian | `level2/trajectory.py`: `traj_*.npz`, `mastery_*.csv`, heatmap, đường cong | Đã có |
| §7 ≥ 2 tiêu chí hợp lý của quỹ đạo | 4 tiêu chí: `consistency_strengthen`, `consistency_spread`, `locality`, `forget_slope` | Đã có |
| §6 chống rò rỉ | split seed 42 tái sinh, `_assert_align` từng người học, đồ thị train-only, nhãn dịch 1 bước | Đã có; `leak_report.txt` |
| §7 AUC/ACC/NLL + ≥ 3 seed | `level2_metrics.csv`, `multiseed_metrics.csv` | Đã có; số liệu chờ chạy |
| §7 phân tích nâng cao (tuỳ chọn) | ECE / ACE / Brier / reliability (`level2/calibration.py`) | Đã có |
| §5 cấp 3: xAI 10 case + quiz | Kế hoạch ở Giai đoạn 3 của `KE_HOACH.md` | Chưa làm |

**Lưu ý khi đọc kết quả Junyi:** `timestamp_TW` được làm tròn 15 phút ⇒ Δt = 0 trong một
phiên ⇒ `forget_slope` và `mean_abs_delta_learn_forget` bằng 0 là **giới hạn dữ liệu**; phải
ghi rõ trong báo cáo thay vì trình bày như kết quả của mô hình.
