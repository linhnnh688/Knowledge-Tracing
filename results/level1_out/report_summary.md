# Báo cáo tổng hợp Cấp độ 1 — sinh tự động

> Sinh bởi `2026-11-07 level1-full+resume+select+evalonly` lúc 2026-11-07T08:23:32Z. Số liệu lấy từ `tables/`, hình từ `figures/`.
> Chạy `QUICK_RUN=False` trên ASSISTments 2009 đầy đủ mới dùng cho báo cáo.

## 1. Metadata

- Git commit: `2e8110a4c006304fba44b595a57e0a078dc3b718`
- Dữ liệu: `skill_builder_data.csv` | SHA-256 `f4b8791ae5968b884b77ab3b08adec48130503ec57f3113158c9a5af88af3374`
- Kích thước: 78.85 MB | 525,534 dòng | 4,217 người học | 123 khái niệm
- Cấu hình: epochs=20, batch=32, max_seq_len=200, seeds=[42, 123, 456, 789, 1011]
- Thiết bị: `cuda`

## 2. Chia tập (theo người học, seed=42)

| Tập | Số người học | Số tương tác (sau cắt) |
|---|---|---|
| train | 2330 | 142299 |
| val | 776 | 47508 |
| test | 778 | 48136 |

## 3. Đồ thị

| Đồ thị | Số đỉnh | Số cạnh khác 0 | Mật độ | Số hàng có cạnh | Số cột có cạnh | Tổng trọng số hàng TB | Trọng số lớn nhất | Đường chéo khác 0 |
|---|---|---|---|---|---|---|---|---|
| Transition | 123 | 884 | 0.0589 | 109 | 114 | 0.8862 | 1.0000 | 0 |
| Dense | 123 | 15006 | 1.0000 | 123 | 123 | 1.0000 | 0.0082 | 0 |
| Identity | 123 | 123 | 0.0082 | 123 | 123 | 1.0000 | 1.0000 | 123 |
| Zero | 123 | 0 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 |
| Shuffled | 123 | 814 | 0.0542 | 109 | 114 | 0.8862 | 1.0000 | 0 |
| ShuffledRow | 123 | 884 | 0.0589 | 109 | 114 | 0.8862 | 1.0000 | 0 |

Độ phủ đồ thị Transition theo dự đoán test:

| Tiêu chí | Số dự đoán test | Số dự đoán được phủ | Tỉ lệ |
|---|---|---|---|
| Có cạnh đi ra (out-degree > 0) | 47358 | 47117 | 0.9949 |
| Có cạnh đi vào (in-degree > 0) | 47358 | 46744 | 0.9870 |
| Có ít nhất một cạnh (out hoặc in) | 47358 | 47341 | 0.9996 |

## 4. Kết quả chính (seed 42)

| Mô hình | Loại đồ thị | Test AUC | Test Accuracy | Test NLL | Best Val AUC | Số tham số | Thời gian (s) |
|---|---|---|---|---|---|---|---|
| DKT | N/A | 0.7866 | 0.7340 | 0.5272 | 0.7922 | 39899 | 35.1091 |
| GKT-MHA | MHA | 0.7782 | 0.7306 | 0.5374 | 0.7836 | 38236 | 4072.8470 |

## 5. Đa seed (mean ± std)

| Mô hình | Đồ thị | Số seed | AUC mean | AUC std | Acc mean | Acc std | NLL mean | NLL std |
|---|---|---|---|---|---|---|---|---|
| DKT | N/A | 5 | 0.7862 | 0.0006 | 0.7343 | 0.0013 | 0.5273 | 0.0005 |
| GKT-MHA | MHA | 5 | 0.7668 | 0.0095 | 0.7213 | 0.0087 | 0.5505 | 0.0088 |

## 6. Kiểm định ghép cặp theo seed

| comparison | metric | n_seeds | mean_a | mean_b | mean_diff | ci95_low | ci95_high | p | cohen_dz | significant_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| DKT − GKT-MHA | test_auc | 5 | 0.7862 | 0.7668 | 0.0194 | 0.0065 | 0.0324 | 0.0140 | 1.8644 | True |
| DKT − GKT-MHA | test_loss | 5 | 0.5273 | 0.5505 | -0.0232 | -0.0353 | -0.0111 | 0.0060 | -2.3755 | True |

### Kết luận từ số liệu

- Mô hình tốt nhất theo Test AUC trung bình: **DKT** (0.7862 ± 0.0006).
- DKT (không đồ thị): 0.7862 ± 0.0006; GKT-MHA: 0.7668 ± 0.0095.
- DKT − GKT-MHA: Δ=+0.0194 (CI95 [+0.0065, +0.0324], p=0.014).
- Chỉ **100.0%** dự đoán test có khái niệm đích với ít nhất một cạnh trong đồ thị Transition train-only.
- Không có mô hình đồ thị tĩnh trong lần chạy này nên không chạy ablation A := 0; GKT-MHA suy luận đồ thị động trong lúc chạy.

## 7. Hiệu chỉnh xác suất

| model | n | auc | nll | brier | ece | ace | mean_pred | observed_rate | over_prediction |
|---|---|---|---|---|---|---|---|---|---|
| DKT | 47358 | 0.7866 | 0.5249 | 0.1765 | 0.0154 | 0.0185 | 0.6348 | 0.6263 | 0.0085 |
| GKT-MHA | 47358 | 0.7454 | 0.5645 | 0.1927 | 0.0442 | 0.0323 | 0.6705 | 0.6263 | 0.0442 |

## 8. Đồ thị đóng góp lúc suy luận (A := 0)

> Lần chạy này không có mô hình đồ thị tĩnh (không bật GKT-Transition/Dense/Zero/...), nên không áp dụng ablation `A := 0`; GKT-MHA suy luận đồ thị động trong lúc chạy.

| model | auc_with_graph | auc_zero_graph | delta_auc | nll_with_graph | nll_zero_graph | delta_nll | mean_abs_dp | max_abs_dp |
|---|---|---|---|---|---|---|---|---|

## 9. Case study (5 đúng + 5 sai)

| bucket | rank | model | user_id | t | skill_name | y_true | y_prob | y_pred | confidence | evidence_ok | max_flip_abs_dp | graph_dp | max_occ_abs_dp | target_state_norm | target_state_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correct | 1 | GKT-MHA | 75316 | 199 | 90:Area Triangle | 1 | 0.9826 | 1 | 0.4826 | False | nan | nan | 1.1730 | 2.4870 | 1.0000 |
| correct | 2 | GKT-MHA | 96270 | 173 | 20:Pythagorean Theorem | 1 | 0.7000 | 1 | 0.2000 | True | 0.0013 | nan | 0.1474 | 1.1779 | 83.0000 |
| correct | 3 | GKT-MHA | 80641 | 33 | 81:Multiplication and Division Integers | 1 | 0.9000 | 1 | 0.4000 | False | nan | nan | 0.1847 | 1.8837 | 1.0000 |
| correct | 4 | GKT-MHA | 83954 | 57 | 45:Subtraction Whole Numbers | 1 | 0.6000 | 1 | 0.1000 | False | nan | nan | 0.3785 | 1.5833 | 2.0000 |
| correct | 5 | GKT-MHA | 88744 | 72 | 55:Absolute Value | 1 | 0.8000 | 1 | 0.3000 | False | nan | nan | 0.0951 | 1.4962 | 1.0000 |
| incorrect | 1 | GKT-MHA | 78591 | 104 | 44:Percent Of | 0 | 0.9811 | 1 | 0.4811 | False | nan | nan | 1.0335 | 2.9565 | 1.0000 |
| incorrect | 2 | GKT-MHA | 78171 | 91 | 98:Equation Solving Two or Fewer Steps | 0 | 0.7000 | 1 | 0.2000 | True | 0.0003 | nan | 0.1442 | 1.1805 | 75.0000 |
| incorrect | 3 | GKT-MHA | 78154 | 72 | 56:Exponents | 0 | 0.8999 | 1 | 0.3999 | False | nan | nan | 0.4021 | 1.3159 | 15.0000 |
| incorrect | 4 | GKT-MHA | 83322 | 73 | 79:Addition and Subtraction Integers | 0 | 0.6000 | 1 | 0.1000 | False | nan | nan | 0.4825 | 1.9376 | 1.0000 |
| incorrect | 5 | GKT-MHA | 88136 | 126 | 80:Addition and Subtraction Positive Decimals | 0 | 0.7998 | 1 | 0.2998 | False | nan | nan | 1.1271 | 1.2866 | 20.0000 |

Chi tiết đầy đủ: `cases.md` và `cases.json`.

## 10. Hình cần dùng trong báo cáo

| Hình | Tệp |
|---|---|
| Phân phối độ dài chuỗi | `figures/eda_sequence_length.png` |
| Learning curves | `figures/learning_curves.png` |
| So sánh AUC/NLL (mean ± std) | `figures/model_comparison.png` |
| Reliability diagram | `figures/calibration_reliability.png` |
| Đồ thị đóng góp lúc suy luận (chỉ khi bật GKT tĩnh) | `figures/graph_inference_ablation.png` |
| Độ phủ đồ thị | `figures/graph_coverage.png` |
| Bằng chứng 10 case | `figures/case_evidence.png` |
| Một người học trong dashboard | `figures/learner_*.png` |

## 11. Đối chiếu yêu cầu


| Yêu cầu / GAP | Nơi đáp ứng trong notebook này |
|---|---|
| §5 Cấp 1: baseline không đồ thị + mô hình đồ thị | DKT + GKT (2 cấu hình: DKT, GKT-MHA) — §3 |
| §5 Cấp 1: ablation bỏ cạnh / ma trận đơn vị | Dense, Identity, Zero, Shuffled, ShuffledRow (bật bằng `TRAIN_MODELS`); MHA suy luận đồ thị động — §2, §4 |
| §6.2 đồ thị chỉ xây từ train | `leak_report.txt` đối chiếu support + khối lượng test-only — §2.2 |
| §6.3 cùng một split + seed tái lập | `split.json`, assert split trùng trên mọi bundle — §2, §2.1 |
| §6.4 tách theo người học | assert ba tập rời nhau — §2 |
| §7 AUC / Accuracy / NLL | bảng chính + đa seed — §4 |
| §7 mean ± std ≥ 3 seed | 5 seed, split cố định — §3, §4 |
| §7 vai trò đồ thị | paired t-test + bootstrap CI + `A:=0` lúc suy luận + độ phủ — §4, §6 |
| §7 mở rộng calibration | Brier / ECE / ACE / reliability — §5 |
| §8 dashboard chọn người học | `learner_view.json`, bảng tóm tắt, hình, dropdown — §8 |
| §8 5 đúng + 5 sai kèm bằng chứng | `cases.md` + `cases.json` + hình — §7 |
| GAP P0.1 đa seed, split cố định | §3 chạy 2 cấu hình × 5 seed (đủ 8 khi `TRAIN_MODELS=None`) |
| GAP P1.6 bằng chứng case | phản chứng, ablation đồ thị, occlusion, trạng thái `h_t` — §7 |
| GAP P1.8 Identity ≡ Zero, edge-shuffle | giải thích trong `README.md`; thêm `ShuffledRow` giữ out-degree — §2.1 |
| GAP E3 chưa chứng minh cấu trúc đồ thị | báo cáo trung thực bằng kiểm định + ablation suy luận — §4, §6 |
| GAP P0.3/P2.11 sản phẩm báo cáo | `report_summary.md` + 2 zip — §9, §10 |

**Còn phải làm thủ công:** viết báo cáo 12–18 trang và slide từ `report_summary.md`; Cấp độ 3
(quiz tự thu thập, 10 case đồ thị nâng cao) nằm ngoài phạm vi Cấp độ 1.


## 12. Lưu ý trung thực

- ASSISTments 2009 có **một khái niệm mỗi câu hỏi**, nên đồ thị Transition thưa; phần lớn dự đoán
  test có thể không có cạnh. Khi đó GKT gần với self-update và không khác Zero graph về bản chất.
- Nếu Shuffled ≈ Transition (khi bật biến thể đó trong `TRAIN_MODELS`), kết luận đúng là **chưa
  chứng minh được vai trò cấu trúc cạnh** ở Cấp độ 1; lợi ích (nếu có) đến từ mật độ hàng xóm,
  không từ hình dạng đồ thị.
- Norm `h_t` là proxy trạng thái ẩn, **không phải mastery đã hiệu chỉnh**. Không dùng norm để kết
  luận người học "giỏi/yếu" tuyệt đối.
- `report_summary.md` chỉ chứa số liệu và diễn giải tự động; phần thảo luận, so sánh với bài báo
  và slide vẫn phải viết thủ công.
