# Báo cáo tổng hợp Cấp độ 1 — sinh tự động

> Sinh bởi `2026-11-08 level1-full+ablation-extend` lúc 2026-11-09T05:06:38Z. Số liệu lấy từ `tables/`, hình từ `figures/`.
> Chạy `QUICK_RUN=False` trên ASSISTments 2009 đầy đủ mới dùng cho báo cáo.

## 1. Metadata

- Git commit: `639c908478ca78a4c558a0b6696e679c46637ebe`
- Dữ liệu: `skill_builder_data.csv` | SHA-256 `f4b8791ae5968b884b77ab3b08adec48130503ec57f3113158c9a5af88af3374`
- Kích thước: 79.35 MB | 525,534 dòng | 4,217 người học | 123 khái niệm
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
| GKT-Transition | Transition | 0.7427 | 0.7012 | 0.5740 | 0.7529 | 34140 | 1378.9948 |
| GKT-Identity | Identity | 0.7549 | 0.7082 | 0.5570 | 0.7658 | 34140 | 2755.4948 |
| GKT-Zero | Zero | 0.7549 | 0.7082 | 0.5570 | 0.7658 | 34140 | 2912.3827 |
| GKT-Shuffled | Shuffled | 0.7542 | 0.6980 | 0.5675 | 0.7671 | 34140 | 3409.4123 |
| GKT-MHA | MHA | 0.7782 | 0.7306 | 0.5374 | 0.7836 | 38236 | 4072.8470 |
| GKT-ShuffledRow | ShuffledRow | 0.7651 | 0.7194 | 0.5462 | 0.7756 | 34140 | 3789.0394 |

## 5. Đa seed (mean ± std)

| Mô hình | Đồ thị | Số seed | AUC mean | AUC std | Acc mean | Acc std | NLL mean | NLL std |
|---|---|---|---|---|---|---|---|---|
| DKT | N/A | 5 | 0.7862 | 0.0006 | 0.7343 | 0.0013 | 0.5273 | 0.0005 |
| GKT-Transition | Transition | 5 | 0.7609 | 0.0096 | 0.7127 | 0.0077 | 0.5567 | 0.0105 |
| GKT-Identity | Identity | 5 | 0.7578 | 0.0056 | 0.7099 | 0.0077 | 0.5546 | 0.0068 |
| GKT-Zero | Zero | 5 | 0.7576 | 0.0055 | 0.7098 | 0.0077 | 0.5562 | 0.0073 |
| GKT-Shuffled | Shuffled | 5 | 0.7579 | 0.0039 | 0.7094 | 0.0064 | 0.5594 | 0.0063 |
| GKT-MHA | MHA | 5 | 0.7668 | 0.0095 | 0.7213 | 0.0087 | 0.5505 | 0.0088 |
| GKT-ShuffledRow | ShuffledRow | 5 | 0.7576 | 0.0127 | 0.7113 | 0.0121 | 0.5555 | 0.0119 |

## 6. Kiểm định ghép cặp theo seed

| comparison | metric | n_seeds | mean_a | mean_b | mean_diff | ci95_low | ci95_high | p | cohen_dz | significant_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| DKT − GKT-Transition | test_auc | 5 | 0.7862 | 0.7609 | 0.0254 | 0.0116 | 0.0391 | 0.0069 | 2.2851 | True |
| DKT − GKT-Transition | test_loss | 5 | 0.5273 | 0.5567 | -0.0294 | -0.0443 | -0.0145 | 0.0054 | -2.4493 | True |
| DKT − GKT-Identity | test_auc | 5 | 0.7862 | 0.7578 | 0.0284 | 0.0203 | 0.0364 | 0.0006 | 4.3857 | True |
| DKT − GKT-Identity | test_loss | 5 | 0.5273 | 0.5546 | -0.0273 | -0.0367 | -0.0180 | 0.0013 | -3.6221 | True |
| DKT − GKT-Zero | test_auc | 5 | 0.7862 | 0.7576 | 0.0286 | 0.0208 | 0.0364 | 0.0005 | 4.5686 | True |
| DKT − GKT-Zero | test_loss | 5 | 0.5273 | 0.5562 | -0.0290 | -0.0393 | -0.0187 | 0.0014 | -3.4946 | True |
| DKT − GKT-Shuffled | test_auc | 5 | 0.7862 | 0.7579 | 0.0283 | 0.0227 | 0.0340 | 0.0002 | 6.2222 | True |
| DKT − GKT-Shuffled | test_loss | 5 | 0.5273 | 0.5594 | -0.0322 | -0.0411 | -0.0232 | 0.0006 | -4.4437 | True |
| DKT − GKT-MHA | test_auc | 5 | 0.7862 | 0.7668 | 0.0194 | 0.0065 | 0.0324 | 0.0140 | 1.8644 | True |
| DKT − GKT-MHA | test_loss | 5 | 0.5273 | 0.5505 | -0.0232 | -0.0353 | -0.0111 | 0.0060 | -2.3755 | True |
| DKT − GKT-ShuffledRow | test_auc | 5 | 0.7862 | 0.7576 | 0.0286 | 0.0105 | 0.0467 | 0.0117 | 1.9654 | True |
| DKT − GKT-ShuffledRow | test_loss | 5 | 0.5273 | 0.5555 | -0.0282 | -0.0452 | -0.0113 | 0.0099 | -2.0679 | True |
| GKT-Transition − GKT-Zero | test_auc | 5 | 0.7609 | 0.7576 | 0.0033 | -0.0105 | 0.0170 | 0.5462 | 0.2945 | False |
| GKT-Transition − GKT-Zero | test_loss | 5 | 0.5567 | 0.5562 | 0.0004 | -0.0147 | 0.0155 | 0.9427 | 0.0342 | False |
| GKT-Transition − GKT-Identity | test_auc | 5 | 0.7609 | 0.7578 | 0.0030 | -0.0106 | 0.0166 | 0.5705 | 0.2760 | False |
| GKT-Transition − GKT-Identity | test_loss | 5 | 0.5567 | 0.5546 | 0.0020 | -0.0111 | 0.0152 | 0.6889 | 0.1926 | False |
| GKT-Transition − GKT-Shuffled | test_auc | 5 | 0.7609 | 0.7579 | 0.0030 | -0.0080 | 0.0139 | 0.4952 | 0.3352 | False |
| GKT-Transition − GKT-Shuffled | test_loss | 5 | 0.5567 | 0.5594 | -0.0028 | -0.0104 | 0.0048 | 0.3685 | -0.4529 | False |
| GKT-Transition − GKT-ShuffledRow | test_auc | 5 | 0.7609 | 0.7576 | 0.0033 | -0.0223 | 0.0289 | 0.7403 | 0.1589 | False |
| GKT-Transition − GKT-ShuffledRow | test_loss | 5 | 0.5567 | 0.5555 | 0.0012 | -0.0238 | 0.0261 | 0.9034 | 0.0578 | False |
| GKT-Shuffled − GKT-Zero | test_auc | 5 | 0.7579 | 0.7576 | 0.0003 | -0.0046 | 0.0052 | 0.8752 | 0.0749 | False |
| GKT-Shuffled − GKT-Zero | test_loss | 5 | 0.5594 | 0.5562 | 0.0032 | -0.0047 | 0.0110 | 0.3241 | 0.5024 | False |
| GKT-ShuffledRow − GKT-Zero | test_auc | 5 | 0.7576 | 0.7576 | -0.0000 | -0.0189 | 0.0188 | 0.9984 | -0.0009 | False |
| GKT-ShuffledRow − GKT-Zero | test_loss | 5 | 0.5555 | 0.5562 | -0.0007 | -0.0147 | 0.0132 | 0.8891 | -0.0664 | False |
| GKT-Identity − GKT-Zero | test_auc | 5 | 0.7578 | 0.7576 | 0.0002 | -0.0003 | 0.0008 | 0.3050 | 0.5257 | False |
| GKT-Identity − GKT-Zero | test_loss | 5 | 0.5546 | 0.5562 | -0.0016 | -0.0071 | 0.0038 | 0.4540 | -0.3705 | False |

### Kết luận từ số liệu

- Mô hình tốt nhất theo Test AUC trung bình: **DKT** (0.7862 ± 0.0006).
- DKT (không đồ thị): 0.7862 ± 0.0006; GKT-Transition: 0.7609 ± 0.0096.
- DKT − GKT-Transition: Δ=+0.0254 (CI95 [+0.0116, +0.0391], p=0.007).
- Transition − Zero: Δ=+0.0033 (CI95 [-0.0105, +0.0170], p=0.546) → chưa đủ bằng chứng.
- Transition − Shuffled: Δ=+0.0030 (CI95 [-0.0080, +0.0139], p=0.495) → chưa chứng minh được vai trò cấu trúc cạnh.
- Transition − ShuffledRow (giữ out-degree): Δ=+0.0033 (CI95 [-0.0223, +0.0289], p=0.740).
- Chỉ **100.0%** dự đoán test có khái niệm đích với ít nhất một cạnh trong đồ thị Transition train-only.
- GKT-Transition: bỏ đồ thị lúc suy luận ΔAUC=-0.0109, mean|Δp|=0.0259.
- GKT-Identity: bỏ đồ thị lúc suy luận ΔAUC=+0.0000, mean|Δp|=0.0000.
- GKT-Zero: bỏ đồ thị lúc suy luận ΔAUC=+0.0000, mean|Δp|=0.0000.
- GKT-Shuffled: bỏ đồ thị lúc suy luận ΔAUC=-0.0031, mean|Δp|=0.0075.
- GKT-MHA: bỏ đồ thị lúc suy luận ΔAUC=-0.0029, mean|Δp|=0.0026.
- GKT-ShuffledRow: bỏ đồ thị lúc suy luận ΔAUC=-0.0109, mean|Δp|=0.0248.

## 7. Hiệu chỉnh xác suất

| model | n | auc | nll | brier | ece | ace | mean_pred | observed_rate | over_prediction |
|---|---|---|---|---|---|---|---|---|---|
| DKT | 47358 | 0.7866 | 0.5249 | 0.1765 | 0.0154 | 0.0185 | 0.6348 | 0.6263 | 0.0085 |
| GKT-Transition | 47358 | 0.7427 | 0.5725 | 0.1951 | 0.0507 | 0.0592 | 0.6758 | 0.6263 | 0.0495 |
| GKT-Identity | 47358 | 0.7549 | 0.5546 | 0.1886 | 0.0223 | 0.0274 | 0.6103 | 0.6263 | -0.0160 |
| GKT-Zero | 47358 | 0.7549 | 0.5546 | 0.1886 | 0.0223 | 0.0274 | 0.6103 | 0.6263 | -0.0160 |
| GKT-Shuffled | 47358 | 0.7542 | 0.5652 | 0.1933 | 0.0706 | 0.0636 | 0.5559 | 0.6263 | -0.0704 |
| GKT-MHA | 47358 | 0.7454 | 0.5645 | 0.1927 | 0.0442 | 0.0323 | 0.6705 | 0.6263 | 0.0442 |
| GKT-ShuffledRow | 47358 | 0.7651 | 0.5440 | 0.1841 | 0.0105 | 0.0117 | 0.6189 | 0.6263 | -0.0074 |

## 8. Đồ thị đóng góp lúc suy luận (A := 0)



| model | graph_kind | auc_with_graph | auc_zero_graph | delta_auc | nll_with_graph | nll_zero_graph | delta_nll | mean_abs_dp | max_abs_dp |
|---|---|---|---|---|---|---|---|---|---|
| GKT-Transition | static A | 0.7427 | 0.7317 | -0.0109 | 0.5725 | 0.5805 | 0.0080 | 0.0259 | 0.4933 |
| GKT-Identity | static A | 0.7549 | 0.7549 | 0.0000 | 0.5546 | 0.5546 | 0.0000 | 0.0000 | 0.0000 |
| GKT-Zero | static A | 0.7549 | 0.7549 | 0.0000 | 0.5546 | 0.5546 | 0.0000 | 0.0000 | 0.0000 |
| GKT-Shuffled | static A | 0.7542 | 0.7511 | -0.0031 | 0.5652 | 0.5671 | 0.0019 | 0.0075 | 0.4913 |
| GKT-MHA | dynamic MHA | 0.7454 | 0.7425 | -0.0029 | 0.5645 | 0.5666 | 0.0021 | 0.0026 | 0.1806 |
| GKT-ShuffledRow | static A | 0.7651 | 0.7542 | -0.0109 | 0.5440 | 0.5520 | 0.0080 | 0.0248 | 0.6868 |

## 9. Case study (5 đúng + 5 sai)

| bucket | rank | model | user_id | t | skill_name | y_true | y_prob | y_pred | confidence | evidence_ok | max_flip_abs_dp | graph_dp | max_occ_abs_dp | target_state_norm | target_state_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correct | 1 | GKT-Transition | 78537 | 59 | 7:Venn Diagram | 1 | 0.9532 | 1 | 0.4532 | False | nan | nan | 0.5915 | 1.4235 | 6.0000 |
| correct | 2 | GKT-Transition | 88136 | 88 | 78:Multiplication and Division Positive Decimals | 1 | 0.7000 | 1 | 0.2000 | False | nan | nan | 0.6884 | 1.3618 | 5.0000 |
| correct | 3 | GKT-Transition | 89443 | 10 | 6:Table | 1 | 0.9000 | 1 | 0.4000 | False | nan | nan | 0.8333 | 1.0366 | 5.0000 |
| correct | 4 | GKT-Transition | 77949 | 14 | 30:Conversion of Fraction Decimals Percents | 0 | 0.4000 | 0 | 0.1000 | False | nan | nan | 0.4703 | 1.5489 | 3.0000 |
| correct | 5 | GKT-Transition | 96283 | 184 | 13:Probability of Two Distinct Events | 1 | 0.7999 | 1 | 0.2999 | False | nan | nan | 0.8531 | 1.8917 | 3.0000 |
| incorrect | 1 | GKT-Transition | 75169 | 139 | 7:Venn Diagram | 0 | 0.9531 | 1 | 0.4531 | False | nan | nan | 0.6268 | 1.4050 | 6.0000 |
| incorrect | 2 | GKT-Transition | 88478 | 51 | 30:Conversion of Fraction Decimals Percents | 0 | 0.7000 | 1 | 0.2000 | False | nan | nan | 1.0023 | 1.1358 | 11.0000 |
| incorrect | 3 | GKT-Transition | 83206 | 116 | 44:Percent Of | 0 | 0.9000 | 1 | 0.4000 | False | nan | nan | 0.3485 | 1.5984 | 7.0000 |
| incorrect | 4 | GKT-Transition | 84837 | 24 | 79:Addition and Subtraction Integers | 0 | 0.6000 | 1 | 0.1000 | False | nan | nan | 0.5757 | 1.7499 | 17.0000 |
| incorrect | 5 | GKT-Transition | 73962 | 42 | 9:Median | 0 | 0.8000 | 1 | 0.3000 | False | nan | nan | 0.7117 | 1.1601 | 6.0000 |

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
| §5 Cấp 1: baseline không đồ thị + mô hình đồ thị | DKT + GKT (7 cấu hình: DKT, GKT-Transition, GKT-Identity, GKT-Zero, GKT-Shuffled, GKT-MHA, GKT-ShuffledRow) — §3 |
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
| GAP P0.1 đa seed, split cố định | §3 chạy 7 cấu hình × 5 seed (đủ 8 khi `TRAIN_MODELS=None`) |
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
