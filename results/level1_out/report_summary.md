# Báo cáo tổng hợp Cấp độ 1 — sinh tự động

> Sinh bởi `2026-11-09 level1-full+order-dedup+bn-eval` lúc 2026-11-11T00:34:41Z. Số liệu lấy từ `tables/`, hình từ `figures/`.
> Chạy `QUICK_RUN=False` trên ASSISTments 2009 đầy đủ mới dùng cho báo cáo.

## 1. Metadata

- Git commit: `2908d1150991dce4fd39dc1a0626ee56dbd06ea4`
- Dữ liệu: `skill_builder_data.csv` | SHA-256 `f4b8791ae5968b884b77ab3b08adec48130503ec57f3113158c9a5af88af3374`
- Kích thước: 79.35 MB | 525,534 dòng | 4,217 người học | 123 khái niệm
- Cấu hình: epochs=20, batch=32, max_seq_len=200, seeds=[42, 123, 456]
- Thiết bị: `cuda`

## 2. Chia tập (theo người học, seed=42)

| Tập | Số người học | Số tương tác (sau cắt) |
|---|---|---|
| train | 2311 | 116964 |
| val | 770 | 41224 |
| test | 771 | 40156 |

## 3. Đồ thị

| Đồ thị | Số đỉnh | Số cạnh khác 0 | Mật độ | Số hàng có cạnh | Số cột có cạnh | Tổng trọng số hàng TB | Trọng số lớn nhất | Đường chéo khác 0 |
|---|---|---|---|---|---|---|---|---|
| Transition | 112 | 2665 | 0.2144 | 108 | 110 | 0.9643 | 1.0000 | 0 |
| Dense | 112 | 12432 | 1.0000 | 112 | 112 | 1.0000 | 0.0090 | 0 |
| Identity | 112 | 112 | 0.0090 | 112 | 112 | 1.0000 | 1.0000 | 112 |
| Zero | 112 | 0 | 0.0000 | 0 | 0 | 0.0000 | 0.0000 | 0 |
| Shuffled | 112 | 2146 | 0.1726 | 108 | 110 | 0.9643 | 1.0000 | 0 |
| ShuffledRow | 112 | 2665 | 0.2144 | 108 | 110 | 0.9643 | 1.0000 | 0 |

Độ phủ đồ thị Transition theo dự đoán test:

| Tiêu chí | Số dự đoán test | Số dự đoán được phủ | Tỉ lệ |
|---|---|---|---|
| Có cạnh đi ra (out-degree > 0) | 39385 | 39376 | 0.9998 |
| Có cạnh đi vào (in-degree > 0) | 39385 | 39384 | 1.0000 |
| Có ít nhất một cạnh (out hoặc in) | 39385 | 39384 | 1.0000 |

## 4. Kết quả chính (seed 42)

| Mô hình | Loại đồ thị | Test AUC | Test Accuracy | Test NLL | Best Val AUC | Số tham số | Thời gian (s) |
|---|---|---|---|---|---|---|---|
| DKT | N/A | 0.7614 | 0.7224 | 0.5495 | 0.7537 | 36720 | 39.0173 |
| GKT-Transition | Transition | 0.7346 | 0.7071 | 0.5717 | 0.7251 | 33073 | 2667.8028 |
| GKT-Zero | Zero | 0.7152 | 0.6931 | 0.5845 | 0.7096 | 33073 | 2664.5647 |
| GKT-Shuffled | Shuffled | 0.7338 | 0.7096 | 0.5728 | 0.7276 | 33073 | 3546.7009 |

## 5. Đa seed (mean ± std)

| Mô hình | Đồ thị | Số seed | AUC mean | AUC std | Acc mean | Acc std | NLL mean | NLL std |
|---|---|---|---|---|---|---|---|---|
| DKT | N/A | 3 | 0.7608 | 0.0005 | 0.7228 | 0.0007 | 0.5500 | 0.0004 |
| GKT-Transition | Transition | 3 | 0.7358 | 0.0010 | 0.7081 | 0.0009 | 0.5720 | 0.0002 |
| GKT-Zero | Zero | 3 | 0.7183 | 0.0027 | 0.6952 | 0.0037 | 0.5833 | 0.0027 |
| GKT-Shuffled | Shuffled | 3 | 0.7290 | 0.0045 | 0.7069 | 0.0025 | 0.5764 | 0.0030 |

## 6. Kiểm định ghép cặp theo seed

| comparison | metric | n_seeds | mean_a | mean_b | mean_diff | ci95_low | ci95_high | p | cohen_dz | significant_0.05 |
|---|---|---|---|---|---|---|---|---|---|---|
| DKT − GKT-Transition | test_auc | 3 | 0.7608 | 0.7358 | 0.0250 | 0.0212 | 0.0288 | 0.0012 | 16.4682 | True |
| DKT − GKT-Transition | test_loss | 3 | 0.5500 | 0.5720 | -0.0220 | -0.0226 | -0.0215 | 0.0000 | -102.4055 | True |
| DKT − GKT-Zero | test_auc | 3 | 0.7608 | 0.7183 | 0.0425 | 0.0328 | 0.0523 | 0.0028 | 10.8501 | True |
| DKT − GKT-Zero | test_loss | 3 | 0.5500 | 0.5833 | -0.0334 | -0.0427 | -0.0241 | 0.0041 | -8.9474 | True |
| DKT − GKT-Shuffled | test_auc | 3 | 0.7608 | 0.7290 | 0.0318 | 0.0194 | 0.0441 | 0.0080 | 6.3969 | True |
| DKT − GKT-Shuffled | test_loss | 3 | 0.5500 | 0.5764 | -0.0265 | -0.0343 | -0.0186 | 0.0047 | -8.3641 | True |
| GKT-Transition − GKT-Zero | test_auc | 3 | 0.7358 | 0.7183 | 0.0175 | 0.0101 | 0.0249 | 0.0095 | 5.8913 | True |
| GKT-Transition − GKT-Zero | test_loss | 3 | 0.5720 | 0.5833 | -0.0114 | -0.0202 | -0.0026 | 0.0309 | -3.2104 | True |
| GKT-Transition − GKT-Shuffled | test_auc | 3 | 0.7358 | 0.7290 | 0.0068 | -0.0083 | 0.0219 | 0.1930 | 1.1159 | False |
| GKT-Transition − GKT-Shuffled | test_loss | 3 | 0.5720 | 0.5764 | -0.0045 | -0.0128 | 0.0039 | 0.1496 | -1.3196 | False |
| GKT-Shuffled − GKT-Zero | test_auc | 3 | 0.7290 | 0.7183 | 0.0108 | -0.0113 | 0.0328 | 0.1705 | 1.2128 | False |
| GKT-Shuffled − GKT-Zero | test_loss | 3 | 0.5764 | 0.5833 | -0.0069 | -0.0230 | 0.0092 | 0.2061 | -1.0661 | False |

### Kết luận từ số liệu

- Mô hình tốt nhất theo Test AUC trung bình: **DKT** (0.7608 ± 0.0005).
- DKT (không đồ thị): 0.7608 ± 0.0005; GKT-Transition: 0.7358 ± 0.0010.
- DKT − GKT-Transition: Δ=+0.0250 (CI95 [+0.0212, +0.0288], p=0.001).
- Transition − Zero: Δ=+0.0175 (CI95 [+0.0101, +0.0249], p=0.009) → có ý nghĩa thống kê.
- Transition − Shuffled: Δ=+0.0068 (CI95 [-0.0083, +0.0219], p=0.193) → chưa chứng minh được vai trò cấu trúc cạnh.
- Chỉ **100.0%** dự đoán test có khái niệm đích với ít nhất một cạnh trong đồ thị Transition train-only.
- GKT-Transition: bỏ đồ thị lúc suy luận ΔAUC=-0.0291, mean|Δp|=0.0551.
- GKT-Zero: bỏ đồ thị lúc suy luận ΔAUC=+0.0000, mean|Δp|=0.0000.
- GKT-Shuffled: bỏ đồ thị lúc suy luận ΔAUC=-0.0223, mean|Δp|=0.0444.

## 7. Hiệu chỉnh xác suất

| model | n | auc | nll | brier | ece | ace | mean_pred | observed_rate | over_prediction |
|---|---|---|---|---|---|---|---|---|---|
| DKT | 39385 | 0.7614 | 0.5504 | 0.1851 | 0.0111 | 0.0176 | 0.6329 | 0.6256 | 0.0073 |
| GKT-Transition | 39385 | 0.7346 | 0.5728 | 0.1942 | 0.0242 | 0.0224 | 0.6490 | 0.6256 | 0.0234 |
| GKT-Zero | 39385 | 0.7152 | 0.5855 | 0.2001 | 0.0090 | 0.0193 | 0.6197 | 0.6256 | -0.0059 |
| GKT-Shuffled | 39385 | 0.7338 | 0.5744 | 0.1948 | 0.0330 | 0.0266 | 0.6584 | 0.6256 | 0.0328 |

## 8. Đồ thị đóng góp lúc suy luận (A := 0)



| model | graph_kind | auc_with_graph | auc_zero_graph | delta_auc | nll_with_graph | nll_zero_graph | delta_nll | mean_abs_dp | max_abs_dp |
|---|---|---|---|---|---|---|---|---|---|
| GKT-Transition | static A | 0.7346 | 0.7056 | -0.0291 | 0.5728 | 0.5960 | 0.0231 | 0.0551 | 0.6118 |
| GKT-Zero | static A | 0.7152 | 0.7152 | 0.0000 | 0.5855 | 0.5855 | 0.0000 | 0.0000 | 0.0000 |
| GKT-Shuffled | static A | 0.7338 | 0.7115 | -0.0223 | 0.5744 | 0.5896 | 0.0152 | 0.0444 | 0.6222 |

## 9. Case study (5 đúng + 5 sai)

| bucket | rank | model | user_id | t | skill_name | y_true | y_prob | y_pred | confidence | evidence_ok | max_flip_abs_dp | graph_dp | max_occ_abs_dp | target_state_norm | target_state_rank |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| correct | 1 | GKT-Transition | 82163 | 63 | 34:Addition Whole Numbers | 1 | 0.9674 | 1 | 0.4674 | True | 0.0650 | 0.1225 | 2.2129 | 4.5018 | 1.0000 |
| correct | 2 | GKT-Transition | 96270 | 57 | 6:Table | 1 | 0.7000 | 1 | 0.2000 | True | 0.3449 | 0.0602 | 1.0062 | 2.6018 | 2.0000 |
| correct | 3 | GKT-Transition | 88479 | 182 | 72:Multiplication and Division Integers | 0 | 0.1000 | 0 | 0.4000 | True | 0.1776 | -0.0795 | 2.2954 | 3.4779 | 3.0000 |
| correct | 4 | GKT-Transition | 90831 | 70 | 28:Conversion of Fraction Decimals Percents | 0 | 0.4000 | 0 | 0.1000 | True | 0.0153 | -0.2736 | 0.7795 | 0.7569 | 101.0000 |
| correct | 5 | GKT-Transition | 78449 | 80 | 54:nan | 1 | 0.8000 | 1 | 0.3000 | True | 0.0559 | 0.0005 | 0.7410 | 2.9266 | 1.0000 |
| incorrect | 1 | GKT-Transition | 78863 | 107 | 34:Addition Whole Numbers | 0 | 0.9673 | 1 | 0.4673 | True | 0.0805 | 0.0921 | 2.1050 | 4.4407 | 1.0000 |
| incorrect | 2 | GKT-Transition | 78958 | 113 | 43:Finding Percents | 1 | 0.3000 | 0 | 0.2000 | True | 0.3120 | -0.2099 | 1.2169 | 3.1349 | 1.0000 |
| incorrect | 3 | GKT-Transition | 80662 | 27 | 72:Multiplication and Division Integers | 0 | 0.8999 | 1 | 0.3999 | True | 0.0931 | 0.0039 | 0.9920 | 3.5070 | 1.0000 |
| incorrect | 4 | GKT-Transition | 77970 | 65 | 28:Conversion of Fraction Decimals Percents | 0 | 0.6000 | 1 | 0.1000 | True | 0.1479 | -0.0002 | 0.2873 | 2.7790 | 3.0000 |
| incorrect | 5 | GKT-Transition | 79054 | 126 | 93:Write Linear Equation from Ordered Pairs | 0 | 0.7999 | 1 | 0.2999 | True | 0.0609 | 0.1334 | 0.3015 | 2.1898 | 3.0000 |

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
| §5 Cấp 1: baseline không đồ thị + mô hình đồ thị | DKT + GKT (4 cấu hình: DKT, GKT-Transition, GKT-Zero, GKT-Shuffled) — §3 |
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
| GAP P0.1 đa seed, split cố định | §3 chạy 4 cấu hình × 3 seed (đủ 8 khi `TRAIN_MODELS=None`) |
| GAP P1.6 bằng chứng case | phản chứng, ablation đồ thị, occlusion, trạng thái `h_t` — §7 |
| GAP P1.8 Identity ≡ Zero, edge-shuffle | giải thích trong `README.md`; thêm `ShuffledRow` giữ out-degree — §2.1 |
| GAP E3 chưa chứng minh cấu trúc đồ thị | báo cáo trung thực bằng kiểm định + ablation suy luận — §4, §6 |
| GAP P0.3/P2.11 sản phẩm báo cáo | `report_summary.md` + 2 zip — §9, §10 |

**Còn phải làm thủ công:** viết báo cáo 12–18 trang và slide từ `report_summary.md`; Cấp độ 3
(quiz tự thu thập, 10 case đồ thị nâng cao) nằm ngoài phạm vi Cấp độ 1.


## 12. Lưu ý trung thực

- File ASSISTments 2009 xếp theo kỹ năng chứ không theo thời gian, và câu nhiều kỹ năng bị tách
  thành nhiều dòng cùng `order_id`. Từ `order-dedup-v1` chuỗi được sắp theo `order_id` và mỗi lượt
  chỉ giữ một dòng (kỹ năng có `skill_id` nhỏ nhất) — số liệu không so sánh trực tiếp được với
  các lần chạy trước bản này (chúng bị rò rỉ thứ tự + nhãn trùng).
- Mỗi lượt chỉ còn một khái niệm nên đồ thị Transition thưa. Với dự đoán không có cạnh, GKT gần
  với self-update và không khác Zero graph về bản chất.
- Nếu Shuffled ≈ Transition (khi bật biến thể đó trong `TRAIN_MODELS`), kết luận đúng là **chưa
  chứng minh được vai trò cấu trúc cạnh** ở Cấp độ 1; lợi ích (nếu có) đến từ mật độ hàng xóm,
  không từ hình dạng đồ thị.
- Norm `h_t` là proxy trạng thái ẩn, **không phải mastery đã hiệu chỉnh**. Không dùng norm để kết
  luận người học "giỏi/yếu" tuyệt đối.
- `report_summary.md` chỉ chứa số liệu và diễn giải tự động; phần thảo luận, so sánh với bài báo
  và slide vẫn phải viết thủ công.
