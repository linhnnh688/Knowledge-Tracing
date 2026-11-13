# Báo cáo Cấp độ 2 — GRKT (KDD 2024) và liên kết Cấp độ 1

- Notebook: `2026-11-09 (order-dedup + bn-eval; đóng gói định kỳ)`
- Git commit: `fc1882d fix: download Git LFS data files on Kaggle in both levels`
- Sinh lúc (UTC): 2026-11-11T15:14:40Z
- Thiết bị: `cuda`

## 0. Cấu hình đã khoá (run_config.json)

```json
{
  "alpha": 0.01,
  "baseline_epochs": 20,
  "batch_size": 32,
  "cold_thresh": 5,
  "d_hidden": 128,
  "datasets": [
    "assist2009",
    "junyi"
  ],
  "epochs": 20,
  "gap_cap_min": 240,
  "graph_thresh": 0.6,
  "k_hidden": 16,
  "k_hop": 1,
  "lr": 0.001,
  "max_seq_len": 200,
  "max_users": 3000,
  "multi_seed_datasets": [
    "assist2009"
  ],
  "patience": 5,
  "prep_version": "order-dedup-v1",
  "run_baselines": true,
  "run_calibration": true,
  "seeds": [
    42,
    123,
    456
  ],
  "split_seed": 42,
  "use_committed_ckpts": false,
  "variants_by_dataset": {
    "assist2009": [
      "train",
      "zero"
    ],
    "junyi": [
      "train"
    ]
  }
}
```

## 1. Phạm vi và quy mô (2 tập có bảng kết quả)

| tập | người học | tương tác | khái niệm | train/val/test | cạnh rel/pre (train) |
|---|---|---|---|---|---|
| assist2009 | 3852.0 | 198344.0 |  | 2311/770/771 | 3246/2609 |
| junyi | 3000.0 | 251598.0 |  | 1800/600/600 | 17866/10173 |

> **Junyi là TẬP CON**: 3,000.0 / 72,630 người (4.1%), 251,598.0 / 16,217,311 lượt (1.6%). Không trích như 'toàn bộ Junyi'.

## 2. Kết quả chính (AUC / Accuracy / NLL)

### assist2009

| model | graph | auc | acc | loss | val_auc | n_test | seed | checkpoint_seed |
|---|---|---|---|---|---|---|---|---|
| GRKT-train | train | 0.7691 | 0.7259 | 0.547 | 0.7584 | 39385 | 42 |  |
| DKT |  | 0.7608 | 0.7237 | 0.5493 | 0.7532 | 39385 | 42 | 42.0 |
| GRKT-zero | zero | 0.7538 | 0.7131 | 0.5625 | 0.7468 | 39385 | 42 |  |
| GKT-Transition | Transition | 0.7325 | 0.7081 | 0.5721 | 0.7191 | 39385 | 42 | 42.0 |
| GKT-Zero | Zero | 0.7233 | 0.7 | 0.5833 | 0.7181 | 39385 | 42 | 42.0 |

- GRKT tốt nhất (`GRKT-train`) − baseline tốt nhất (`DKT`) = **+0.0083 AUC**.
- ΔAUC(train − zero) = **+0.0153** (có lợi ích từ đồ thị).

### junyi

| model | graph | auc | acc | loss | val_auc | n_test | seed | checkpoint_seed |
|---|---|---|---|---|---|---|---|---|
| GRKT-train | train | 0.7896 | 0.8017 | 0.4378 | 0.7938 | 48844 | 42 |  |
| DKT |  | 0.7501 | 0.7864 | 0.4623 | 0.752 | 48844 | 42 | 42.0 |
| GKT-Zero | Zero | 0.7225 | 0.7821 | 0.4798 | 0.728 | 48844 | 42 | 42.0 |
| GKT-Transition | Transition | 0.7202 | 0.78 | 0.479 | 0.7208 | 48844 | 42 | 42.0 |

- GRKT tốt nhất (`GRKT-train`) − baseline tốt nhất (`DKT`) = **+0.0395 AUC**.

## 3. Quỹ đạo mastery m(u,c,t)

### assist2009

**Tiêu chí thật (dùng làm bằng chứng):**

| graph | users | precedence | precedence_gap | stability_spearman | stability_changed_frac | forget_slope | pearson_dt_dm |
|---|---|---|---|---|---|---|---|
| train | 150 | 0.5098097502972652 | -0.0071608469892874 | 0.982812449115501 | 0.5272023809523809 | -0.000422374162078 | -0.3086845057340957 |
| zero | 150 | 0.3535969084423305 | -0.0363963780418224 | 0.9926087286850748 | 0.0089285714285714 | -0.0004166143467372 | -0.5904860135159451 |

**Kiểm chứng cài đặt (kỳ vọng = 1,0; KHÔNG phải bằng chứng):**

| graph | consistency_strengthen | consistency_spread | locality |
|---|---|---|---|
| train | 1.0 | 1.0 | 0.9999999987612744 |
| zero | 1.0 |  | 1.0 |

- `precedence_gap` trung bình = **-0.0218** -> mastery khái niệm tiên quyết KHÔNG cao hơn — tiêu chí thứ tự chưa đạt; phải ghi đúng như vậy trong báo cáo, không suy diễn ngược.
- `forget_slope` trung bình = **-4.195e-04** (âm = mastery giảm khi không luyện).

### junyi

**Tiêu chí thật (dùng làm bằng chứng):**

| graph | users | precedence | precedence_gap | stability_spearman | stability_changed_frac | forget_slope | pearson_dt_dm |
|---|---|---|---|---|---|---|---|
| train | 150 | 0.6851065472513898 | 0.0187043647139911 | 0.9861028345811428 | 0.5119666666666668 | 1.0079433011140092e-05 | -0.0002112530722936 |

**Kiểm chứng cài đặt (kỳ vọng = 1,0; KHÔNG phải bằng chứng):**

| graph | consistency_strengthen | consistency_spread | locality |
|---|---|---|---|
| train | 1.0 | 1.0 | 1.0000000002684435 |

- `precedence_gap` trung bình = **+0.0187** -> mastery khái niệm tiên quyết cao hơn khái niệm phụ thuộc (đúng chiều).
- `forget_slope` trung bình = **1.008e-05** (âm = mastery giảm khi không luyện).

## 4. Hiệu chỉnh xác suất

### assist2009

| model | n | brier | ece | ace | mean_pred | observed_rate | over_prediction | graph |
|---|---|---|---|---|---|---|---|---|
| DKT | 39385 | 0.18505384010133 | 0.0064902483729317 | 0.0099226082095602 | 0.6259017435869991 | 0.6255935000634759 | 0.0003082435235232 |  |
| GRKT-train | 39385 | 0.1830727741162462 | 0.0117398284637094 | 0.019431204362873 | 0.6198644873177719 | 0.6255935000634759 | -0.005729012745704 | train |
| GKT-Transition | 39385 | 0.1945733166565406 | 0.0132242346277761 | 0.0167639553472307 | 0.6353656093652507 | 0.6255935000634759 | 0.0097721093017747 |  |
| GRKT-zero | 39385 | 0.189883733518714 | 0.0332175748935489 | 0.0410111067990395 | 0.6187230964510517 | 0.6255935000634759 | -0.0068704036124241 | zero |
| GKT-Zero | 39385 | 0.1992653956698506 | 0.036631996013102 | 0.0345457217049666 | 0.6621453218469466 | 0.6255935000634759 | 0.0365518217834707 |  |

- ECE thấp nhất: `DKT` = **0.0065**.

### junyi

| model | n | brier | ece | ace | mean_pred | observed_rate | over_prediction | graph |
|---|---|---|---|---|---|---|---|---|
| GKT-Transition | 48844 | 0.1565505447895952 | 0.0112951988710197 | 0.0303031454112882 | 0.7788478771631743 | 0.7720293178281877 | 0.0068185593349865 |  |
| GRKT-train | 48844 | 0.1403689819388945 | 0.0126763375985776 | 0.0288404295148804 | 0.7763689646968417 | 0.7720293178281877 | 0.0043396468686539 | train |
| DKT | 48844 | 0.1504260835523246 | 0.0170991174140234 | 0.0389325044531122 | 0.7618396382572347 | 0.7720293178281877 | -0.010189679570953 |  |
| GKT-Zero | 48844 | 0.1564491825345408 | 0.0309783411546705 | 0.0416579791313669 | 0.8029067742672382 | 0.7720293178281877 | 0.0308774564390504 |  |

- ECE thấp nhất: `GKT-Transition` = **0.0113**.

## 5. Đa seed

### assist2009

| model | auc_mean | auc_std | acc_mean | acc_std | loss_mean | loss_std | n_seed |
|---|---|---|---|---|---|---|---|
| GRKT-train | 0.7693924690451363 | 0.0006394944146464 | 0.7257416105962508 | 0.0010083954810955 | 0.5488933324813843 | 0.0028899382804771 | 3 |
| GRKT-zero | 0.7551422893799264 | 0.0011319668659784 | 0.7152638483348145 | 0.0019243717236673 | 0.560550332069397 | 0.0016638398713251 | 3 |

## 6. Liên kết Cấp độ 1

- `split.json` C1: **missing** | C2: **ok**
- Tập người học trùng Cấp độ 1: **None** (max_seq_len=200, split_seed=42)

**Checkpoint Cấp độ 1 đã commit (GAP F2):** EVAL_COMMITTED_CKPTS=False hoặc không có checkpoint -> bỏ qua.

## 7. Dashboard người học

- 60 người học, cột: dataset, graph, user_id, steps, acc, mean_pred, brier, auc, corr_pred_correct, mastery_first, mastery_last, mastery_delta, mean_neighbor_mastery, evidence_rate, alignment, model

| dataset | graph | user_id | steps | acc | mean_pred | brier | auc | corr_pred_correct | mastery_first | mastery_last | mastery_delta | mean_neighbor_mastery | evidence_rate | alignment | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| assist2009 | train | 78903 | 199 | 0.804 | 0.7472 | 0.1622 | 0.6526 | 0.1699 | 0.651 | 0.7063 | 0.0553 | 0.5602 | 0.7889 | khớp S-1 | GRKT-train |
| assist2009 | train | 96294 | 199 | 0.8543 | 0.7797 | 0.1105 | 0.7935 | 0.3972 | 0.6593 | 0.7199 | 0.0605 | 0.6262 | 0.7889 | khớp S-1 | GRKT-train |
| assist2009 | train | 89444 | 31 | 0.5806 | 0.6598 | 0.2555 | 0.5812 | 0.1037 | 0.5726 | 0.6151 | 0.0425 | 0.4793 | 0.8387 | khớp S-1 | GRKT-train |
| assist2009 | train | 84105 | 12 | 0.5 | 0.6077 | 0.3127 | 0.3056 | -0.354 | 0.6505 | 0.5911 | -0.0594 | 0.4927 | 0.9167 | khớp S-1 | GRKT-train |
| assist2009 | train | 79279 | 9 | 0.7778 | 0.6756 | 0.2026 | 0.4286 | 0.0518 | 0.5854 | 0.5903 | 0.0049 | 0.5081 | 0.6667 | khớp S-1 | GRKT-train |
| assist2009 | train | 91988 | 10 | 0.8 | 0.7987 | 0.0893 | 1.0 | 0.9053 | 0.6799 | 0.6151 | -0.0648 | 0.4987 | 0.9 | khớp S-1 | GRKT-train |
| assist2009 | train | 80009 | 35 | 0.7143 | 0.6719 | 0.1775 | 0.728 | 0.3763 | 0.5265 | 0.6181 | 0.0916 | 0.5037 | 0.8571 | khớp S-1 | GRKT-train |
| assist2009 | train | 87249 | 4 | 1.0 | 0.597 | 0.2242 |  |  | 0.4512 | 0.625 | 0.1738 | 0.5036 | 0.75 | khớp S-1 | GRKT-train |
| assist2009 | train | 88228 | 8 | 1.0 | 0.8669 | 0.0199 |  |  | 0.5662 | 0.7205 | 0.1543 | 0.5025 | 1.0 | khớp S-1 | GRKT-train |
| assist2009 | train | 87392 | 8 | 0.75 | 0.5653 | 0.1834 | 0.75 | 0.4515 | 0.5906 | 0.7544 | 0.1638 | 0.5001 | 0.875 | khớp S-1 | GRKT-train |

## 8. Đối chiếu yêu cầu và khoảng trống

| Mục | Bằng chứng (đọc từ đĩa) | Kết luận |
|---|---|---|
| §5 GRKT trên ≥2 tập công khai | 2 tập: assist2009, junyi | Đạt |
| §5 dùng split theo người học của C1 | same_split=False, seed=42, max_seq_len=200 | Một phần (cùng seed/hàm, thiếu bằng chứng file) |
| §6.2 đồ thị chỉ từ train | graphs_train_only.npz + leak_report.txt | Đạt |
| §6.4 train/val/test rời nhau | 0 dòng đối chiếu | Cần kiểm tra split.json |
| §7 AUC / ACC / NLL | level2_metrics.csv có đủ auc/acc/loss | Đạt |
| §7 mean ± std ≥3 seed | assist2009 n_seed min=3 | Đạt |
| §7 ablation đồ thị (train vs zero) | assist2009 có zero=True | Đạt |
| §7 một phân tích nâng cao | quỹ đạo thật=có, calibration=có | Đạt |
| §8 dashboard người học C2 | learner_dashboard_summary.csv=có | Đạt |
| GAP F1 tiêu chí quỹ đạo không đóng băng | tách bảng THẬT vs KIỂM CHỨNG CÀI ĐẶT | Đạt |
| GAP F2 seed checkpoint C1 | EVAL_COMMITTED_CKPTS=False hoặc không có checkpoint -> bỏ qua. | Một phần |
| GAP F3 artifact Junyi khớp mô tả | junyi users=3000.0, cap=3000 | Đạt (tập con, cap khớp) |
| GAP F4 GRKT cùng ngân sách epoch với baseline | epochs_run=[9, 11, 13, 14, 20], cap=20, patience=5 | Đạt |
| GAP F5 config/history/provenance | run_config=có, run_sessions=có, environment=có, learning_curves=có | Đạt |
| GAP F6 biến thể đồ thị | assist2009 variants=['train', 'zero'] | Đạt |
| GAP F7 dashboard C2 | có | Đạt |

## 9. Kết luận tự động

- **assist2009**: mô hình tốt nhất theo Test AUC là `GRKT-train` (đồ thị `train`), AUC = **0.7691**.
- **junyi**: mô hình tốt nhất theo Test AUC là `GRKT-train` (đồ thị `train`), AUC = **0.7896**.
- **Vai trò đồ thị trên assist2009**: ΔAUC = **+0.0153**; có lợi ích đo được.
- **Dashboard** đã có đủ xác suất từng bước, nhãn, mastery trước/sau lan truyền; Junyi chỉ là tập con nên không dùng để suy rộng.

## 10. Hạn chế phải nêu trong báo cáo

1. Junyi là **tập con** (người học lấy mẫu có seed) — mọi số Junyi chỉ có giá trị trên tập con.
2. Chọn 200 khái niệm Junyi dùng thống kê **toàn bộ log** (transductive).
3. `consistency_strengthen`, `consistency_spread`, `locality` ≡ 1 do cấu trúc mô hình, không phải bằng chứng đồ thị (GAP F1).
4. Ablation đồ thị và đa seed chỉ chạy trên assist2009; Junyi 1 seed, 1 đồ thị.
5. ΔAUC(train − zero) chỉ từ một seed: chưa đủ để kết luận nhân quả về đồ thị.
6. Checkpoint Cấp độ 1 đã commit không chứng minh được seed 42; bảng chính dùng đường nền train lại, checkpoint cũ chỉ để chẩn đoán (§5.2).
