# Báo cáo Cấp độ 2 — GRKT (KDD 2024) và liên kết Cấp độ 1

- Notebook: `2026-11-07a (đóng gói định kỳ + zip tự chứa cho EVAL_ONLY)`
- Git commit: `154d4e4 feat: add PeriodicPackager background thread for atomic periodic output archiving in notebook and experiment modules`
- Sinh lúc (UTC): 2026-11-07T16:07:07Z
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
| assist2009 | 3884.0 | 237943.0 |  | 2330/776/778 | 2720/2642 |
| junyi | 3000.0 | 251598.0 |  | 1800/600/600 | 17866/10173 |

> **Junyi là TẬP CON**: 3,000.0 / 72,630 người (4.1%), 251,598.0 / 16,217,311 lượt (1.6%). Không trích như 'toàn bộ Junyi'.

## 2. Kết quả chính (AUC / Accuracy / NLL)

### assist2009

| model | graph | auc | acc | loss | val_auc | n_test | seed | checkpoint_seed |
|---|---|---|---|---|---|---|---|---|
| GRKT-train | train | 0.7972 | 0.7399 | 0.5217 | 0.8035 | 47358 | 42 |  |
| DKT |  | 0.7859 | 0.7357 | 0.5278 | 0.7926 | 47358 | 42 | 42.0 |
| GRKT-zero | zero | 0.784 | 0.7317 | 0.5311 | 0.7924 | 47358 | 42 |  |
| GKT-Zero | Zero | 0.7644 | 0.7199 | 0.5475 | 0.7758 | 47358 | 42 | 42.0 |
| GKT-Transition | Transition | 0.7536 | 0.7091 | 0.5595 | 0.7626 | 47358 | 42 | 42.0 |

- GRKT tốt nhất (`GRKT-train`) − baseline tốt nhất (`DKT`) = **+0.0113 AUC**.
- ΔAUC(train − zero) = **+0.0132** (có lợi ích từ đồ thị).

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
| train | 150 | 0.3418125544056485 | -0.1107746084273089 | 0.9711989827281317 | 0.4459078590785908 | -0.0004629093395165 | -0.6103817004512125 |
| zero | 150 | 0.2014701615243253 | -0.0776421198699963 | 0.9936149352134392 | 0.008130081300813 | -0.0002380656562116 | -0.2289733702604383 |

**Kiểm chứng cài đặt (kỳ vọng = 1,0; KHÔNG phải bằng chứng):**

| graph | consistency_strengthen | consistency_spread | locality |
|---|---|---|---|
| train | 1.0 | 1.0 | 1.0000000012624355 |
| zero | 1.0 |  | 1.0 |

- `precedence_gap` trung bình = **-0.0942** -> mastery khái niệm tiên quyết KHÔNG cao hơn — tiêu chí thứ tự chưa đạt; phải ghi đúng như vậy trong báo cáo, không suy diễn ngược.
- `forget_slope` trung bình = **-3.505e-04** (âm = mastery giảm khi không luyện).

### junyi

**Tiêu chí thật (dùng làm bằng chứng):**

| graph | users | precedence | precedence_gap | stability_spearman | stability_changed_frac | forget_slope | pearson_dt_dm |
|---|---|---|---|---|---|---|---|
| train | 150 | 0.3650800463539208 | -0.1014209058011958 | 0.9861028345811428 | 0.5119666666666668 | 1.0079433011140092e-05 | -0.0002112530722936 |

**Kiểm chứng cài đặt (kỳ vọng = 1,0; KHÔNG phải bằng chứng):**

| graph | consistency_strengthen | consistency_spread | locality |
|---|---|---|---|
| train | 1.0 | 1.0 | 1.0000000002684435 |

- `precedence_gap` trung bình = **-0.1014** -> mastery khái niệm tiên quyết KHÔNG cao hơn — tiêu chí thứ tự chưa đạt; phải ghi đúng như vậy trong báo cáo, không suy diễn ngược.
- `forget_slope` trung bình = **1.008e-05** (âm = mastery giảm khi không luyện).

## 4. Hiệu chỉnh xác suất

### assist2009

| model | n | brier | ece | ace | mean_pred | observed_rate | over_prediction | graph |
|---|---|---|---|---|---|---|---|---|
| GKT-Transition | 47358 | 0.1893437170425469 | 0.0096924949486622 | 0.0198379730093662 | 0.6196052444359883 | 0.6262933400903754 | -0.0066880956543871 |  |
| DKT | 47358 | 0.1768346525215318 | 0.0134278220413272 | 0.0179147140124384 | 0.6356979908988083 | 0.6262933400903754 | 0.0094046508084328 |  |
| GKT-Zero | 47358 | 0.1847556527285321 | 0.0163140640029623 | 0.0184711314105118 | 0.6287431370801577 | 0.6262933400903754 | 0.0024497969897823 |  |
| GRKT-zero | 47358 | 0.1788288616473821 | 0.0290956494223673 | 0.0313706764241773 | 0.6300819979092406 | 0.6262933400903754 | 0.0037886578188651 | zero |
| GRKT-train | 47358 | 0.1743184730745666 | 0.0379275056173355 | 0.042842135128782 | 0.6238332634405898 | 0.6262933400903754 | -0.0024600766497856 | train |

- ECE thấp nhất: `GKT-Transition` = **0.0097**.

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
| GRKT-train | 0.7983377393155631 | 0.0009387878018991 | 0.742401846924842 | 0.0022609970181526 | 0.5170153776804606 | 0.0037454328032328 | 3 |

## 6. Liên kết Cấp độ 1

- `split.json` C1: **git-lfs-pointer** | C2: **ok**
- Tập người học trùng Cấp độ 1: **None** (max_seq_len=200, split_seed=42)

**Đường nền train lại seed 42 so với `level1_metrics.csv`:**

| model | level1_auc | level2_auc | delta | tolerance | checkpoint_seed | same_weights | status |
|---|---|---|---|---|---|---|---|
| DKT | 0.7866 | 0.7859 | -0.0007 | 0.02 | 42 | False | OK |
| GKT-Transition | 0.7443 | 0.7536 | 0.0093 | 0.02 | 42 | False | OK |
| GKT-Zero | 0.7549 | 0.7644 | 0.0095 | 0.02 | 42 | False | OK |

> same_weights=False -> dung sai 0.02. Δ nhỏ là kiểm tra split/cửa sổ, không chứng minh trọng số trùng Cấp độ 1.

**Checkpoint Cấp độ 1 đã commit (GAP F2):** Checkpoint cũ KHÔNG tái lập bảng C1 -> khớp GAP F2; dùng đường nền train lại.

| model | legacy_auc | legacy_acc | legacy_loss | checkpoint_seed | ckpt_source |
|---|---|---|---|---|---|
| DKT | 0.7851 | 0.7329 | 0.5282 |  | /kaggle/working/Knowledge-Tracing/outputs/level1/DKT.pt |
| GKT-Transition | 0.7722 | 0.7253 | 0.5414 |  | /kaggle/working/Knowledge-Tracing/outputs/level1/GKT-Transition.pt |
| GKT-Zero | 0.762 | 0.712 | 0.5505 |  | /kaggle/working/Knowledge-Tracing/outputs/level1/GKT-Zero.pt |

## 7. Dashboard người học

- 60 người học, cột: dataset, graph, user_id, steps, acc, mean_pred, brier, auc, corr_pred_correct, mastery_first, mastery_last, mastery_delta, mean_neighbor_mastery, evidence_rate, alignment, model

| dataset | graph | user_id | steps | acc | mean_pred | brier | auc | corr_pred_correct | mastery_first | mastery_last | mastery_delta | mean_neighbor_mastery | evidence_rate | alignment | model |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| assist2009 | train | 86709 | 6 | 0.8333 | 0.8512 | 0.0979 | 1.0 | 0.9474 | 0.7769 | 0.8899 | 0.113 | 0.5016 | 1.0 | khớp S-1 | GRKT-train |
| assist2009 | train | 87115 | 40 | 0.6 | 0.8619 | 0.2994 | 0.6172 | 0.1954 | 0.8713 | 0.9028 | 0.0315 | 0.493 | 1.0 | khớp S-1 | GRKT-train |
| assist2009 | train | 88793 | 41 | 0.1707 | 0.3631 | 0.1766 | 0.7647 | 0.3171 | 0.5217 | 0.4274 | -0.0943 | 0.392 | 0.8537 | khớp S-1 | GRKT-train |
| assist2009 | train | 89200 | 24 | 0.6667 | 0.6967 | 0.2477 | 0.5078 | -0.0319 | 0.5654 | 0.5173 | -0.0481 | 0.5086 | 0.75 | khớp S-1 | GRKT-train |
| assist2009 | train | 88038 | 25 | 0.52 | 0.6294 | 0.2125 | 0.7628 | 0.4447 | 0.4722 | 0.5808 | 0.1086 | 0.4641 | 0.8 | khớp S-1 | GRKT-train |
| assist2009 | train | 83306 | 87 | 0.5632 | 0.6009 | 0.1967 | 0.7573 | 0.4543 | 0.6207 | 0.42 | -0.2007 | 0.4175 | 0.954 | khớp S-1 | GRKT-train |
| assist2009 | train | 90993 | 12 | 0.8333 | 0.646 | 0.1054 | 0.95 | 0.704 | 0.7345 | 0.5645 | -0.17 | 0.5352 | 0.6667 | khớp S-1 | GRKT-train |
| assist2009 | train | 90433 | 125 | 0.704 | 0.7045 | 0.2223 | 0.5937 | 0.1714 | 0.636 | 0.6365 | 0.0006 | 0.5293 | 0.848 | khớp S-1 | GRKT-train |
| assist2009 | train | 90515 | 9 | 0.6667 | 0.6867 | 0.158 | 0.7778 | 0.5524 | 0.5139 | 0.5898 | 0.0759 | 0.4872 | 1.0 | khớp S-1 | GRKT-train |
| assist2009 | train | 92355 | 9 | 1.0 | 0.7365 | 0.0766 |  |  | 0.5833 | 0.6062 | 0.0229 | 0.5149 | 0.8889 | khớp S-1 | GRKT-train |

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
| GAP F2 seed checkpoint C1 | Checkpoint cũ KHÔNG tái lập bảng C1 -> khớp GAP F2; dùng đường nền train lại. | Đạt (đã đo; bảng chính dùng đường nền train lại) |
| GAP F3 artifact Junyi khớp mô tả | junyi users=3000.0, cap=3000 | Đạt (tập con, cap khớp) |
| GAP F4 GRKT cùng ngân sách epoch với baseline | epochs_run=[11, 12, 14, 17, 20], cap=20, patience=5 | Đạt |
| GAP F5 config/history/provenance | run_config=có, run_sessions=có, environment=có, learning_curves=có | Đạt |
| GAP F6 biến thể đồ thị | assist2009 variants=['train', 'zero'] | Đạt |
| GAP F7 dashboard C2 | có | Đạt |

## 9. Kết luận tự động

- **assist2009**: mô hình tốt nhất theo Test AUC là `GRKT-train` (đồ thị `train`), AUC = **0.7972**.
- **junyi**: mô hình tốt nhất theo Test AUC là `GRKT-train` (đồ thị `train`), AUC = **0.7896**.
- **Vai trò đồ thị trên assist2009**: ΔAUC = **+0.0132**; có lợi ích đo được.
- **Dashboard** đã có đủ xác suất từng bước, nhãn, mastery trước/sau lan truyền; Junyi chỉ là tập con nên không dùng để suy rộng.

## 10. Hạn chế phải nêu trong báo cáo

1. Junyi là **tập con** (người học lấy mẫu có seed) — mọi số Junyi chỉ có giá trị trên tập con.
2. Chọn 200 khái niệm Junyi dùng thống kê **toàn bộ log** (transductive).
3. `consistency_strengthen`, `consistency_spread`, `locality` ≡ 1 do cấu trúc mô hình, không phải bằng chứng đồ thị (GAP F1).
4. Ablation đồ thị và đa seed chỉ chạy trên assist2009; Junyi 1 seed, 1 đồ thị.
5. ΔAUC(train − zero) chỉ từ một seed: chưa đủ để kết luận nhân quả về đồ thị.
6. Checkpoint Cấp độ 1 đã commit không chứng minh được seed 42; bảng chính dùng đường nền train lại, checkpoint cũ chỉ để chẩn đoán (§5.2).
