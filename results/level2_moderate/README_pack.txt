GÓI KẾT QUẢ CẤP ĐỘ 2 (GRKT) — Knowledge-Tracing
============================================================
notebook     : 2026-11-09 (order-dedup + bn-eval; đóng gói định kỳ)
git commit   : fc1882d fix: download Git LFS data files on Kaggle in both levels
xuất lúc     : 2026-11-11T15:14:40Z
thiết bị     : cuda
thư mục gốc  : level2_moderate/

PHẠM VI SỐ LIỆU (đọc trước khi trích dẫn)
  assist2009 : toàn bộ split Cấp độ 1, split_seed=42, max_seq_len=200
  junyi      : TẬP CON tối đa 3000 người học (KHÔNG phải toàn bộ 72.630 người).
               Lọc >=10 tương tác, giữ 200 bài phổ biến nhất, lấy mẫu người học có seed.
               Chọn 200 khái niệm dùng thống kê trên toàn bộ log (transductive).

CẤU TRÚC
  <tập>/level2_metrics.csv        bảng DKT / GKT / GRKT (seed + checkpoint_seed)
  <tập>/summary.json              quy mô dữ liệu, số cạnh đồ thị, cấu hình
  <tập>/split.json                danh sách người học train/val/test
  <tập>/trajectory_metrics.csv    tiêu chí quỹ đạo THẬT và kiểm chứng cài đặt
  <tập>/calibration.csv           ECE / Brier
  <tập>/multiseed_metrics.csv     mean ± std trên >=3 seed (nếu bật)
  <tập>/GRKT-<graph>.pt            trọng số GRKT của từng biến thể/seed
  <tập>/baselines/<tên>.pt         trọng số DKT/GKT train lại cho EVAL_ONLY
  <tập>/trajectories/             npz + mastery csv + heatmap
  dashboard/                      dashboard người học (JSON/CSV)
  diagnostics/level1_linkage.json đối chiếu split + AUC với Cấp độ 1
  figures/, tables/               hình và bảng cho báo cáo
  report_summary.md               báo cáo sinh tự động từ đĩa
  run_config.json                 cấu hình đã khoá
  run_sessions.json               lịch sử phiên, history từng epoch, resume
  environment.txt                 phiên bản thư viện / GPU

ĐÁNH GIÁ LẠI KHÔNG TRAIN:
  Đặt EVAL_ONLY=True, attach level2_moderate_output.zip, Run All; §3 nạp GRKT +
  baseline từ zip và chỉ đánh giá. Thiếu checkpoint nào notebook báo đúng đường dẫn.
  Zip tự chứa cả <tập>/*_canonical.csv + .meta.json nên KHÔNG cần attach dữ liệu gốc,
  trừ khi CSV đó bị loại vì lớn hơn max_mb (xem dòng 'bỏ N file' khi đóng gói).

GHI CHÚ
  - consistency_strengthen / consistency_spread / locality = 1,0 là KIỂM CHỨNG CÀI ĐẶT,
    không phải bằng chứng vai trò đồ thị (GAP F1).
  - Zip report_pack KHÔNG có .pt/.npz và KHÔNG có CSV chuẩn hoá -> không resume được,
    chỉ để đọc số liệu.
  - Zip output được luồng nền cập nhật mỗi PACK_EVERY_MIN phút và ghi nguyên tử, nên
    phiên bị kill cứng vẫn để lại zip không cũ hơn một chu kỳ đóng gói.