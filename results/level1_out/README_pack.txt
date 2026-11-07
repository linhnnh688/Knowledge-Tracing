level1_out — sản phẩm Cấp độ 1
Sinh bởi: 2026-11-07 level1-full+resume+select+evalonly
Git commit: 2e8110a4c006304fba44b595a57e0a078dc3b718
Dữ liệu: skill_builder_data.csv (sha256=f4b8791ae5968b884b77ab3b08adec48130503ec57f3113158c9a5af88af3374)
Cấu hình: epochs=20, batch=32, max_seq_len=200, seeds=[42, 123, 456, 789, 1011]
Cấu hình train: ['DKT', 'GKT-MHA'] (2 cấu hình × 5 seed)

Cấu trúc:
  tables/                  bảng số liệu CSV (UTF-8 BOM, mở được bằng Excel)
  figures/                 hình PNG 150 dpi để dán vào báo cáo
  cases.md / cases.json    10 case study + bằng chứng định lượng
  learner_view.json        dashboard người học (giới hạn 25 người)
  learner_view_summary.csv bảng tóm tắt người học
  learner_example.txt      ví dụ render một người học
  split.json               split theo người học seed 42 (tái lập)
  leak_report.txt          kiểm tra chống rò rỉ
  run_manifest.json        manifest tái lập
  environment.txt          phiên bản môi trường
  report_summary.md        báo cáo tổng hợp tự động
  checkpoints/             trọng số của MỌI run đã xong (2 cấu hình × 5 seed)
                           seed 42: <tên>.pt | seed khác: <tên>-seed<s>.pt
                           + ckpt_<tên>-seed<s>.pt của run đang dở (nếu có)
  runs_progress.json       sổ tiến độ từng (mô hình, seed) — dùng để resume
  run_config.json          cấu hình đã khoá; resume khác cấu hình sẽ bị chặn
  run_sessions.json        lịch sử các phiên Kaggle (started/done/eval_only/paused/error)

RESUME SAU KHI HẾT 12 GIỜ:
  1. Mỗi nhóm seed xong, notebook tự tạo level1_output.zip và in đường dẫn.
  2. Tải zip -> Kaggle Dataset (private) -> Add Input cho phiên sau.
  3. Chạy lại toàn bộ notebook. §0.3 tự tìm zip; nếu có nhiều zip thì đặt RESUME_ZIP.
  4. §3 bỏ qua run đã xong và resume đúng epoch của run dở.
  Giữ nguyên EPOCHS/SEEDS/BATCH_SIZE/MAX_SEQ_LEN/QUICK_RUN/TRAIN_MODELS; đổi sẽ bị
  run_config.json chặn (trừ khi ALLOW_CONFIG_MISMATCH=True).
  Nếu phiên bị kill đột ngột, đặt TIME_BUDGET_MIN=480 để chắc chắn có zip trước khi hết giờ.

CHỈ ĐÁNH GIÁ + DASHBOARD TỪ OUTPUT CŨ:
  Đặt EVAL_ONLY=True ở §0.2, attach zip này vào /kaggle/input, rồi Run All.
  §3 KHÔNG train; thiếu checkpoint/sổ nào sẽ báo đúng tên file.

Cách dùng cho báo cáo:
  1. Mở report_summary.md để lấy bảng và kết luận.
  2. Dùng figures/ cho phần trực quan.
  3. Dùng cases.md cho mục xAI 5 đúng + 5 sai.
  4. Trích số từ tables/ khi cần bảng chi tiết.

Lưu ý: bản zip này không thay thế báo cáo/slide; chỉ cung cấp số liệu và hình.
