"""GRKT (Cui et al., KDD 2024) — bản port từ https://github.com/JJCui96/GRKT.

Khác bản gốc: mã được đóng gói thành Python package để import an toàn theo dạng
``from GRKT.model.GRKT import GRKT``.

Lý do bắt buộc phải sửa import: thư mục ``GKT/`` (Cấp độ 1) cũng có một file tên
``utils.py`` và cả ``GKT/`` lẫn ``GRKT/`` đều được thêm vào ``sys.path``, nên câu
lệnh ``from utils import *`` trong bản gốc có thể nạp nhầm module của mô hình kia.
Toàn bộ thay đổi so với bản gốc được khai báo trong ``KE_HOACH.md`` (mục khai báo
mã nguồn mở) và được ghi chú trực tiếp tại từng chỗ sửa.
"""
