"""Tiện ích dùng chung cho Cấp độ 1 (``level1``/``GKT``) và Cấp độ 2 (``level2``/``GRKT``).

Mỗi module ở đây là MỘT nguồn sự thật cho một việc từng bị viết lặp lại ở nhiều nơi:

* :mod:`kt_common.interactions` — tiền xử lý CSV tương tác thành chuỗi theo người học;
* :mod:`kt_common.checkpoint` — ghi nguyên tử, state/optimizer/RNG, fingerprint, tên file;
* :mod:`kt_common.metrics` — AUC/ACC toàn cục;
* :mod:`kt_common.ids` — chuẩn hoá id và tên khái niệm;
* :mod:`kt_common.log` — logger in ra stdout (hiện trong notebook như ``print``);
* :mod:`kt_common.seed` — đặt seed tái lập.
"""
