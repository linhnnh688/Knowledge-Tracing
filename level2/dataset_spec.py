"""Mô tả từng tập dữ liệu và bước chuẩn hoá về một định dạng duy nhất.

Mục tiêu: để Cấp độ 2 đánh giá GRKT trên ≥ 2 tập dữ liệu công khai với **cùng một quy
trình chia dữ liệu** như Cấp độ 1, mọi tập đều được đổi thành CSV "chuẩn hoá" có đúng
các cột của ``GKT/data/skill_builder_data.csv``:

    order_id, user_id, problem_id, skill_id, skill_name, correct, ms_first_response, timestamp_ms

Nhờ đó ``GKT/processing.load_dataset`` (nơi Cấp độ 1 tạo split theo người học và dựng
đồ thị Transition từ train) chạy y hệt trên cả hai tập — không có hai đường tiền xử lý
tách rời, không có chỗ cho rò rỉ khác nhau giữa hai tập.

ASSISTments 2009 đã đúng định dạng này nên được dùng trực tiếp, không ghi lại file.
Junyi Academy (Kaggle) được đổi qua ``ensure_canonical()``, có dò tên cột vì schema
công bố có thể thay đổi theo phiên bản dữ liệu.

Lưu ý quan trọng với Junyi trên Kaggle: ``Log_Problem.csv`` có ba định danh — ``uuid``
(người học, 72.630 giá trị), ``ucid`` (bài tập/khái niệm, 1.326 giá trị) và ``upid``
(câu hỏi, 25.785 giá trị) — nhưng không có TÊN bài tập. Khái niệm lấy thẳng từ ``ucid``;
chỉ khi cần tên hiển thị mới tra ``Info_Content.csv``. Không được dùng ``upid`` làm khái
niệm (ma trận dense 25.786² không chạy nổi) và không được đảo ``ucid`` lên vai người học.
Xem ``JUNYI.columns`` và ``JUNYI.content_columns``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kt_common.log import get_logger

from .canonicalize import (  # noqa: F401 - re-export cho code/notebook cũ
    _DURATION_SECONDS,
    _to_duration_ms,
    _to_epoch_ms,
    _canonicalize_junyi,
    _cap_concepts,
    _norm_name,
    _match_column,
    _resolve_columns,
    _join_key,
    _correct_flags,
    _find_content_file,
    _load_content_map,
    _CodeMap,
)
from .kaggle_io import (  # noqa: F401 - re-export
    _probe_header,
    find_kaggle_input,
    list_kaggle_inputs,
)

log = get_logger("level2.dataset_spec")

# Thứ tự + tên cột của CSV chuẩn hoá (khớp skill_builder_data.csv ở những cột Cấp độ 1 cần)
CANONICAL_COLUMNS = [
    "order_id",
    "user_id",
    "problem_id",
    "skill_id",
    "skill_name",
    "correct",
    "ms_first_response",
    "timestamp_ms",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Định vị một tập dữ liệu: file gốc, cách ánh xạ cột, cách dùng thông tin thời gian."""

    name: str
    #: File CSV chuẩn hoá mà ``level1.pipeline.load_kt_data`` sẽ nạp (đường dẫn tương đối
    #: được hiểu so với ``GKT/data/``, giống quy ước của Cấp độ 1).
    canonical_file: str
    #: ``"duration_ms"`` — cộng dồn thời gian làm bài (ASSIST không có timestamp tuyệt đối);
    #: ``"timestamp_ms"`` — dùng khoảng cách thời gian thật; ``"step"`` — mỗi bước = 1 đơn vị.
    time_mode: str = "duration_ms"
    #: File gốc cần chuẩn hoá (None = canonical_file đã sẵn sàng trong repo).
    raw_file: Optional[str] = None
    #: Tên cột khả dụng trong file gốc, theo thứ tự ưu tiên.
    columns: dict = field(default_factory=dict)
    #: Số khái niệm tối đa giữ lại cho một lượt làm bài (1 = dùng khái niệm đầu tiên).
    max_concepts: int = 1
    #: Bỏ người học có ít hơn mức này tương tác (khi chuẩn hoá).
    min_interactions: int = 5
    #: Giới hạn số người học (sub-sample có seed) để vừa ngân sách GPU; None = giữ toàn bộ.
    max_users: Optional[int] = None
    #: Cắt chuỗi mỗi người học ở mức này khi ghi CSV chuẩn hoá (Cấp độ 1 còn cắt lại theo
    #: ``max_seq_len``, nên giá trị lớn hơn max_seq_len chỉ làm file to thêm).
    max_keep_per_user: int = 500
    #: Giữ lại tối đa N KHÁI NIỆM hay gặp nhất khi chuẩn hoá (None = giữ hết). Bắt buộc có
    #: với tập có hàng nghìn khái niệm: GRKT giữ ``rel_map``/``pre_map`` dense (K+1)² và
    #: trạng thái mastery [B, K+1, k_hidden] cho MỌI khái niệm tại MỌI bước.
    top_concepts: Optional[int] = None
    #: Số dòng nạp mỗi lần khi quét file log lớn (đọc theo mảnh để không giữ 16 triệu chuỗi
    #: trong RAM — toàn bộ các cột chỉ số được mã hoá thành số nguyên trong lúc quét).
    chunksize: int = 2_000_000
    #: File bảng tra khái niệm (dùng khi file log KHÔNG có cột khái niệm nào).
    content_file: Optional[str] = None
    #: Mẫu dò file bảng tra trong /kaggle/input.
    content_patterns: tuple = ()
    #: Cột của file bảng tra: ``key`` = khoá nối với cột câu hỏi của log, ``skill`` = tên khái niệm.
    content_columns: dict = field(default_factory=dict)
    subsample_seed: int = 42
    #: Ghi chú cho báo cáo / vấn đáp.
    note: str = ""


ASSIST2009 = DatasetSpec(
    name="assist2009",
    canonical_file="skill_builder_data.csv",
    time_mode="duration_ms",
    max_concepts=1,
    # 3 = đúng min_seq_len mặc định của level1.pipeline.load_kt_data. Đổi giá trị này là đổi
    # TẬP NGƯỜI HỌC, tức đổi split: build_grkt_data sẽ fail ở _assert_align và checkpoint
    # DKT/GKT của Cấp độ 1 không còn áp dụng được.
    min_interactions=3,
    note=(
        "ASSISTments 2009/2010 (skill_builder_data.csv, 525.534 dòng, 4.217 người học, "
        "123 kỹ năng, 26.688 problem_id). LƯU Ý: file gốc ĐÃ explode Q-matrix sẵn — một lượt "
        "trả lời gắn k kỹ năng xuất hiện thành k dòng cùng order_id (~34% số dòng), và thứ tự "
        "dòng không theo thời gian; kt_common.interactions sắp theo order_id và gộp dòng trùng "
        "(giữ kỹ năng đầu tiên). "
        "Lọc tối thiểu 3 tương tác/người như Cấp độ 1."
    ),
)

ASSIST2009_SMOKE = DatasetSpec(
    name="assist2009_smoke",
    # File 15 người học của Cấp độ 1 (GKT/data/assistment_test15.csv) — dùng cho QUICK_RUN.
    canonical_file="assistment_test15.csv",
    time_mode="duration_ms",
    max_concepts=1,
    min_interactions=3,
    note=(
        "Bản cắt nhỏ của ASSISTments 2009 (1.888 dòng / 15 người học) mà Cấp độ 1 đã dùng "
        "làm smoke test. Chỉ để kiểm tra mã trong vài phút; SỐ LIỆU KHÔNG ĐƯA VÀO BÁO CÁO."
    ),
)

JUNYI = DatasetSpec(
    name="junyi",
    # Được tạo ra bởi ensure_canonical() trong thư mục xuất của notebook.
    canonical_file="junyi_canonical.csv",
    raw_file=None,  # do notebook điền đường dẫn /kaggle/input/... sau khi dò
    time_mode="timestamp_ms",
    max_concepts=1,
    min_interactions=10,
    # Lần chạy báo cáo dùng TOÀN BỘ người học (~72.630, sau lọc >=10 tương tác ≈ 16,2 triệu
    # lượt). QUICK_RUN truyền max_users=300 để đè lên spec này. Bộ nhớ GRKT tỉ lệ với B và K
    # chứ không với tổng số người, nên số người chỉ ảnh hưởng THỜI GIAN, không ảnh hưởng VRAM.
    max_users=None,
    # Log có 1.326 exercise. Giữ 200 bài phổ biến nhất để đồ thị dense (K+1)² và trạng thái
    # [B, K+1, k_hidden] của MỌI bước vừa 15 GiB T4 với batch 32, max_seq_len 200. Lấy cả
    # 1.326 khái niệm buộc batch còn 4-8 và ước tính >10 giờ/epoch (xem grkt_activation_gib).
    top_concepts=200,
    # Info_Content.csv chỉ để LẤY TÊN bài tập nếu cần; khái niệm vẫn là ucid đọc thẳng từ log
    # (xem columns["skill"]). Để dò được khi người dùng truyền content_path.
    content_file="Info_Content.csv",
    content_patterns=("info_content", "junyi"),
    content_columns={
        # Khoá nối của bảng tra: bản Kaggle khoá theo ucid; bản junyi2015/PSLC khoá theo upid.
        "key": [
            "ucid",
            "upid",
            "problem_id",
            "question_id",
            "id",
            "Problem Id",
            "Anon Problem Id",
        ],
        # Tên khái niệm/bài tập trong bảng tra. _match_column khớp chính xác trước, rồi khớp
        # mờ (bỏ hoa/thường, cách, gạch dưới); nếu vẫn không có, lỗi in ra TOÀN BỘ cột của
        # bảng tra để bổ sung tên đúng vào danh sách này.
        "skill": [
            "exercise",
            "exercise_name",
            "name",
            "title",
            "skill",
            "skills",
            "kc",
            "list_skills",
            "topic",
            "unit",
            "area",
            "chapter",
            "category",
            "tags",
        ],
    },
    columns={
        # Tên cột = danh sách ỨNG VIÊN, khớp CHUỖI CHÍNH XÁC trước rồi khớp MỜ (hoa/thường,
        # cách/gạch dưới). Thứ tự là thứ tự ưu tiên. Danh sách gồm cả tên cột thật của Kaggle
        # 'Log_Problem.csv' (2020) lẫn tên cột bản junyi2015/PSLC, để attach phiên bản nào cũng
        # chạy được; usecols chỉ đọc cột thật sự tồn tại nên thêm ứng viên không tốn RAM.
        #
        # Log_Problem.csv: uuid = NGƯỜI HỌC (72.630 giá trị, khoá của Info_UserData.csv),
        # ucid = BÀI TẬP/khái niệm (1.326 giá trị), upid = CÂU HỎI (25.785 giá trị). Đừng đảo
        # ucid lên vai người học (sẽ còn 1.326 "người học") và đừng lấy upid làm khái niệm
        # (ma trận 25.786² không chạy nổi).
        "user": ["uuid", "user_id", "Anon Student Id", "student_id"],
        "skill": [
            "ucid",
            "exercise",
            "skill",
            "list_skills",
            "tags",
            "KC (Exercise)",
            "Exercise name",
            "KC (Default - Skill for Drill) overview",
        ],
        # "upid" = định danh câu hỏi; "problem_number" là số thứ tự câu trong bài tập nên
        # KHÔNG được dùng làm định danh. "uuid" cũng bị loại: nó là người học, lấy làm
        # problem_id sẽ sinh embedding 72.630 hàng vô nghĩa.
        "problem": [
            "upid",
            "problem",
            "problem_id",
            "assistment_id",
            "Question Id",
            "Problem Name",
            "Step Name",
        ],
        "correct": ["is_correct", "correct", "Outcome"],
        "answer_count": [
            "total_attempt_cnt",
            "answer_count",
            "count_attempts",
            "Answer Count",
        ],
        "timestamp": [
            "timestamp_TW",
            "timestamp_ms",
            "time_done",
            "time",
            "Timestamp",
            "Time of Last Response",
            "Time of First Attempt",
        ],
        "duration": [
            "total_sec_taken",
            "ms_first_response",
            "answer_time",
            "time_taken",
            "Duration (ms)",
        ],
    },
    note=(
        "Junyi Academy — Kaggle 'junyiacademy/learning-activity-public-dataset-by-junyi-academy' "
        "(dataset id 614042, zip ~1.6 GB: Log_Problem.csv 16.217.311 lượt làm bài / 72.630 người "
        "học / 1.326 bài tập / 25.785 câu hỏi, Info_Content.csv, Info_UserData.csv; "
        "2018/08-2019/07, CC BY-NC-SA 4.0). Dataset này YÊU CẦU BẤM 'Agree/Subscribe' "
        "(requiresSubscription=True) rồi mới Add input được, nếu không Kaggle attaches rỗng và "
        "find_kaggle_input() trả None. Cột thật của Log_Problem.csv: timestamp_TW, uuid, ucid, "
        "upid, problem_number, exercise_problem_repeat_session, is_correct, total_sec_taken, "
        "total_attempt_cnt, used_hint_cnt, is_hint_used, is_downgrade, is_upgrade, level. "
        "uuid = người học, ucid = khái niệm (bài tập) và đã có sẵn trong log nên KHÔNG cần nối "
        "bảng tra; chỉ giữ top_concepts khái niệm hay gặp nhất. is_correct = đúng ngay lần đầu "
        "và không dùng gợi ý; chỉ khi không có cột nhãn mới suy từ số lần thử "
        "(1 -> đúng, >=3 -> sai, bỏ 2 — quy ước pyKT). timestamp_TW ở UTC+8 và làm tròn 15 phút "
        "(nên Δt trong cùng phiên thường bằng 0); total_sec_taken tính bằng GIÂY."
    ),
)

_REGISTRY = {spec.name: spec for spec in (ASSIST2009, ASSIST2009_SMOKE, JUNYI)}


def get_spec(name: str) -> DatasetSpec:
    if name not in _REGISTRY:
        raise KeyError(
            f"Tập dữ liệu chưa được mô tả: {name}. Hợp lệ: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def register_spec(spec: DatasetSpec) -> None:
    """Thêm/ghi đè một mô tả tập dữ liệu (dùng khi đổi schema Junyi hoặc thêm tập thứ ba)."""
    _REGISTRY[spec.name] = spec


def resolve_canonical_path(spec: DatasetSpec, search_dirs=None) -> Path:
    """Tìm file CSV chuẩn hoá: đường dẫn tuyệt đối -> thư mục Cấp độ 1 -> các thư mục tìm thêm."""
    path = Path(spec.canonical_file)
    if path.is_absolute() and path.exists():
        return path
    if path.exists():
        return path
    from level1.paths import repo_root

    candidates = [repo_root() / "GKT" / "data" / spec.canonical_file]
    for d in search_dirs or []:
        candidates.append(Path(d) / spec.canonical_file)
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Không tìm thấy CSV chuẩn hoá của '{spec.name}': {spec.canonical_file}. "
        f"Đã thử: {[str(c) for c in candidates]}"
    )


def probe_columns(path) -> list:
    """Đọc riêng dòng header — dùng để in ra notebook khi cần sửa ánh xạ cột."""
    return list(pd.read_csv(path, nrows=0).columns)


def describe_schema(
    raw_path=None, content_path=None, spec=None, verbose: bool = True
) -> dict:
    """In ra notebook những gì SẼ được dùng để chuẩn hoá, mà không nạp file log.

    Chỉ đọc hai dòng header và bảng tra, nên chạy được trên máy CPU trong vài giây: dùng để
    kiểm chứng ánh xạ cột TRƯỚC khi bỏ hàng giờ GPU vào cell thực nghiệm.
    """
    spec = spec or JUNYI
    raw = (
        Path(raw_path)
        if raw_path is not None
        else find_kaggle_input(
            patterns=(spec.name, "log_problem"), preferred_names=("Log_Problem.csv",)
        )
    )
    info = {
        "spec": spec.name,
        "raw": None if raw is None else str(raw),
        "log_columns": None,
        "mapping": None,
        "content": None,
        "error": None,
    }
    if raw is None or not Path(raw).exists():
        info["error"] = (
            f"Không tìm thấy file log của '{spec.name}' trong /kaggle/input."
        )
        if verbose:
            log.info(f"  [schema] {info['error']}\n  {list_kaggle_inputs()}")
        return info
    header = _probe_header(raw)
    info["log_columns"] = header
    try:
        resolved = _resolve_columns(header, spec)
    except KeyError as exc:  # mô tả thì không được chết, chỉ báo cáo
        info["error"] = str(exc)
        if verbose:
            log.warning(f"  [schema] Ánh xạ cột thất bại: {info['error']}")
        return info
    info["mapping"] = {k: v for k, v in resolved.items() if k != "usecols"}
    if resolved["skill"] is None:
        try:
            skill_of_key, cinfo = _load_content_map(
                spec,
                Path(raw),
                resolved["problem"],
                content_path=content_path,
                verbose=False,
            )
            info["content"] = {
                **cinfo,
                "columns": _probe_header(Path(cinfo["file"])),
                "examples": skill_of_key.drop_duplicates().head(10).to_dict(),
            }
        except KeyError as exc:
            info["error"] = str(exc)
    if verbose:
        log.info(f"  [schema] {Path(raw).name}: {header}")
        log.info(f"  [schema] ánh xạ vai trò: {info['mapping']}")
        if info["content"]:
            c = info["content"]
            log.info(f"  [schema] bảng tra {Path(c['file']).name}: cột {c['columns']}")
            log.info(
                f"  [schema]   nối {c['key']} -> {c['skill']}: {c['n_entries']:,} mục, "
                f"{c['n_concepts']:,} khái niệm, phủ {c['coverage']:.1%} lượt"
            )
            log.info(f"  [schema]   ví dụ khái niệm: {list(c['examples'].values())[:10]}")
        if info["error"]:
            log.warning(f"  [schema] LỖI: {info['error']}")
    return info


def has_canonical(spec: DatasetSpec, out_dir: Path) -> bool:
    """CSV chuẩn hoá đã tồn tại (ở out_dir hoặc chỗ resolve_canonical_path tìm thấy)?"""
    if (Path(out_dir) / Path(spec.canonical_file).name).exists():
        return True
    try:
        resolve_canonical_path(spec, search_dirs=[out_dir])
        return True
    except FileNotFoundError:
        return False


def _canonical_meta_path(target: Path) -> Path:
    return target.with_name(target.name + ".meta.json")


def _check_canonical_meta(target: Path, requested: dict, verbose: bool = True) -> None:
    """Chặn dùng lại CSV chuẩn hoá được tạo với cấu hình khác.

    ``ensure_canonical`` trả về file có sẵn mà không đọc lại dữ liệu gốc, nên nếu lần trước
    chạy QUICK_RUN (cap 300 người) rồi lần này chạy toàn bộ, kết quả sẽ lặng lẽ là của 300
    người. File ``.meta.json`` đi kèm ghi đúng ba tham số đã dùng để phát hiện trường hợp đó.
    """
    meta_path = _canonical_meta_path(target)
    if not meta_path.exists():
        if verbose:
            log.warning(
                f"  [cảnh báo] {target.name} không có {meta_path.name}: không xác định được "
                "max_users/min_interactions/top_concepts đã dùng. Nếu lần trước chạy với cap "
                "khác thì xoá file này trước khi chạy lại."
            )
        return
    try:
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    diff = {
        k: (previous.get(k), v) for k, v in requested.items() if previous.get(k) != v
    }
    if diff:
        detail = ", ".join(f"{k}: {old} -> {new}" for k, (old, new) in diff.items())
        raise RuntimeError(
            f"{target.name} được tạo với cấu hình khác ({detail}). Xoá {target} và "
            f"{meta_path.name} rồi chạy lại — dùng lại file cũ sẽ cho kết quả lệch nhưng "
            f"không báo lỗi."
        )


def ensure_canonical(
    spec: DatasetSpec,
    out_dir=None,
    raw_path=None,
    max_users: Optional[int] = None,
    min_interactions: Optional[int] = None,
    top_concepts: Optional[int] = None,
    content_path=None,
    verbose: bool = True,
) -> Path:
    """Trả về đường dẫn CSV chuẩn hoá; nếu spec có file gốc thì tạo file chuẩn hoá một lần.

    ``max_users`` / ``min_interactions`` / ``top_concepts`` cho phép ghi đè cấu hình trong spec
    (dùng khi smoke test trên Kaggle với ngân sách GPU nhỏ). File sinh ra có sidecar
    ``<tên>.meta.json`` ghi ba tham số này, nên lần chạy sau đổi cấu hình sẽ bị chặn thay vì
    lặng lẽ dùng lại dữ liệu cũ. ``content_path`` truyền thẳng bảng tra khái niệm khi
    ``find_kaggle_input`` dò sai (chỉ cần thiết với tập mà file log không có cột khái niệm,
    vd Junyi).
    """
    out_dir = (
        Path(out_dir)
        if out_dir is not None
        else (Path(spec.canonical_file).parent or Path("."))
    )
    min_int = int(
        min_interactions if min_interactions is not None else spec.min_interactions
    )
    cap_users = (
        int(max_users if max_users is not None else (spec.max_users or 0)) or None
    )
    cap_concepts = (
        int(top_concepts if top_concepts is not None else (spec.top_concepts or 0))
        or None
    )
    target = out_dir / Path(spec.canonical_file).name
    if target.exists():
        _check_canonical_meta(
            target,
            {
                "max_users": cap_users,
                "min_interactions": min_int,
                "top_concepts": cap_concepts,
            },
            verbose=verbose,
        )
    if raw_path is None and spec.raw_file is None and not has_canonical(spec, out_dir):
        # Tự dò file gốc trong /kaggle/input: notebook không phải truyền đường dẫn, và lỗi
        # "quên attach dataset" được báo ở ngay cell cấu hình thay vì cell chuẩn hoá.
        found = find_kaggle_input(patterns=(spec.name, "junyi", "log_problem"))
        if found is not None:
            raw_path = found
            if verbose:
                log.info(
                    f"[{spec.name}] Tự dò thấy dữ liệu gốc trong /kaggle/input: {raw_path}"
                )
        elif verbose and spec.name == "junyi":
            log.info(f"  [dò] {list_kaggle_inputs()}")
    if spec.raw_file is None and raw_path is None:
        try:
            path = resolve_canonical_path(spec, search_dirs=[out_dir])
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{spec.name} cần file dữ liệu gốc: attach dataset vào notebook rồi truyền "
                f"raw_path='/kaggle/input/<dataset>/<file>.csv' (xem level2.dataset_spec.find_kaggle_input()). "
                f"Chi tiết: {exc}\n"
                f"Trạng thái /kaggle/input: {list_kaggle_inputs()}"
            ) from exc
        if verbose:
            log.info(f"[{spec.name}] Dùng trực tiếp CSV chuẩn hoá có sẵn: {path}")
        return path

    out_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if verbose:
            log.info(
                f"[{spec.name}] Đã có CSV chuẩn hoá, dùng lại: {target} "
                "(xoá file này nếu đổi max_users/min_interactions/top_concepts hoặc nếu dữ "
                "liệu nguồn thay đổi)"
            )
        return target

    raw = Path(raw_path if raw_path is not None else spec.raw_file)
    if not raw.exists():
        raise FileNotFoundError(
            f"Không thấy file gốc của {spec.name}: {raw}. Với dữ liệu Kaggle, attach dataset "
            f"rồi truyền raw_path='/kaggle/input/<thư-mục-dataset>/<file>.csv'."
        )

    if verbose:
        log.info(
            f"[{spec.name}] Chuẩn hoá {raw.name} -> {target.name} "
            f"(min {min_int} tương tác/người, cap {'toàn bộ' if cap_users is None else cap_users} người, "
            f"cap {'toàn bộ' if cap_concepts is None else cap_concepts} khái niệm)"
        )

    frame = _canonicalize_junyi(
        raw, spec, content_path=content_path, top_concepts=cap_concepts, verbose=verbose
    )
    # Đọc attrs NGAY tại đây: mọi phép sắp xếp/lọc bên dưới đều có quyền bỏ attrs của pandas.
    correct_origin = frame.attrs.get("correct_origin", "không rõ")
    mapping = frame.attrs.get("column_mapping", {})
    decoders = frame.attrs.get("decoders", {})
    frame = frame.sort_values(["user_id", "timestamp_ms"], kind="mergesort")

    counts = frame.groupby("user_id", sort=True).size()
    keep = counts[counts >= min_int]
    if verbose:
        log.info(f"  người học sau lọc tối thiểu tương tác: {len(keep)} / {len(counts)}")
    if cap_users is not None and len(keep) > cap_users:
        rng = np.random.RandomState(spec.subsample_seed)
        chosen = np.sort(
            rng.choice(np.asarray(keep.index), size=cap_users, replace=False)
        )
        keep = keep.loc[list(chosen)]
        if verbose:
            log.info(
                f"  sub-sample theo seed {spec.subsample_seed}: {len(keep)} người học"
            )
    frame = frame[frame["user_id"].isin(keep.index)]

    # Cắt chuỗi mỗi người theo thứ tự thời gian trước khi ghi file (giữ thời gian thật của phần còn lại)
    frame = frame.groupby("user_id", sort=False, group_keys=False).head(
        spec.max_keep_per_user
    )
    # Giải mã sau cùng, khi khung đã nhỏ: 16 triệu chuỗi khái niệm giữ trong RAM là lý do
    # notebook Kaggle chết bằng "Killed" ở lần chạy trước.
    for column, decoding in decoders.items():
        if column in frame.columns:
            frame[column] = frame[column].map(decoding)
    if "skill_id" in frame.columns:
        frame["skill_name"] = frame["skill_id"]
    frame = frame.sort_values(
        ["user_id", "timestamp_ms"], kind="mergesort"
    ).reset_index(drop=True)
    frame["order_id"] = np.arange(1, len(frame) + 1, dtype="int64")

    # Khoảng thời gian giữa hai bước của cùng người học phải không âm và bị chặn ở bước
    # build_grkt_data; ở đây chỉ ghi nguyên vẹn timestamp để tái sử dụng được nhiều cấu hình.
    frame[CANONICAL_COLUMNS].to_csv(target, index=False)
    _canonical_meta_path(target).write_text(
        json.dumps(
            {
                "max_users": cap_users,
                "min_interactions": min_int,
                "top_concepts": cap_concepts,
                "raw_file": str(raw),
                "rows": int(len(frame)),
                "users": int(frame["user_id"].nunique()),
                "concepts": int(frame["skill_id"].nunique()),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if verbose:
        if mapping:
            log.info("  ánh xạ cột: " + " | ".join(f"{k}={v}" for k, v in mapping.items()))
        else:
            log.info(f"  nhãn đúng/sai lấy từ: {correct_origin}")
        log.info(
            f"  đã ghi {len(frame)} tương tác | {frame['user_id'].nunique()} người học | "
            f"{frame['skill_id'].nunique()} khái niệm | {frame['problem_id'].nunique()} câu hỏi"
        )
        log.info(f"  tỉ lệ đúng = {frame['correct'].mean():.3f}")
    return target


