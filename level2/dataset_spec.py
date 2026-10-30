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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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
        "123 kỹ năng, 26.688 problem_id). Mỗi dòng đúng một skill_id — cùng tính chất với "
        "Cấp độ 1, nên không explode Q-matrix. Lọc tối thiểu 3 tương tác/người như Cấp độ 1."
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
    # 800 người học ~ 96.000 lượt (cap max_keep_per_user) — GRKT tốn bộ nhớ tỉ lệ
    # S × B × (K+1) × k_hidden cho MỌI bước, nên 4.000 người học không chạy nổi trên T4.
    max_users=800,
    # Log có 1.326 exercise; K=200 là mức để đồ thị (K+1)² và các tensor [B, K+1, 16]
    # vừa VRAM 16 GB với batch 32-64, max_seq_len 200.
    top_concepts=200,
    # Info_Content.csv chỉ để LẤY TÊN bài tập nếu cần; khái niệm vẫn là ucid đọc thẳng từ log
    # (xem columns["skill"]). Để dò được khi người dùng truyền content_path.
    content_file="Info_Content.csv",
    content_patterns=("info_content", "junyi"),
    content_columns={
        # Khoá nối của bảng tra: bản Kaggle khoá theo ucid; bản junyi2015/PSLC khoá theo upid.
        "key": ["ucid", "upid", "problem_id", "question_id", "id", "Problem Id", "Anon Problem Id"],
        # Tên khái niệm/bài tập trong bảng tra. _match_column khớp chính xác trước, rồi khớp
        # mờ (bỏ hoa/thường, cách, gạch dưới); nếu vẫn không có, lỗi in ra TOÀN BỘ cột của
        # bảng tra để bổ sung tên đúng vào danh sách này.
        "skill": ["exercise", "exercise_name", "name", "title", "skill", "skills", "kc",
                  "list_skills", "topic", "unit", "area", "chapter", "category", "tags"],
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
        "skill": ["ucid", "exercise", "skill", "list_skills", "tags", "KC (Exercise)",
                  "Exercise name", "KC (Default - Skill for Drill) overview"],
        # "upid" = định danh câu hỏi; "problem_number" là số thứ tự câu trong bài tập nên
        # KHÔNG được dùng làm định danh. "uuid" cũng bị loại: nó là người học, lấy làm
        # problem_id sẽ sinh embedding 72.630 hàng vô nghĩa.
        "problem": ["upid", "problem", "problem_id", "assistment_id",
                    "Question Id", "Problem Name", "Step Name"],
        "correct": ["is_correct", "correct", "Outcome"],
        "answer_count": ["total_attempt_cnt", "answer_count", "count_attempts", "Answer Count"],
        "timestamp": ["timestamp_TW", "timestamp_ms", "time_done", "time", "Timestamp",
                      "Time of Last Response", "Time of First Attempt"],
        "duration": ["total_sec_taken", "ms_first_response", "answer_time", "time_taken",
                     "Duration (ms)"],
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
        raise KeyError(f"Tập dữ liệu chưa được mô tả: {name}. Hợp lệ: {sorted(_REGISTRY)}")
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
    for d in (search_dirs or []):
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


def _probe_header(path) -> list:
    """Đọc dòng header của file GỐC (BOM đã được gỡ) — không nạp dữ liệu."""
    return list(pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns)


def describe_schema(raw_path=None, content_path=None, spec=None, verbose: bool = True) -> dict:
    """In ra notebook những gì SẼ được dùng để chuẩn hoá, mà không nạp file log.

    Chỉ đọc hai dòng header và bảng tra, nên chạy được trên máy CPU trong vài giây: dùng để
    kiểm chứng ánh xạ cột TRƯỚC khi bỏ hàng giờ GPU vào cell thực nghiệm.
    """
    spec = spec or JUNYI
    raw = Path(raw_path) if raw_path is not None else find_kaggle_input(
        patterns=(spec.name, "log_problem"), preferred_names=("Log_Problem.csv",))
    info = {"spec": spec.name, "raw": None if raw is None else str(raw), "log_columns": None,
            "mapping": None, "content": None, "error": None}
    if raw is None or not Path(raw).exists():
        info["error"] = f"Không tìm thấy file log của '{spec.name}' trong /kaggle/input."
        if verbose:
            print(f"  [schema] {info['error']}\n  {list_kaggle_inputs()}")
        return info
    header = _probe_header(raw)
    info["log_columns"] = header
    try:
        resolved = _resolve_columns(header, spec)
    except KeyError as exc:                      # mô tả thì không được chết, chỉ báo cáo
        info["error"] = str(exc)
        if verbose:
            print(f"  [schema] Ánh xạ cột thất bại: {info['error']}")
        return info
    info["mapping"] = {k: v for k, v in resolved.items() if k != "usecols"}
    if resolved["skill"] is None:
        try:
            skill_of_key, cinfo = _load_content_map(spec, Path(raw), resolved["problem"],
                                                   content_path=content_path, verbose=False)
            info["content"] = {**cinfo, "columns": _probe_header(Path(cinfo["file"])),
                               "examples": skill_of_key.drop_duplicates().head(10).to_dict()}
        except KeyError as exc:
            info["error"] = str(exc)
    if verbose:
        print(f"  [schema] {Path(raw).name}: {header}")
        print(f"  [schema] ánh xạ vai trò: {info['mapping']}")
        if info["content"]:
            c = info["content"]
            print(f"  [schema] bảng tra {Path(c['file']).name}: cột {c['columns']}")
            print(f"  [schema]   nối {c['key']} -> {c['skill']}: {c['n_entries']:,} mục, "
                  f"{c['n_concepts']:,} khái niệm, phủ {c['coverage']:.1%} lượt")
            print(f"  [schema]   ví dụ khái niệm: {list(c['examples'].values())[:10]}")
        if info["error"]:
            print(f"  [schema] LỖI: {info['error']}")
    return info


#: Các cột độ dài làm bài tính bằng GIÂY; mọi cột khác (``ms_first_response``) đã là miligiây.
_DURATION_SECONDS = frozenset({"total_sec_taken", "answer_time", "time_taken"})


def _to_duration_ms(series: pd.Series, column: str) -> pd.Series:
    """Đổi độ dài làm bài sang miligiây; đơn vị lấy theo TÊN CỘT, không đoán từ phân phối.

    Đoán theo median sẽ sai với Junyi: ``total_sec_taken`` có median vài chục giây, nằm dưới
    ngưỡng 500 nên bị coi là miligiây và ghi thiếu 1000 lần.
    """
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0)
    unit = 1000.0 if column in _DURATION_SECONDS else 1.0
    return (values * unit).astype("int64")


def _to_epoch_ms(series: pd.Series) -> pd.Series:
    """Đổi cột thời gian sang miligiây từ epoch; nhận ISO string, giây, miligiây, microgiây."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        anchor = values.dropna()
        if anchor.empty:
            return pd.Series(np.full(len(series), np.nan), index=series.index)
        top = float(anchor.abs().max())
        if top > 1e14:  # microgiây (Junyi time_done)
            factor = 1e-3
        elif top > 1e11:  # miligiây
            factor = 1.0
        elif top > 1e8:  # giây
            factor = 1e3
        else:  # ngày dạng số hoặc chuỗi ngắn -> coi như không dùng được
            return pd.Series(np.full(len(series), np.nan), index=series.index)
        return (values * factor).astype("Float64").astype("float64")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    # Trừ mốc rồi đo giây: astype("int64") trên datetime64 có NaT cho ra số rất âm (pandas 2)
    # hoặc raise (pandas 3), còn cách này trả NaN -> _canonicalize_junyi thay bằng bộ đếm dòng.
    epoch = pd.Timestamp(0, tz="UTC")
    return (parsed - epoch).dt.total_seconds() * 1e3


def _canonicalize_junyi(raw_path: Path, spec: DatasetSpec, content_path=None,
                        top_concepts=None, verbose: bool = True) -> pd.DataFrame:
    """Đổi file log về khung chuẩn hoá, đọc THEO MẢNH và chỉ giữ các cột thật sự cần.

    Hai ràng buộc định hình hàm này:

    * ``Log_Problem.csv`` có 16,2 triệu dòng. Nạp nguyên chuỗi (user/exercise/timestamp dạng
      Python object) rồi mới lọc là cách notebook Kaggle chết giữa chừng bằng ``Killed``; vì
      vậy mỗi mảnh được giảm ngay về các cột số nguyên (mã hoá id bằng :class:`_CodeMap`) và
      ``ensure_canonical`` chỉ GIẢI MÃ các dòng đã được giữ.
    * ``GRKT`` giữ ``rel_map``/``pre_map`` dense trên khái niệm, nên số khái niệm phải bị chặn
      (``top_concepts``) TRƯỚC khi lọc người học — lọc người học sau khi bỏ khái niệm hiếm mới
      đúng số tương tác thực tế của mỗi người.
    """
    raw_path = Path(raw_path)
    header = _probe_header(raw_path)
    # _resolve_columns raise ngay nếu một vai trò bắt buộc không khớp tên cột nào: thông báo
    # đã kèm danh sách ứng viên và TOÀN BỘ cột có trong file, không cần đoán tiếp.
    resolved = _resolve_columns(header, spec)
    usecols = resolved["usecols"]
    col_user, col_prob = resolved["user"], resolved["problem"]
    ts_col, dur_col = resolved["timestamp"], resolved["duration"]
    col_correct, col_ac, col_skill = resolved["correct"], resolved["answer_count"], resolved["skill"]
    if col_correct is None and col_ac is None:
        raise KeyError(f"[{raw_path.name}] Không có cột nhãn đúng/sai cũng không có cột số lần thử. "
                       f"Ứng viên: {spec.columns.get('correct')} / {spec.columns.get('answer_count')}. "
                       f"Cột có sẵn: {header}.")
    if verbose:
        print(f"  đọc {len(usecols)}/{len(header)} cột: {usecols}")
        print(f"  ánh xạ: user={col_user} | câu hỏi={col_prob} | thời điểm={ts_col} | "
              f"độ dài={dur_col} | nhãn={'(' + col_correct + ')' if col_correct else 'suy từ ' + col_ac}")

    # Khái niệm: hoặc có sẵn trong log (bản junyi2015/PSLC có cột 'exercise'), hoặc phải nối
    # bảng tra Info_Content.csv theo định danh câu hỏi.
    skill_of_key = None
    if col_skill is None:
        skill_of_key, cinfo = _load_content_map(spec, raw_path, col_prob, content_path, verbose=verbose)
    elif verbose:
        print(f"  khái niệm lấy trực tiếp từ cột '{col_skill}' của file log")

    maps = {c: _CodeMap(c) for c in ("user_id", "skill_id", "problem_id")}
    pieces, n_raw, n_kept = [], 0, 0
    for chunk in pd.read_csv(raw_path, usecols=usecols, chunksize=max(1, int(spec.chunksize)),
                            low_memory=False, encoding="utf-8-sig"):
        n_raw += len(chunk)
        flags, origin = _correct_flags(chunk, col_correct, col_ac)
        skill = (chunk[col_skill].astype("string").str.strip() if skill_of_key is None
                 else _join_key(chunk[col_prob]).map(skill_of_key).astype("string"))
        ok = (flags.isin([0.0, 1.0]) & skill.notna() & (skill.str.len() > 0)
              & chunk[col_user].notna() & chunk[col_prob].notna())
        ok = ok.to_numpy()
        if not ok.any():
            continue
        chunk, flags, skill = chunk[ok], flags[ok], skill[ok]
        piece = pd.DataFrame({
            "user_id": maps["user_id"].encode(chunk[col_user]),
            "skill_id": maps["skill_id"].encode(skill),
            "problem_id": maps["problem_id"].encode(chunk[col_prob]),
            "correct": flags.astype("float64").astype("int8").to_numpy(),
            "timestamp_ms": _to_epoch_ms(chunk[ts_col]).to_numpy(dtype="float64"),
            "ms_first_response": _to_duration_ms(chunk[dur_col], dur_col).to_numpy(dtype="int64"),
        })
        pieces.append(piece)
        n_kept += len(piece)
    if not pieces:
        raise ValueError(
            f"[{raw_path.name}] Không giữ được dòng nào từ {n_raw:,} dòng đã đọc "
            f"(nhãn đúng/sai từ '{col_correct or origin}', khái niệm từ "
            f"'{col_skill or 'bảng tra'}'). Kiểm lại ánh xạ cột/khoá nối."
        )
    frame = pd.concat(pieces, ignore_index=True)
    if verbose:
        kept_note = (f"{len(skill_of_key):,} mục tra" if skill_of_key is not None else f"cột '{col_skill}'")
        print(f"  đã quét {n_raw:,} dòng, giữ {n_kept:,} dòng hợp lệ | khái niệm: {len(maps['skill_id'].values):,} ({kept_note})")

    # Thời gian thiếu/hợp lệ -> thay bằng vị trí dòng (đơn điệu theo thứ tự file), mỗi dòng cách nhau 1 giây
    bad_ts = frame["timestamp_ms"].isna() | (frame["timestamp_ms"] <= 0)
    if bad_ts.any():
        filler = np.arange(len(frame), dtype="float64") * 1000.0
        frame.loc[bad_ts, "timestamp_ms"] = filler[bad_ts.to_numpy()]

    cap = int(top_concepts if top_concepts is not None else (spec.top_concepts or 0)) or None
    if cap is not None:
        frame = _cap_concepts(frame, cap, verbose=verbose)

    # Ghi attrs SAU cùng: pandas chỉ giữ attrs qua phép biến đổi "một-một", mọi phép lọc ở
    # trên có quyền xoá nó, và ensure_canonical cần correct_origin + bảng giải mã ngay khi
    # hàm này trả về (trước khi nó lọc người học).
    frame.attrs["correct_origin"] = origin
    frame.attrs["decoders"] = {c: m.decode_map() for c, m in maps.items()}
    frame.attrs["column_mapping"] = {
        "người học": col_user,
        "khái niệm": col_skill if skill_of_key is None else f"bảng tra theo {col_prob}",
        "câu hỏi": col_prob,
        "đúng/sai": origin,
        "thời điểm": ts_col,
        "thời gian làm bài": dur_col,
    }
    return frame


def _cap_concepts(frame: pd.DataFrame, cap: int, verbose: bool = True) -> pd.DataFrame:
    """Giữ ``cap`` khái niệm hay gặp nhất (thứ tự ưu tiên: số lượt giảm dần, rồi mã khái niệm)."""
    counts = frame.groupby("skill_id", sort=True).size()
    if len(counts) <= cap:
        if verbose:
            print(f"  khái niệm: {len(counts)} (dưới cap {cap}) — giữ nguyên")
        return frame
    keep = set(counts.sort_values(ascending=False, kind="mergesort").head(cap).index.tolist())
    before = len(frame)
    out = frame[frame["skill_id"].isin(keep)].reset_index(drop=True)
    if verbose:
        print(f"  giới hạn khái niệm: giữ {cap}/{len(counts)} khái niệm hay gặp nhất, "
              f"bỏ {before - len(out):,} lượt ({100 * (before - len(out)) / max(before, 1):.1f}%) "
              "trên khái niệm hiếm — GRKT không chạy nổi với K lớn hơn")
    return out


def _norm_name(x) -> str:
    # Bỏ HOA/thường TRƯỚC khi bỏ ký tự không phải chữ/số: ngược lại thì "IsCorrect" mất cả
    # chữ I lẫn C và khớp với bất kỳ tên nào ("sorrect").
    return re.sub(r"[^a-z0-9]", "", str(x).lower())


def _match_column(header, candidates, what, path, required: bool = True):
    """Khớp tên cột: chính xác trước, rồi khớp mờ (bỏ hoa/thường, khoảng trắng, gạch dưới).

    Khớp mờ là vì Kaggle đổi tên cột giữa các phiên bản dataset (và giữa tên công bố với tên
    thật trong file); nó chỉ mở rộng tập ứng viên ĐÃ khai báo, không tự đoán cột mới.
    """
    for c in candidates or ():
        if c in header:
            return c
    norm = {}
    for c in header:
        norm.setdefault(_norm_name(c), c)
    for c in candidates or ():
        hit = norm.get(_norm_name(c))
        if hit:
            return hit
    if required:
        raise KeyError(
            f"[{Path(path).name}] Không tìm thấy cột cho '{what}'. Ứng viên: {list(candidates or ())}. "
            f"Cột có sẵn: {list(header)}. Sửa DatasetSpec.columns rồi chạy lại (không đoán bừa)."
        )
    return None


def _resolve_columns(header, spec: DatasetSpec) -> dict:
    """Ánh xạ từng vai trò (user/skill/problem/...) sang cột thật trong ``header``.

    Bốn vai trò bắt buộc (người học, câu hỏi, thời điểm, độ dài) raise ngay khi không khớp,
    kèm danh sách cột thật của file; các vai trò còn lại (khái niệm, nhãn, số lần thử) được
    phép ``None`` vì còn đường bảng tra / đường suy nhãn.
    """
    required = ("user", "problem", "timestamp", "duration")
    resolved = {role: _match_column(header, spec.columns.get(role, []), role, spec.name,
                                    required=role in required)
                for role in spec.columns}
    read = set()
    for role in ("user", "problem", "timestamp", "duration", "skill", "correct", "answer_count"):
        col = resolved.get(role)
        if col:
            read.add(col)
    resolved["usecols"] = [c for c in header if c in read]
    return resolved


def _join_key(series: pd.Series):
    """Chuẩn hoá khoá nối: số thành Int64, chuỗi thành string đã cắt khoảng trắng.

    Bắt buộc dùng chung một kiểu cho hai bên nối — ``upid`` trong log là số nguyên còn trong
    bảng tra có thể là chuỗi, và dict tra thông thường sẽ coi ``5`` với ``'5'`` là hai khoá.
    """
    if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("Int64")
    text = series.astype("string").str.strip()
    as_num = pd.to_numeric(text, errors="coerce")
    if len(text) and float(as_num.notna().mean()) > 0.99:
        return as_num.astype("Int64")
    return text


def _correct_flags(chunk: pd.DataFrame, col_correct, col_ac):
    """Cột nhãn 0/1, hoặc suy từ số lần thử theo quy ước pyKT khi file không có nhãn."""
    if col_correct is not None:
        raw = chunk[col_correct]
        origin = col_correct
        if pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw):
            return pd.to_numeric(raw, errors="coerce"), origin
        text = raw.astype("string").str.strip().str.lower()
        truthy = {"true", "1", "1.0", "correct", "yes"}
        falsy = {"false", "0", "0.0", "incorrect", "no", "wrong"}
        return text.map(lambda v: 1.0 if v in truthy else (0.0 if v in falsy else np.nan)), origin
    attempts = pd.to_numeric(chunk[col_ac], errors="coerce")
    # Quy ước pyKT: làm đúng ngay lần đầu = đúng; cần >=3 lần = sai; 2 lần là nhập nhằng -> bỏ.
    return attempts.map(lambda a: np.nan if pd.isna(a) or a == 2 else float(a == 1)), f"answer_count ({col_ac})"


def _find_content_file(spec: DatasetSpec, content_path=None) -> Optional[Path]:
    if content_path is not None:
        return Path(content_path)
    if not spec.content_file and not spec.content_patterns:
        return None
    preferred = (spec.content_file,) if spec.content_file else ()
    patterns = tuple(spec.content_patterns or ()) + (spec.name,)
    return find_kaggle_input(patterns=patterns, preferred_names=preferred)


def _load_content_map(spec: DatasetSpec, raw_path: Path, col_prob: str, content_path=None,
                      verbose: bool = True):
    """Đọc bảng tra khái niệm (Info_Content.csv) -> Series khoá câu hỏi -> tên khái niệm.

    Không có bảng này thì không có khái niệm cho Junyi: ``Log_Problem.csv`` chỉ định danh câu
    hỏi (``upid``, 25.785 giá trị). Dùng ``upid`` làm khái niệm sẽ cho ma trận 25.786² —
    vượt ngoài khả năng của GPU lẫn của chính quy trình GRKT.
    """
    content = _find_content_file(spec, content_path)
    if content is None or not Path(content).exists():
        raise KeyError(
            f"[{raw_path.name}] File log không có cột khái niệm và KHÔNG tìm thấy bảng tra "
            f"'{spec.content_file}' trong /kaggle/input. Khái niệm của Junyi phải tra từ "
            f"Info_Content.csv theo upid; upid đơn thuần có 25.785 giá trị nên không dùng làm "
            "khái niệm cho GRKT được.\n"
            f"  Cách sửa: Add input -> chọn ĐÚNG dataset 'junyiacademy/learning-activity-public-"
            "dataset-by-junyi-academy' (attach cả zip 3 file, không phải riêng Log_Problem.csv), "
            "hoặc truyền content_path='/kaggle/input/<thư-mục>/Info_Content.csv' vào "
            "ensure_canonical().\n"
            f"  Hiện trạng input: {list_kaggle_inputs()}"
        )
    cheader = _probe_header(content)
    key = _match_column(cheader, [col_prob, *(spec.content_columns.get("key") or [])],
                        "khoá nối bảng tra", content)
    skill = _match_column(cheader, spec.content_columns.get("skill"), "tên khái niệm", content,
                          required=False)
    if skill is None:
        fuzzy = [c for c in cheader
                 if any(w in _norm_name(c) for w in ("exercis", "name", "title", "skill", "topic",
                                                     "unit", "kc", "chapter", "categ", "tags"))]
        raise KeyError(
            f"[{content.name}] Không khớp cột khái niệm nào trong "
            f"{list(spec.content_columns.get('skill') or ())}. Cột có sẵn: {cheader}. "
            f"Cột có vẻ là khái niệm: {fuzzy}. Thêm đúng tên vào JUNYI.content_columns['skill'] rồi chạy lại."
        )
    table = pd.read_csv(content, usecols=[key, skill], low_memory=False, encoding="utf-8-sig")
    keys = _join_key(table[key])
    vals = table[skill].astype("string").str.strip()
    bad = keys.isna() | vals.isna() | (vals.str.len() == 0)
    table = pd.DataFrame({"k": keys[~bad], "v": vals[~bad]}).drop_duplicates("k", keep="first")
    if table.empty:
        raise KeyError(f"[{content.name}] Bảng tra rỗng sau khi lọc: key='{key}', skill='{skill}'.")
    skill_of_key = table.set_index("k")["v"]

    # Kiểm tra khoá nối có thật sự khớp với log không, trên một mẫu nhỏ (đọc 200k dòng đầu).
    sample = pd.read_csv(raw_path, usecols=[col_prob], nrows=200_000, encoding="utf-8-sig")
    coverage = float(_join_key(sample[col_prob]).isin(skill_of_key.index).mean())
    if verbose:
        print(f"  bảng tra: {content.name} | khoá '{key}' -> khái niệm '{skill}' | "
              f"{len(skill_of_key):,} mục | {skill_of_key.nunique():,} khái niệm khác nhau")
        print(f"  phủ {coverage:.1%} lượt làm bài (đo trên 200.000 dòng đầu của {raw_path.name})")
    if coverage < 0.5:
        raise KeyError(
            f"[{content.name}] Nối theo khoá '{key}' chỉ phủ {coverage:.1%} lượt làm bài — nhiều "
            "khả năng khoá hoặc cột khái niệm đã bị chọn sai. Cột của bảng tra: "
            f"{cheader}. Cột của log: {_probe_header(raw_path)}. "
            "Sửa JUNYI.content_columns rồi chạy lại."
        )
    return skill_of_key, {"file": str(content), "key": key, "skill": skill,
                          "n_entries": int(len(skill_of_key)),
                          "n_concepts": int(skill_of_key.nunique()), "coverage": coverage}


class _CodeMap:
    """Gom giá trị (số hoặc chuỗi) thành mã nguyên tố liên tục để giữ RAM khi quét file lớn.

    ``map(dict)`` trả NaN cho khoá chưa biết, nên mỗi mảnh chỉ phải ``unique()`` trên các giá
    trị chưa thấy — không vòng lặp Python trên 16 triệu dòng.
    """

    def __init__(self, name: str):
        self.name = name
        self._lookup: dict = {}
        self.values: list = []

    def encode(self, series: pd.Series) -> pd.Series:
        codes = series.map(self._lookup)
        missing = codes.isna()
        if bool(missing.any()):
            for value in series[missing].unique():
                self._lookup[value] = len(self.values)
                self.values.append(value)
            codes = series.map(self._lookup)
        return codes.astype("int64")

    def decode_map(self) -> dict:
        return {i: v for i, v in enumerate(self.values)}


def has_canonical(spec: DatasetSpec, out_dir: Path) -> bool:
    """CSV chuẩn hoá đã tồn tại (ở out_dir hoặc chỗ resolve_canonical_path tìm thấy)?"""
    if (Path(out_dir) / Path(spec.canonical_file).name).exists():
        return True
    try:
        resolve_canonical_path(spec, search_dirs=[out_dir])
        return True
    except FileNotFoundError:
        return False


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
    (dùng khi smoke test trên Kaggle với ngân sách GPU nhỏ). ``content_path`` truyền thẳng bảng
    tra khái niệm khi ``find_kaggle_input`` dò sai (chỉ cần thiết với tập mà file log không có
    cột khái niệm, vd Junyi).
    """
    out_dir = Path(out_dir) if out_dir is not None else (Path(spec.canonical_file).parent or Path("."))
    if raw_path is None and spec.raw_file is None and not has_canonical(spec, out_dir):
        # Tự dò file gốc trong /kaggle/input: notebook không phải truyền đường dẫn, và lỗi
        # "quên attach dataset" được báo ở ngay cell cấu hình thay vì cell chuẩn hoá.
        found = find_kaggle_input(patterns=(spec.name, "junyi", "log_problem"))
        if found is not None:
            raw_path = found
            if verbose:
                print(f"[{spec.name}] Tự dò thấy dữ liệu gốc trong /kaggle/input: {raw_path}")
        elif verbose and spec.name == "junyi":
            print(f"  [dò] {list_kaggle_inputs()}")
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
            print(f"[{spec.name}] Dùng trực tiếp CSV chuẩn hoá có sẵn: {path}")
        return path

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(spec.canonical_file).name
    if target.exists():
        if verbose:
            print(f"[{spec.name}] Đã có CSV chuẩn hoá, dùng lại: {target} "
                  "(xoá file này nếu đổi max_users/min_interactions/top_concepts hoặc nếu dữ "
                  "liệu nguồn thay đổi)")
        return target

    raw = Path(raw_path if raw_path is not None else spec.raw_file)
    if not raw.exists():
        raise FileNotFoundError(
            f"Không thấy file gốc của {spec.name}: {raw}. Với dữ liệu Kaggle, attach dataset "
            f"rồi truyền raw_path='/kaggle/input/<thư-mục-dataset>/<file>.csv'."
        )

    min_int = int(min_interactions if min_interactions is not None else spec.min_interactions)
    cap_users = int(max_users if max_users is not None else (spec.max_users or 0)) or None
    cap_concepts = int(top_concepts if top_concepts is not None else (spec.top_concepts or 0)) or None

    if verbose:
        print(f"[{spec.name}] Chuẩn hoá {raw.name} -> {target.name} "
              f"(min {min_int} tương tác/người, cap {'toàn bộ' if cap_users is None else cap_users} người, "
              f"cap {'toàn bộ' if cap_concepts is None else cap_concepts} khái niệm)")

    frame = _canonicalize_junyi(raw, spec, content_path=content_path,
                                top_concepts=cap_concepts, verbose=verbose)
    # Đọc attrs NGAY tại đây: mọi phép sắp xếp/lọc bên dưới đều có quyền bỏ attrs của pandas.
    correct_origin = frame.attrs.get("correct_origin", "không rõ")
    mapping = frame.attrs.get("column_mapping", {})
    decoders = frame.attrs.get("decoders", {})
    frame = frame.sort_values(["user_id", "timestamp_ms"], kind="mergesort")

    counts = frame.groupby("user_id", sort=True).size()
    keep = counts[counts >= min_int]
    if verbose:
        print(f"  người học sau lọc tối thiểu tương tác: {len(keep)} / {len(counts)}")
    if cap_users is not None and len(keep) > cap_users:
        rng = np.random.RandomState(spec.subsample_seed)
        chosen = np.sort(rng.choice(np.asarray(keep.index), size=cap_users, replace=False))
        keep = keep.loc[list(chosen)]
        if verbose:
            print(f"  sub-sample theo seed {spec.subsample_seed}: {len(keep)} người học")
    frame = frame[frame["user_id"].isin(keep.index)]

    # Cắt chuỗi mỗi người theo thứ tự thời gian trước khi ghi file (giữ thời gian thật của phần còn lại)
    frame = frame.groupby("user_id", sort=False, group_keys=False).head(spec.max_keep_per_user)
    # Giải mã sau cùng, khi khung đã nhỏ: 16 triệu chuỗi khái niệm giữ trong RAM là lý do
    # notebook Kaggle chết bằng "Killed" ở lần chạy trước.
    for column, decoding in decoders.items():
        if column in frame.columns:
            frame[column] = frame[column].map(decoding)
    if "skill_id" in frame.columns:
        frame["skill_name"] = frame["skill_id"]
    frame = frame.sort_values(["user_id", "timestamp_ms"], kind="mergesort").reset_index(drop=True)
    frame["order_id"] = np.arange(1, len(frame) + 1, dtype="int64")

    # Khoảng thời gian giữa hai bước của cùng người học phải không âm và bị chặn ở bước
    # build_grkt_data; ở đây chỉ ghi nguyên vẹn timestamp để tái sử dụng được nhiều cấu hình.
    frame[CANONICAL_COLUMNS].to_csv(target, index=False)
    if verbose:
        if mapping:
            print("  ánh xạ cột: " + " | ".join(f"{k}={v}" for k, v in mapping.items()))
        else:
            print(f"  nhãn đúng/sai lấy từ: {correct_origin}")
        print(f"  đã ghi {len(frame)} tương tác | {frame['user_id'].nunique()} người học | "
              f"{frame['skill_id'].nunique()} khái niệm | {frame['problem_id'].nunique()} câu hỏi")
        print(f"  tỉ lệ đúng = {frame['correct'].mean():.3f}")
    return target


def find_kaggle_input(patterns=("junyi",), preferred_names=("Log_Problem.csv",)) -> Optional[Path]:
    """ dò file Junyi trong /kaggle/input, ưu tiên tên quen thuộc; None nếu chưa attach."""
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    hits = [p for p in root.rglob("*.csv") if any(k in str(p).lower() for k in patterns)]
    if not hits:
        return None
    for name in preferred_names:
        for p in hits:
            if p.name.lower() == name.lower():
                return p
    return max(hits, key=lambda p: p.stat().st_size)


def list_kaggle_inputs(max_files: int = 40) -> str:
    """Mô tả ngắn những gì THỰC SỰ có trong /kaggle/input, dùng trong thông báo lỗi.

    Phân biệt được ba tình huống mà trước đây đều ra cùng một FileNotFoundError: chưa attach
    gì, attach rồi nhưng Kaggle không giải nén (dataset phân mảnh/thiếu quyền), và attach đúng
    nhưng tên file/thư mục khác mẫu dò.
    """
    root = Path("/kaggle/input")
    if not root.exists():
        return "không có /kaggle/input (không phải môi trường Kaggle)."
    try:
        mounts = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError as exc:
        return f"không đọc được /kaggle/input: {exc}"
    if not mounts:
        return "/kaggle/input RỖNG — chưa attach dataset nào (Add input > Datasets)."
    lines = [f"/kaggle/input có {len(mounts)} dataset đã attach:"]
    for m in mounts:
        try:
            files = sorted(m.rglob("*"), key=lambda p: p.stat().st_size if p.is_file() else -1, reverse=True)
            csvs = [p for p in files if p.is_file() and p.suffix.lower() == ".csv"]
            size = sum(p.stat().st_size for p in csvs) / 1024 / 1024
            shown = ", ".join(f"{p.relative_to(m)} ({p.stat().st_size / 1024 / 1024:.0f} MB)"
                             for p in (csvs or files)[:5])
            lines.append(f"  {m.name}: {len(csvs)} file CSV, {size:,.0f} MB | {shown or '(trống — chưa giải nén?)'}")
            if len(csvs) > 5:
                lines.append(f"      ... còn lại {len(csvs) - 5} file CSV khác")
        except OSError as exc:
            lines.append(f"  {m.name}: không liệt kê được ({exc})")
    return "\n".join(lines[:max_files])
