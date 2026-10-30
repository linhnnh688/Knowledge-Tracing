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
"""

from __future__ import annotations

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
    max_users=4000,
    columns={
        "user": ["user_id", "Anon Student Id", "student_id"],
        "skill": ["exercise", "skill", "list_skills", "tags", "KC (Exercise)"],
        "problem": ["problem", "problem_id", "assistment_id", "problem_name"],
        "correct": ["correct", "Outcome"],
        "answer_count": ["answer_count", "count_attempts"],
        "timestamp": ["timestamp_ms", "time_done", "time", "Timestamp"],
        "duration": ["ms_first_response", "answer_time", "time_taken"],
    },
    note=(
        "Junyi Academy — Kaggle 'junyiacademy/learning-activity-public-dataset-by-junyi-academy' "
        "(16,2 triệu lượt làm bài / 72.630 người học, 2018/08–2019/07, CC BY-NC-SA, phải bấm "
        "agree/subscribe một lần rồi attach vào notebook). Khái niệm = 'exercise'; đúng/sai lấy "
        "từ cột 'correct' nếu có, nếu không thì suy từ 'answer_count' (1 -> đúng, >=3 -> sai, "
        "bỏ 2 — quy ước pyKT)."
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


def _pick(columns, candidates, what: str, path) -> str:
    for c in candidates:
        if c in columns:
            return c
    raise KeyError(
        f"[{Path(path).name}] Không tìm thấy cột cho '{what}'. Cột có sẵn: {list(columns)}. "
        f"Sửa DatasetSpec.columns rồi chạy lại (không đoán bừa)."
    )


def _to_duration_ms(series: pd.Series) -> pd.Series:
    """Suy ra miligiây từ một cột độ dài/độ trễ; tự đoán ms so với giây."""
    values = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0)
    positive = values[values > 0]
    # Đơn vị tính theo_median: trên 500 thì chỉ có thể là miligiây (500 GIÂY = 8 phút cho một
    # lượt làm bài là vô lý); dưới 500 thì là giây và phải nhân lên. ASSIST ms_first_response
    # median ~2.5e4 -> giữ nguyên; Junyi answer_time/time_taken (giây) -> nhân 1000.
    if positive.empty:
        unit = 1000.0
    else:
        unit = 1.0 if positive.median() > 500 else 1000.0
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


def _canonicalize_junyi(raw_path: Path, spec: DatasetSpec) -> pd.DataFrame:
    # Thăm header trước rồi CHỈ đọc các cột nằm trong danh sách ứng viên của spec. File Junyi
    # (Log_Problem.csv) có ~20 cột và hàng chục triệu dòng; đọc toàn bộ rồi bỏ sẽ ngốn hết RAM
    # của notebook Kaggle trước khi kịp chuẩn hoá.
    header = list(pd.read_csv(raw_path, nrows=0, encoding="utf-8-sig").columns)
    candidates = {name for names in spec.columns.values() for name in names}
    usecols = [c for c in header if c in candidates]
    if not usecols:
        raise KeyError(
            f"[{raw_path.name}] Không khớp cột nào trong {sorted(candidates)}. "
            f"Cột có sẵn trong file: {header}. Sửa DatasetSpec.columns rồi chạy lại."
        )
    print(f"  đọc {len(usecols)}/{len(header)} cột: {usecols}")
    df = pd.read_csv(raw_path, usecols=usecols, low_memory=False, encoding="utf-8-sig")
    print(f"  đã đọc {len(df):,} dòng từ {raw_path.name}")
    cols = list(df.columns)
    col_user = _pick(cols, spec.columns["user"], "người học", raw_path)
    col_skill = _pick(cols, spec.columns["skill"], "khái niệm", raw_path)
    col_prob = _pick(cols, spec.columns["problem"], "câu hỏi", raw_path)
    ts_col = _pick(cols, spec.columns["timestamp"], "thời điểm", raw_path)
    dur_col = _pick(cols, spec.columns["duration"], "thời gian làm bài", raw_path)

    out = pd.DataFrame(index=df.index)
    out["user_id"] = df[col_user]
    out["skill_id"] = df[col_skill].astype("string")
    out["problem_id"] = df[col_prob].astype("string")
    out["timestamp_ms"] = _to_epoch_ms(df[ts_col])
    out["ms_first_response"] = _to_duration_ms(df[dur_col])

    col_correct = col_correct_real(cols, spec)
    if col_correct is not None:
        raw = df[col_correct]
        if pd.api.types.is_numeric_dtype(raw):
            out["correct"] = pd.to_numeric(raw, errors="coerce")
        else:
            text = raw.astype("string").str.strip().str.lower()
            truthy = {"true", "1", "1.0", "correct", "yes"}
            falsy = {"false", "0", "0.0", "incorrect", "no", "wrong"}
            out["correct"] = text.map(lambda v: 1.0 if v in truthy else (0.0 if v in falsy else np.nan))
        origin = col_correct
    else:
        col_ac = _pick(cols, spec.columns["answer_count"], "số lần thử", raw_path)
        attempts = pd.to_numeric(df[col_ac], errors="coerce")
        # Quy ước pyKT: làm đúng ngay lần đầu = đúng; cần >=3 lần = sai; 2 lần là nhập nhằng -> bỏ.
        out["correct"] = attempts.map(lambda a: np.nan if pd.isna(a) or a == 2 else float(a == 1))
        origin = f"answer_count ({col_ac})"
    out = out[out["correct"].isin([0.0, 1.0])]
    out = out[out["skill_id"].notna() & (out["skill_id"].str.len() > 0)]
    out = out.reset_index(drop=True)
    out["correct"] = out["correct"].astype("int64")
    out["skill_name"] = out["skill_id"]

    # Thời gian thiếu/hợp lệ -> thay bằng vị trí dòng (đơn điệu theo thứ tự file), mỗi dòng cách nhau 1 giây
    bad_ts = out["timestamp_ms"].isna() | (out["timestamp_ms"] <= 0)
    if bad_ts.any():
        filler = np.arange(len(out), dtype="float64") * 1000.0
        out.loc[bad_ts, "timestamp_ms"] = filler[bad_ts.to_numpy()]

    # Ghi attrs SAU cùng: pandas chỉ giữ attrs qua phép biến đổi "một-một", mọi phép lọc ở
    # trên có quyền xoá nó, và notebook cần correct_origin để in ra nguồn nhãn đúng/sai.
    out.attrs["correct_origin"] = origin
    return out


def col_correct_real(cols, spec: DatasetSpec) -> Optional[str]:
    """Trả về cột nhãn đúng/sai nếu có, ngược lại None (khi đó phải suy từ answer_count)."""
    for c in spec.columns.get("correct", []):
        if c in cols:
            return c
    return None


def _has_canonical(spec: DatasetSpec, out_dir: Path) -> bool:
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
    verbose: bool = True,
) -> Path:
    """Trả về đường dẫn CSV chuẩn hoá; nếu spec có file gốc thì tạo file chuẩn hoá một lần.

    ``max_users`` / ``min_interactions`` cho phép ghi đè cấu hình trong spec (dùng khi
    smoke test trên Kaggle với ngân sách GPU nhỏ).
    """
    out_dir = Path(out_dir) if out_dir is not None else (Path(spec.canonical_file).parent or Path("."))
    if raw_path is None and spec.raw_file is None and not _has_canonical(spec, out_dir):
        # Tự dò file gốc trong /kaggle/input: notebook không phải truyền đường dẫn, và lỗi
        # "quên attach dataset" được báo ở ngay cell cấu hình thay vì cell chuẩn hoá.
        found = find_kaggle_input(patterns=(spec.name, "junyi"))
        if found is not None:
            raw_path = found
            if verbose:
                print(f"[{spec.name}] Tự dò thấy dữ liệu gốc trong /kaggle/input: {raw_path}")
    if spec.raw_file is None and raw_path is None:
        try:
            path = resolve_canonical_path(spec, search_dirs=[out_dir])
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{spec.name} cần file dữ liệu gốc: attach dataset vào notebook rồi truyền "
                f"raw_path='/kaggle/input/<dataset>/<file>.csv' (xem level2.dataset_spec.find_kaggle_input()). "
                f"Chi tiết: {exc}"
            ) from exc
        if verbose:
            print(f"[{spec.name}] Dùng trực tiếp CSV chuẩn hoá có sẵn: {path}")
        return path

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / Path(spec.canonical_file).name
    if target.exists():
        if verbose:
            print(f"[{spec.name}] Đã có CSV chuẩn hoá, dùng lại: {target}")
        return target

    raw = Path(raw_path if raw_path is not None else spec.raw_file)
    if not raw.exists():
        raise FileNotFoundError(
            f"Không thấy file gốc của {spec.name}: {raw}. Với dữ liệu Kaggle, attach dataset "
            f"rồi truyền raw_path='/kaggle/input/<thư-mục-dataset>/<file>.csv'."
        )

    min_int = int(min_interactions if min_interactions is not None else spec.min_interactions)
    cap_users = int(max_users if max_users is not None else (spec.max_users or 0)) or None

    if verbose:
        print(f"[{spec.name}] Chuẩn hoá {raw.name} -> {target.name} "
              f"(min {min_int} tương tác/người, cap {'toàn bộ' if cap_users is None else cap_users} người)")

    frame = _canonicalize_junyi(raw, spec)
    correct_origin = frame.attrs.get("correct_origin", "không rõ")
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
    frame = frame.sort_values(["user_id", "timestamp_ms"], kind="mergesort").reset_index(drop=True)
    frame["order_id"] = np.arange(1, len(frame) + 1, dtype="int64")

    # Khoảng thời gian giữa hai bước của cùng người học phải không âm và bị chặn ở bước
    # build_grkt_data; ở đây chỉ ghi nguyên vẹn timestamp để tái sử dụng được nhiều cấu hình.
    frame[CANONICAL_COLUMNS].to_csv(target, index=False)
    if verbose:
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
