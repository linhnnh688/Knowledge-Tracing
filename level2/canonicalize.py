"""Chuẩn hoá file log THÔ (hiện chỉ Junyi) về CSV chuẩn — tách từ ``dataset_spec.py``.

``dataset_spec.py`` giữ hợp đồng CSV chuẩn hoá, registry ``DatasetSpec`` và
``ensure_canonical``; module này chứa phần đọc/ghép/ép kiểu file thô theo mảnh (chunk), dài và
chỉ dùng cho tập phải chuẩn hoá lại. ``dataset_spec`` re-export các tên này cho code cũ.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from kt_common.log import get_logger

from .kaggle_io import _probe_header, find_kaggle_input, list_kaggle_inputs

if TYPE_CHECKING:  # chỉ dùng cho chú thích kiểu, tránh vòng import với dataset_spec
    from .dataset_spec import DatasetSpec

log = get_logger("level2.canonicalize")


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

def _canonicalize_junyi(
    raw_path: Path,
    spec: DatasetSpec,
    content_path=None,
    top_concepts=None,
    verbose: bool = True,
) -> pd.DataFrame:
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
    col_correct, col_ac, col_skill = (
        resolved["correct"],
        resolved["answer_count"],
        resolved["skill"],
    )
    if col_correct is None and col_ac is None:
        raise KeyError(
            f"[{raw_path.name}] Không có cột nhãn đúng/sai cũng không có cột số lần thử. "
            f"Ứng viên: {spec.columns.get('correct')} / {spec.columns.get('answer_count')}. "
            f"Cột có sẵn: {header}."
        )
    if verbose:
        log.info(f"  đọc {len(usecols)}/{len(header)} cột: {usecols}")
        log.info(
            f"  ánh xạ: user={col_user} | câu hỏi={col_prob} | thời điểm={ts_col} | "
            f"độ dài={dur_col} | nhãn={'(' + col_correct + ')' if col_correct else 'suy từ ' + col_ac}"
        )

    # Khái niệm: hoặc có sẵn trong log (bản junyi2015/PSLC có cột 'exercise'), hoặc phải nối
    # bảng tra Info_Content.csv theo định danh câu hỏi.
    skill_of_key = None
    if col_skill is None:
        skill_of_key, cinfo = _load_content_map(
            spec, raw_path, col_prob, content_path, verbose=verbose
        )
    elif verbose:
        log.info(f"  khái niệm lấy trực tiếp từ cột '{col_skill}' của file log")

    maps = {c: _CodeMap(c) for c in ("user_id", "skill_id", "problem_id")}
    pieces, n_raw, n_kept = [], 0, 0
    for chunk in pd.read_csv(
        raw_path,
        usecols=usecols,
        chunksize=max(1, int(spec.chunksize)),
        low_memory=False,
        encoding="utf-8-sig",
    ):
        n_raw += len(chunk)
        flags, origin = _correct_flags(chunk, col_correct, col_ac)
        skill = (
            chunk[col_skill].astype("string").str.strip()
            if skill_of_key is None
            else _join_key(chunk[col_prob]).map(skill_of_key).astype("string")
        )
        ok = (
            flags.isin([0.0, 1.0])
            & skill.notna()
            & (skill.str.len() > 0)
            & chunk[col_user].notna()
            & chunk[col_prob].notna()
        )
        ok = ok.to_numpy()
        if not ok.any():
            continue
        chunk, flags, skill = chunk[ok], flags[ok], skill[ok]
        piece = pd.DataFrame(
            {
                "user_id": maps["user_id"].encode(chunk[col_user]),
                "skill_id": maps["skill_id"].encode(skill),
                "problem_id": maps["problem_id"].encode(chunk[col_prob]),
                "correct": flags.astype("float64").astype("int8").to_numpy(),
                "timestamp_ms": _to_epoch_ms(chunk[ts_col]).to_numpy(dtype="float64"),
                "ms_first_response": _to_duration_ms(chunk[dur_col], dur_col).to_numpy(
                    dtype="int64"
                ),
            }
        )
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
        kept_note = (
            f"{len(skill_of_key):,} mục tra"
            if skill_of_key is not None
            else f"cột '{col_skill}'"
        )
        log.info(
            f"  đã quét {n_raw:,} dòng, giữ {n_kept:,} dòng hợp lệ | khái niệm: {len(maps['skill_id'].values):,} ({kept_note})"
        )

    # Thời gian thiếu/hợp lệ -> thay bằng vị trí dòng (đơn điệu theo thứ tự file), mỗi dòng cách nhau 1 giây
    bad_ts = frame["timestamp_ms"].isna() | (frame["timestamp_ms"] <= 0)
    if bad_ts.any():
        filler = np.arange(len(frame), dtype="float64") * 1000.0
        frame.loc[bad_ts, "timestamp_ms"] = filler[bad_ts.to_numpy()]

    cap = (
        int(top_concepts if top_concepts is not None else (spec.top_concepts or 0))
        or None
    )
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
            log.info(f"  khái niệm: {len(counts)} (dưới cap {cap}) — giữ nguyên")
        return frame
    keep = set(
        counts.sort_values(ascending=False, kind="mergesort").head(cap).index.tolist()
    )
    before = len(frame)
    out = frame[frame["skill_id"].isin(keep)].reset_index(drop=True)
    if verbose:
        log.info(
            f"  giới hạn khái niệm: giữ {cap}/{len(counts)} khái niệm hay gặp nhất, "
            f"bỏ {before - len(out):,} lượt ({100 * (before - len(out)) / max(before, 1):.1f}%) "
            "trên khái niệm hiếm — GRKT không chạy nổi với K lớn hơn"
        )
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
    resolved = {
        role: _match_column(
            header,
            spec.columns.get(role, []),
            role,
            spec.name,
            required=role in required,
        )
        for role in spec.columns
    }
    read = set()
    for role in (
        "user",
        "problem",
        "timestamp",
        "duration",
        "skill",
        "correct",
        "answer_count",
    ):
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
        return (
            text.map(lambda v: 1.0 if v in truthy else (0.0 if v in falsy else np.nan)),
            origin,
        )
    attempts = pd.to_numeric(chunk[col_ac], errors="coerce")
    # Quy ước pyKT: làm đúng ngay lần đầu = đúng; cần >=3 lần = sai; 2 lần là nhập nhằng -> bỏ.
    return (
        attempts.map(lambda a: np.nan if pd.isna(a) or a == 2 else float(a == 1)),
        f"answer_count ({col_ac})",
    )

def _find_content_file(spec: DatasetSpec, content_path=None) -> Optional[Path]:
    if content_path is not None:
        return Path(content_path)
    if not spec.content_file and not spec.content_patterns:
        return None
    preferred = (spec.content_file,) if spec.content_file else ()
    patterns = tuple(spec.content_patterns or ()) + (spec.name,)
    return find_kaggle_input(patterns=patterns, preferred_names=preferred)

def _load_content_map(
    spec: DatasetSpec,
    raw_path: Path,
    col_prob: str,
    content_path=None,
    verbose: bool = True,
):
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
    key = _match_column(
        cheader,
        [col_prob, *(spec.content_columns.get("key") or [])],
        "khoá nối bảng tra",
        content,
    )
    skill = _match_column(
        cheader,
        spec.content_columns.get("skill"),
        "tên khái niệm",
        content,
        required=False,
    )
    if skill is None:
        fuzzy = [
            c
            for c in cheader
            if any(
                w in _norm_name(c)
                for w in (
                    "exercis",
                    "name",
                    "title",
                    "skill",
                    "topic",
                    "unit",
                    "kc",
                    "chapter",
                    "categ",
                    "tags",
                )
            )
        ]
        raise KeyError(
            f"[{content.name}] Không khớp cột khái niệm nào trong "
            f"{list(spec.content_columns.get('skill') or ())}. Cột có sẵn: {cheader}. "
            f"Cột có vẻ là khái niệm: {fuzzy}. Thêm đúng tên vào JUNYI.content_columns['skill'] rồi chạy lại."
        )
    table = pd.read_csv(
        content, usecols=[key, skill], low_memory=False, encoding="utf-8-sig"
    )
    keys = _join_key(table[key])
    vals = table[skill].astype("string").str.strip()
    bad = keys.isna() | vals.isna() | (vals.str.len() == 0)
    table = pd.DataFrame({"k": keys[~bad], "v": vals[~bad]}).drop_duplicates(
        "k", keep="first"
    )
    if table.empty:
        raise KeyError(
            f"[{content.name}] Bảng tra rỗng sau khi lọc: key='{key}', skill='{skill}'."
        )
    skill_of_key = table.set_index("k")["v"]

    # Kiểm tra khoá nối có thật sự khớp với log không, trên một mẫu nhỏ (đọc 200k dòng đầu).
    sample = pd.read_csv(
        raw_path, usecols=[col_prob], nrows=200_000, encoding="utf-8-sig"
    )
    coverage = float(_join_key(sample[col_prob]).isin(skill_of_key.index).mean())
    if verbose:
        log.info(
            f"  bảng tra: {content.name} | khoá '{key}' -> khái niệm '{skill}' | "
            f"{len(skill_of_key):,} mục | {skill_of_key.nunique():,} khái niệm khác nhau"
        )
        log.info(
            f"  phủ {coverage:.1%} lượt làm bài (đo trên 200.000 dòng đầu của {raw_path.name})"
        )
    if coverage < 0.5:
        raise KeyError(
            f"[{content.name}] Nối theo khoá '{key}' chỉ phủ {coverage:.1%} lượt làm bài — nhiều "
            "khả năng khoá hoặc cột khái niệm đã bị chọn sai. Cột của bảng tra: "
            f"{cheader}. Cột của log: {_probe_header(raw_path)}. "
            "Sửa JUNYI.content_columns rồi chạy lại."
        )
    return skill_of_key, {
        "file": str(content),
        "key": key,
        "skill": skill,
        "n_entries": int(len(skill_of_key)),
        "n_concepts": int(skill_of_key.nunique()),
        "coverage": coverage,
    }

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
