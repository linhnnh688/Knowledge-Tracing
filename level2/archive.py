"""Đóng gói thư mục kết quả Cấp độ 2 thành một zip để tải về và resume phiên Kaggle sau.

Notebook gọi :func:`package_output` ở cell §3 (khi :class:`~level2.grkt_api.TrainingPaused`
hoặc lỗi) và cell §12 (đóng gói thủ công). Ngoài ra :class:`PeriodicPackager` chạy nền để
**luôn** có zip mới trong suốt phiên: Kaggle xoá ``/kaggle/working`` khi phiên kết thúc và
có thể kill phiên cứng (hết quota / rớt kết nối / quá 12 h) mà không chạy khối ``except``,
nên chỉ đóng gói lúc dừng vẫn còn một cửa mất checkpoint.

Zip được ghi nguyên tử (``<tên>.part`` rồi :func:`os.replace`): phiên bị kill giữa lúc đóng
gói chỉ mất file ``.part``, zip tốt của lần trước vẫn còn nguyên để tải về.
"""

from __future__ import annotations

import os
import threading
import time
import zipfile
from pathlib import Path
from typing import Optional

#: Mặc định không loại gì. CSV chuẩn hoá (``*_canonical.csv`` + ``.meta.json``) PHẢI nằm
#: trong zip: có nó thì phiên sau đặt ``EVAL_ONLY=True`` là đánh giá được ngay, không cần
#: attach lại Junyi thô ~3 GB rồi chuẩn hoá lại hàng chục phút. File quá lớn vẫn bị
#: ``max_mb`` loại, nhưng lý do in ra khi đó nói rõ hậu quả.
DEFAULT_SKIP: frozenset = frozenset()

#: Bản zip "nhẹ" chỉ chứa số liệu + ảnh (cell §12): bỏ CSV chuẩn hoá vì người đọc báo cáo
#: không cần dữ liệu, và bản chạy toàn bộ Junyi có thể tới ~1 GB.
LIGHT_SKIP: frozenset = frozenset(
    {"junyi_canonical.csv", "junyi_canonical.csv.meta.json"}
)

#: Đuôi tên cho biết file là dữ liệu đã chuẩn hoá mà ``EVAL_ONLY`` cần.
_EVAL_ONLY_INPUTS = ("_canonical.csv", "_canonical.csv.meta.json")

_PART_SUFFIX = ".part"
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict = {}


def _lock_for(zip_real: str) -> threading.Lock:
    """Khoá theo đường dẫn zip để luồng nền và lời gọi thủ công không ghi đè nhau."""
    with _LOCKS_GUARD:
        lock = _LOCKS.get(zip_real)
        if lock is None:
            lock = _LOCKS[zip_real] = threading.Lock()
        return lock


def is_resume_ckpt(name: str) -> bool:
    """Checkpoint cần cho resume: giữa epoch (``ckpt_*.pt``) hoặc GRKT đã train xong."""
    return (name.startswith("ckpt_") or name.startswith("GRKT-")) and name.endswith(
        ".pt"
    )


def is_eval_only_input(name: str) -> bool:
    """File dữ liệu đã chuẩn hoá — thiếu nó thì ``EVAL_ONLY`` phải quay lại dữ liệu gốc."""
    return any(name.endswith(suf) for suf in _EVAL_ONLY_INPUTS)


def tree_signature(save_dir: Path) -> tuple:
    """Vân tay rẻ của cây thư mục (số file, tổng byte, mtime mới nhất).

    Dùng để bỏ qua lần đóng gói định kỳ khi không có gì mới: nén lại ~150 MB mất vài phút
    CPU, mà đó chính là CPU của dataloader.
    """
    n, total, newest = 0, 0, 0.0
    for p in save_dir.rglob("*"):
        if not p.is_file() or p.suffix in (_PART_SUFFIX, ".tmp"):
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        n += 1
        total += st.st_size
        newest = max(newest, st.st_mtime)
    return (n, total, round(newest, 3))


def package_output(
    save_dir,
    zip_path,
    *,
    max_mb: float = 150,
    keep_ckpts: bool = True,
    skip_names=DEFAULT_SKIP,
    verbose: bool = True,
) -> Path:
    """Gộp ``save_dir`` (giữ tiền tố ``<tên thư mục>/``) thành MỘT zip tại ``zip_path``.

    ``max_mb`` bỏ các file lớn hơn (trừ checkpoint resume — thiếu chúng là mất khả năng
    resume); ``keep_ckpts=False`` bỏ cả ``.pt``/``.npz`` để zip chỉ còn số liệu và ảnh.

    Ghi nguyên tử: zip cũ chỉ bị thay bằng zip mới đã đóng xong và qua kiểm tra toàn vẹn.
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Không có thư mục output {save_dir} để đóng gói")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_real = zip_path.resolve()

    with _lock_for(str(zip_real)):
        part = zip_path.with_name(zip_path.name + _PART_SUFFIX)
        # .part sót lại của lần đóng gói bị kill — không phải sản phẩm, xoá trước khi ghi.
        if part.exists():
            part.unlink()

        added, skipped, unstable = [], [], []
        with zipfile.ZipFile(
            part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            for p in sorted(save_dir.rglob("*")):
                if not p.is_file() or p.resolve() == zip_real:
                    continue
                rel = str(
                    p.relative_to(save_dir.parent)
                )  # "<tên save_dir>/<dataset>/<file>"
                try:
                    mb = p.stat().st_size / 1024 / 1024
                except OSError:
                    unstable.append((rel, "biến mất khi đang quét"))
                    continue
                if p.name in skip_names:
                    skipped.append((rel, mb, "bỏ theo SKIP_NAMES"))
                    continue
                # .tmp/.part còn lại khi phiên bị kill giữa lúc ghi — không phải sản phẩm.
                if p.suffix in (".tmp", _PART_SUFFIX):
                    continue
                if (not keep_ckpts) and p.suffix in (".pt", ".npz"):
                    skipped.append((rel, mb, "KEEP_CKPTS=False"))
                    continue
                if mb > max_mb and not is_resume_ckpt(p.name):
                    why = f"lớn hơn MAX_MB={max_mb}"
                    if is_eval_only_input(p.name):
                        why += " — EVAL_ONLY sẽ phải chuẩn hoá lại từ dữ liệu gốc"
                    skipped.append((rel, mb, why))
                    continue
                try:
                    # Checkpoint được ghi nguyên tử nên có thể bị replace ngay giữa lúc nén;
                    # bỏ qua file đó thay vì làm hỏng cả lần đóng gói.
                    zf.write(p, arcname=rel)
                except OSError as exc:
                    unstable.append((rel, f"{type(exc).__name__}: {exc}"))
                    continue
                added.append((rel, mb))

        with zipfile.ZipFile(part) as check:
            bad = check.testzip()
        if bad is not None:
            part.unlink(missing_ok=True)
            raise RuntimeError(f"zip tạm {part.name} hỏng tại {bad} — giữ zip cũ")
        os.replace(part, zip_path)

    if verbose:
        print(
            f">>> {zip_path} | {zip_path.stat().st_size / 1024 / 1024:.1f} MB "
            f"(nén từ {len(added)} file, {sum(m for _, m in added):.1f} MB chưa nén)"
        )
        groups: dict = {}
        for rel, mb in added:
            parts = rel.split("/")
            key = parts[1] if len(parts) > 1 else parts[0]
            n, tot = groups.get(key, (0, 0.0))
            groups[key] = (n + 1, tot + mb)
        print("    bên trong zip:")
        for key, (n, tot) in sorted(groups.items()):
            print(f"      {key:34s} {n:4d} file  {tot:9.1f} MB")
        n_resume = sum(1 for rel, _ in added if is_resume_ckpt(rel.rsplit("/", 1)[-1]))
        print(
            f"    checkpoint resume trong zip: {n_resume}"
            + (
                ""
                if n_resume
                else "  [cảnh báo] không có -> phiên sau KHÔNG resume được"
            )
        )
        n_eval = sum(
            1 for rel, _ in added if is_eval_only_input(rel.rsplit("/", 1)[-1])
        )
        print(
            f"    dữ liệu đã chuẩn hoá trong zip: {n_eval}"
            + (
                ""
                if n_eval
                else "  [cảnh báo] EVAL_ONLY sẽ phải attach dữ liệu gốc để chuẩn hoá lại"
            )
        )
        if unstable:
            print(f"    bỏ qua {len(unstable)} file đang được ghi song song:")
            for rel, why in unstable:
                print(f"      {rel:52s} | {why}")
        if skipped:
            print(
                f"    bỏ {len(skipped)} file (tăng max_mb / sửa skip_names / keep_ckpts nếu muốn lấy cả):"
            )
            for rel, mb, why in skipped:
                print(f"      {rel:52s} {mb:9.1f} MB | {why}")
    return zip_path


class PeriodicPackager:
    """Đóng gói ``save_dir`` -> ``zip_path`` mỗi ``every_min`` phút ở luồng nền (daemon).

    Lý do tồn tại: phiên Kaggle có thể bị kill cứng mà không kịp chạy khối ``except`` của
    cell §3. Luồng nền đảm bảo trên đĩa luôn có zip không cũ hơn ``every_min`` phút, nên
    tổn thất tối đa khi mất phiên là đúng một khoảng đó.

    Không bao giờ ném lỗi ra luồng chính: sự cố đóng gói chỉ in cảnh báo, vì mất một lần
    đóng gói vẫn tốt hơn là giết buổi train.
    """

    def __init__(
        self,
        save_dir,
        zip_path,
        *,
        every_min: float = 20.0,
        verbose: bool = False,
        pkg_kwargs: Optional[dict] = None,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.zip_path = Path(zip_path)
        # every_min <= 0 = TẮT đóng gói định kỳ (vẫn dùng được pack_now/stop).
        self.enabled = float(every_min) > 0
        self.every_s = max(60.0, float(every_min) * 60.0)
        self.verbose = verbose
        self.pkg_kwargs = dict(pkg_kwargs or {})
        self.n_packs = 0
        self.last_pack: Optional[float] = None
        self.last_error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._signature: Optional[tuple] = None

    # ------------------------------------------------------------------ điều khiển
    def start(self) -> "PeriodicPackager":
        if not self.enabled:
            print(
                "[packager] TẮT đóng gói định kỳ (every_min<=0) — zip chỉ được tạo khi "
                "gọi pack_now / đóng gói thủ công."
            )
            return self
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"packager-{self.zip_path.name}", daemon=True
        )
        self._thread.start()
        print(
            f"[packager] luồng nền: mỗi {self.every_s / 60:.0f} phút đóng gói "
            f"{self.save_dir.name} -> {self.zip_path.name} (ghi nguyên tử .part)"
        )
        return self

    def stop(
        self, *, final_pack: bool = True, reason: str = "dừng đóng gói định kỳ"
    ) -> None:
        """Dừng luồng nền; ``final_pack=True`` thì đóng gói một lần cuối trước khi về."""
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=600)
        if final_pack:
            self.pack_now(reason)
        print(
            f"[packager] đã dừng sau {self.n_packs} lần đóng gói"
            + (f" | lỗi cuối: {self.last_error}" if self.last_error else "")
        )

    def __enter__(self) -> "PeriodicPackager":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        reason = f"kết thúc khối ({exc_type.__name__})" if exc_type else "kết thúc khối"
        self.stop(final_pack=True, reason=reason)

    # ------------------------------------------------------------------ đóng gói
    def pack_now(self, reason: str = "") -> Optional[Path]:
        """Đóng gói ngay (bỏ qua kiểm tra "có gì mới không"). Không ném lỗi."""
        if not self.save_dir.is_dir():
            return None
        try:
            out = package_output(
                self.save_dir,
                self.zip_path,
                verbose=self.verbose,
                **self.pkg_kwargs,
            )
        except Exception as exc:  # luồng nền không được giết buổi train
            self.last_error = f"{type(exc).__name__}: {exc}"
            print(f"[packager] {reason or 'định kỳ'} THẤT BẠI: {self.last_error}")
            return None
        self.n_packs += 1
        self.last_pack = time.time()
        self._signature = tree_signature(self.save_dir)
        mb = out.stat().st_size / 1024 / 1024
        print(f"[packager] {reason or 'định kỳ'} -> {out.name} ({mb:.1f} MB)")
        return out

    def _loop(self) -> None:
        next_at = time.time() + self.every_s
        while not self._stop.is_set():
            # Ngủ từng nhịp ngắn để stop() không phải chờ hết chu kỳ.
            if self._stop.wait(timeout=min(15.0, max(0.0, next_at - time.time()))):
                return
            if time.time() < next_at:
                continue
            next_at = time.time() + self.every_s
            if not self.save_dir.is_dir():
                continue
            sig = tree_signature(self.save_dir)
            if sig == self._signature:
                print("[packager] không có file mới kể từ lần đóng gói trước -> bỏ qua")
                continue
            self.pack_now("định kỳ")
