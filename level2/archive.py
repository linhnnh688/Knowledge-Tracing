"""Đóng gói thư mục kết quả Cấp độ 2 thành một zip để tải về và resume phiên Kaggle sau.

Notebook chỉ cần gọi :func:`package_output` ở hai chỗ: cell §3 (tự đóng gói khi
:class:`~level2.grkt_api.TrainingPaused`) và cell §8.5 (đóng gói thủ công bất kỳ lúc nào).
Việc đóng gói tự động ở §3 là thứ giữ lại hàng giờ GPU: Kaggle xoá ``/kaggle/working``
khi phiên kết thúc, nên không có zip trước lúc đó là mất toàn bộ checkpoint.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

#: File tái sinh được (``ensure_canonical``) nên không cần mang theo: Junyi ~1 GB.
DEFAULT_SKIP: frozenset = frozenset({"junyi_canonical.csv", "junyi_canonical.csv.meta.json"})


def is_resume_ckpt(name: str) -> bool:
    """Checkpoint cần cho resume: giữa epoch (``ckpt_*.pt``) hoặc GRKT đã train xong."""
    return (name.startswith("ckpt_") or name.startswith("GRKT-")) and name.endswith(".pt")


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
    """
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Không có thư mục output {save_dir} để đóng gói")
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    zip_real = zip_path.resolve()

    added, skipped = [], []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(save_dir.rglob("*")):
            if not p.is_file() or p.resolve() == zip_real:
                continue
            rel = str(p.relative_to(save_dir.parent))   # "<tên save_dir>/<dataset>/<file>"
            mb = p.stat().st_size / 1024 / 1024
            if p.name in skip_names:
                skipped.append((rel, mb, "bỏ theo SKIP_NAMES (tái sinh được)"))
                continue
            # .tmp còn lại khi phiên bị kill giữa lúc ghi checkpoint — không phải sản phẩm.
            if p.suffix == ".tmp":
                continue
            if (not keep_ckpts) and p.suffix in (".pt", ".npz"):
                skipped.append((rel, mb, "KEEP_CKPTS=False"))
                continue
            if mb > max_mb and not is_resume_ckpt(p.name):
                skipped.append((rel, mb, f"lớn hơn MAX_MB={max_mb}"))
                continue
            zf.write(p, arcname=rel)
            added.append((rel, mb))

    if verbose:
        print(f">>> {zip_path} | {zip_path.stat().st_size / 1024 / 1024:.1f} MB "
              f"(nén từ {len(added)} file, {sum(m for _, m in added):.1f} MB chưa nén)")
        with zipfile.ZipFile(zip_path) as check:
            bad = check.testzip()
        print("    kiểm tra toàn vẹn zip:", "OK" if bad is None else f"HỎNG tại {bad}")
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
        print(f"    checkpoint resume trong zip: {n_resume}"
              + ("" if n_resume else "  [cảnh báo] không có -> phiên sau KHÔNG resume được"))
        if skipped:
            print(f"    bỏ {len(skipped)} file (tăng max_mb / sửa skip_names / keep_ckpts nếu muốn lấy cả):")
            for rel, mb, why in skipped:
                print(f"      {rel:52s} {mb:9.1f} MB | {why}")
    return zip_path
