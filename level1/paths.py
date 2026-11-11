import hashlib
import re
import subprocess
import sys
import urllib.request
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(*args) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root()), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def materialize_lfs(path) -> Path:
    """Replace a Git LFS pointer file with its real content (Kaggle clones have no git-lfs).

    *.csv / *.json are stored in LFS, so on Kaggle ``GKT/data/*.csv`` is a ~130-byte pointer.
    The object is downloaded from GitHub's media endpoint at the checked-out commit and
    verified against the pointer's sha256 and size. Non-pointer files are returned unchanged.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size >= 1024:
        return path
    head = path.read_bytes()[:200]
    if not head.startswith(b"version https://git-lfs"):
        return path
    fields = {}
    for line in head.decode("utf-8", "replace").splitlines():
        key, _, value = line.partition(" ")
        if key in {"oid", "size"}:
            fields[key] = value.strip()
    oid = fields.get("oid", "").removeprefix("sha256:")
    size = int(fields.get("size", 0))
    remote = _git("remote", "get-url", "origin")
    # owner/repo from https://github.com/o/r.git, git@github.com:o/r.git or an SSH host alias
    match = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?/?$", remote)
    if not match:
        raise RuntimeError(f"Cannot derive owner/repo from remote URL {remote!r} for {path}")
    slug = match.group(1)
    try:
        ref = _git("rev-parse", "HEAD")
    except Exception:
        ref = "main"
    rel = path.resolve().relative_to(repo_root().resolve()).as_posix()
    url = f"https://media.githubusercontent.com/media/{slug}/{ref}/{rel}"
    print(f"[LFS] {path} is a pointer -> downloading {size} B from {url}")
    tmp = path.with_suffix(path.suffix + ".lfs-download")
    digest = hashlib.sha256()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
            for chunk in iter(lambda: resp.read(1 << 20), b""):
                digest.update(chunk)
                out.write(chunk)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download LFS object for {path} from {url}: {exc}") from exc
    if digest.hexdigest() != oid or tmp.stat().st_size != size:
        got = tmp.stat().st_size
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"LFS object for {path} does not match its pointer: "
            f"sha256 {digest.hexdigest()[:16]}…/{got} B, expected {oid[:16]}…/{size} B"
        )
    tmp.replace(path)
    return path


def ensure_gkt_on_path() -> Path:
    gkt = repo_root() / "GKT"
    gkt_str = str(gkt)
    if gkt_str not in sys.path:
        sys.path.insert(0, gkt_str)
    return gkt
