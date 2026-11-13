import hashlib
import json
import os
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
    hint = (
        "it is a Git LFS pointer. Install git-lfs and run `git lfs pull`, or set "
        "LFS_REPO=<owner>/<repo> (and optionally LFS_REF=<commit>) so it can be downloaded."
    )
    slug = os.environ.get("LFS_REPO", "").strip()
    if not slug:
        try:
            remote = _git("remote", "get-url", "origin")
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                f"Cannot read {path}: {hint} (no git remote 'origin': {exc})"
            ) from exc
        # owner/repo from https://github.com/o/r.git, git@github.com:o/r.git or an SSH host alias
        match = re.search(r"[:/]([^/:]+/[^/]+?)(?:\.git)?/?$", remote)
        if not match:
            raise RuntimeError(
                f"Cannot read {path}: {hint} (cannot derive owner/repo from {remote!r})"
            )
        slug = match.group(1)
    ref = os.environ.get("LFS_REF", "").strip()
    if not ref:
        try:
            ref = _git("rev-parse", "HEAD")
        except (OSError, subprocess.CalledProcessError):
            ref = "main"
    try:
        rel = path.resolve().relative_to(repo_root().resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            f"Cannot read {path}: {hint} (file is outside the repo, so its LFS path is unknown)"
        ) from exc
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


def read_json(path):
    """``json.loads`` on a file that may still be a Git LFS pointer (``*.json`` lives in LFS)."""
    return json.loads(materialize_lfs(path).read_text(encoding="utf-8"))


def ensure_gkt_on_path() -> Path:
    gkt = repo_root() / "GKT"
    gkt_str = str(gkt)
    if gkt_str not in sys.path:
        sys.path.insert(0, gkt_str)
    return gkt
