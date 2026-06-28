import stat
from pathlib import Path
from typing import Iterator, NamedTuple

from .config import classify_ext


class MediaFile(NamedTuple):
    path: Path
    media_type: str  # "image" | "video"
    mtime: float
    size: int


def walk(roots: list[Path]) -> Iterator[MediaFile]:
    """Yield every recognised media file under *roots*.

    Path.stat() follows symlinks, so media files reached through a symlink
    (even one pointing outside the root) are intentionally included.  Broken
    symlinks or unreadable files are skipped via OSError.
    """
    for root in roots:
        root = Path(root)
        for p in sorted(root.rglob('*')):
            try:
                st = p.stat()
            except OSError:
                continue
            if not stat.S_ISREG(st.st_mode):
                continue
            media_type = classify_ext(p)
            if media_type is None:
                continue
            yield MediaFile(
                path=p,
                media_type=media_type,
                mtime=st.st_mtime,
                size=st.st_size,
            )
