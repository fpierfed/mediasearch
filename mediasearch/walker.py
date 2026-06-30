from pathlib import Path
from typing import Iterator, NamedTuple

from .config import classify_ext


class MediaFile(NamedTuple):
    path: Path
    media_type: str  # "image" | "video"
    mtime: float
    size: int


def walk(roots: list[Path]) -> Iterator[MediaFile]:
    """
    Yield every recognised media file under *roots*.

    We intentionally follow symlinks, so media files reached through a symlink
    (even one pointing outside the root) are intentionally included.  Broken
    symlinks or unreadable files are skipped via OSError.
    """
    for root in roots:
        root = Path(root).resolve()
        for dirpath, _, filenames in root.walk(
            on_error=print, follow_symlinks=True
        ):
            # We skip directories and files whose name starts with '.'
            if dirpath.name.startswith('.'):
                continue

            for filename in filenames:
                if filename.startswith('.'):
                    continue

                p = dirpath / filename
                media_type = classify_ext(p)
                if media_type is None:
                    continue

                st = p.stat()
                yield MediaFile(
                    path=p,
                    media_type=media_type,
                    mtime=st.st_mtime,
                    size=st.st_size,
                )
