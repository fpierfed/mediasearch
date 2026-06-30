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

    To note:
        * we do not process files or directories whose name starts with '.'
        * we keep track of duplicates, so that a lnk resolves to the same
          file, we do not process it twice.
    """
    seen: set[Path] = set()

    # This might be controversial: we skip files and directories whose name
    # starts with '.', UNLESS it is a directory in roots. The idea here is
    # that if the user asked to process a "hidden" dir, we should really do it
    for root in roots:
        root = Path(root).resolve()
        for dirpath, dirnames, filenames in root.walk(
            on_error=print, follow_symlinks=True
        ):
            # resolve symlinks at the dirpath level.
            dirpath = dirpath.resolve()
            if dirpath in seen:
                continue
            seen.add(dirpath)

            # We skip directories and files whose name starts with '.'
            # NB: we need to update dirnames in place otherwise it will not
            # work as intended. Hence the dirnames[:] syntax.
            dirnames[:] = [n for n in dirnames if not n.startswith('.')]

            for filename in filenames:
                if filename.startswith('.'):
                    continue

                p = dirpath / filename
                p.resolve()
                if p in seen:
                    continue
                seen.add(p)

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
