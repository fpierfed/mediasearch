import logging
import math
import uuid
from pathlib import Path
from typing import Callable, Any

import mlx_whisper
import pillow_heif
from PIL import Image

from .config import DEFAULT_AUDIO_MODEL, Config
from .frames import sample_video
from .store import Store
from .walker import MediaFile, walk

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()


def _process_audio(
    mf: MediaFile, text_embedder: Any, audio_model: str = DEFAULT_AUDIO_MODEL
) -> list[dict]:
    try:
        result = mlx_whisper.transcribe(
            str(mf.path),
            path_or_hf_repo=audio_model,
        )
        segments = result.get('segments', [])
        if not segments:
            return []

        texts = [seg['text'] for seg in segments]
        vecs = text_embedder.embed_texts(texts)

        return [
            {
                'id': uuid.uuid4().hex,
                'media_path': str(mf.path),
                'media_type': 'transcript',
                'text': seg['text'],
                'vector': list(vec),
                'start_time': float(seg['start']),
                'end_time': float(seg['end']),
            }
            for seg, vec in zip(segments, vecs)
        ]
    except Exception:
        logger.error('Audio processing failed for %s', mf.path, exc_info=True)
        return []


def _process(mf: MediaFile, config: Config, embedder: Any) -> list[dict]:
    if mf.media_type == 'image':
        with Image.open(mf.path) as im:
            img = im.convert('RGB')
        vec = embedder.embed_images([img])[0]
        return [
            {
                'id': uuid.uuid4().hex,
                'media_path': str(mf.path),
                'media_type': 'image',
                'vector': list(vec),
                'timestamp': 0.0,
                'frame_idx': 0,
            }
        ]

    frames = sample_video(
        mf.path, config.frame_interval, config.dedup_threshold
    )
    if not frames:
        return []
    vecs = embedder.embed_images([f.image for f in frames])
    return [
        {
            'id': uuid.uuid4().hex,
            'media_path': str(mf.path),
            'media_type': 'video',
            'vector': list(v),
            'timestamp': float(f.timestamp),
            'frame_idx': int(f.frame_idx),
        }
        for f, v in zip(frames, vecs)
    ]


def _unchanged(prev: dict, mf: MediaFile) -> bool:
    # Use math.isclose for mtime because the value round-trips through
    # PyArrow float64 in LanceDB and may differ by 1 ULP from the live
    # os.stat().st_mtime — exact == would cause false-dirty detection.
    # abs_tol of 1 ms is tight enough to still catch genuine modifications
    # on APFS/HFS+ (nanosecond resolution) while tolerating the coarser
    # precision reported by some network filesystems (SMB, NFS).
    return (
        math.isclose(prev['mtime'], mf.mtime, rel_tol=1e-9, abs_tol=1e-3)
        and prev['size'] == mf.size
    )


def index(
    config: Config,
    embedder: Any,
    text_embedder: Any,
    store: Store,
    roots: list[str],
    reindex: bool = False,
    progress: Callable[[], None] | None = None,
) -> None:
    manifest = store.manifest()
    for mf in walk([Path(r) for r in roots]):
        key = str(mf.path)
        prev = manifest.get(key)

        # Skip files that are unchanged AND already settled (done or permanently errored),
        # unless --reindex forces a rebuild. 'pending' is never skipped -> resume.
        if (
            not reindex
            and prev is not None
            and prev['status'] in ('done', 'error')
            and _unchanged(prev, mf)
        ):
            if progress:
                progress()
            continue

        store.delete_file(key)  # clear any stale/partial rows
        store.set_file(
            path=key,
            mtime=mf.mtime,
            size=mf.size,
            media_type=mf.media_type,
            status='pending',
        )
        try:
            visual_rows = _process(mf, config, embedder)
            transcript_rows = []
            if mf.media_type == 'video':
                transcript_rows = _process_audio(
                    mf, text_embedder, config.audio_model
                )

            store.add_embeddings(visual_rows)
            store.add_transcripts(transcript_rows)

            store.set_file(
                path=key,
                mtime=mf.mtime,
                size=mf.size,
                media_type=mf.media_type,
                status='done',
                n_vectors=len(visual_rows),
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the run
            # Clean up any partially-written rows to avoid orphaned data
            # (e.g. embeddings committed before a transcript write failed).
            store.delete_file(key)
            store.set_file(
                path=key,
                mtime=mf.mtime,
                size=mf.size,
                media_type=mf.media_type,
                status='error',
                error_msg=str(exc),
            )
        if progress:
            progress()
