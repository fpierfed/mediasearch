import logging
import math
import uuid
from pathlib import Path
from typing import Callable
import time

import mlx.core as mx
import numpy as np
import pillow_heif
from PIL import Image

from AVFoundation import AVURLAsset, AVMediaTypeAudio
from Foundation import NSURL

from .config import DEFAULT_AUDIO_MODEL, Config
from .embedder import Embedder
from .frames import Frame, sample_video
from .store import ManifestStatus, Store
from .walker import MediaFile, walk

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()


def _has_audio_track(path: Path) -> bool:
    """
    Check if a video file has at least one audio track using AVFoundation.
    """

    try:
        url = NSURL.fileURLWithPath_(str(path))
        if not url:
            return False
        asset = AVURLAsset.URLAssetWithURL_options_(url, None)
        if not asset:
            return False
        return len(asset.tracksWithMediaType_(AVMediaTypeAudio)) > 0
    except Exception:
        # Fallback to true if we fail to check, so we still attempt
        # transcription
        return True


def _process_audio(
    mf: MediaFile,
    text_embedder: Embedder | Callable[[], Embedder],
    audio_model: str = DEFAULT_AUDIO_MODEL,
    store: Store | None = None,
    batch_size: int = 16,
) -> list[dict] | int:
    """
    Transcribe audio from a video file, embed the transcript, and either
    return the rows (``store=None``) or write them to *store* in chunks and
    return the row count.

    *text_embedder* may be an :class:`Embedder` or a zero-arg factory. The
    factory is only invoked once we know there are segments to embed, so a
    silent or trackless video never loads the text model.

    Segments are embedded and written *batch_size* at a time, and the MLX
    buffer cache is cleared after each chunk, so a long video's transcript
    never materialises all of its vectors in memory at once.
    """
    if not _has_audio_track(mf.path):
        logger.debug(
            'Skipping audio processing: no audio tracks found in %s', mf.path
        )
        return [] if store is None else 0

    try:
        import mlx_whisper

        # Note that mlx_whisper uses ffmpeg to extract & downsample the video
        # audio track, so while it could be tempting to extract the audio
        # ourselves, it would be double work. This also means that we do have
        # a hard dependency on ffmpeg.
        result = mlx_whisper.transcribe(
            str(mf.path),
            path_or_hf_repo=audio_model,
        )
        segments = result.get('segments', [])
        if not segments:
            return [] if store is None else 0

        all_rows: list[dict] = []
        total = 0
        resolved: Embedder | None = None

        for i in range(0, len(segments), batch_size):
            chunk = segments[i : i + batch_size]
            if resolved is None:
                resolved = (
                    text_embedder
                    if isinstance(text_embedder, Embedder)
                    else text_embedder()
                )
            vecs = resolved.embed_texts([seg['text'] for seg in chunk])
            rows = [
                {
                    'id': uuid.uuid4().hex,
                    'media_path': str(mf.path),
                    'media_type': 'transcript',
                    'text': seg['text'],
                    'vector': list(vec),
                    'start_time': float(seg['start']),
                    'end_time': float(seg['end']),
                }
                for seg, vec in zip(chunk, vecs)
            ]
            if store is None:
                all_rows.extend(rows)
            else:
                store.add_transcripts(rows)
            total += len(rows)
            mx.clear_cache()

        return all_rows if store is None else total
    except Exception:
        logger.error('Audio processing failed for %s', mf.path, exc_info=True)
        return [] if store is None else 0


def _load_bounded_rgb_image(path: Path, max_size: int | None) -> Image.Image:
    """
    Open *path* as an RGB image whose longer edge is at most *max_size* px.

    For JPEGs, ``Image.draft`` lets the decoder downscale during decode (cheap,
    avoids ever materialising the full-resolution buffer). Any remaining excess
    is trimmed with a high-quality thumbnail.
    """
    with Image.open(path) as im:
        if max_size is not None and im.format == 'JPEG':
            im.draft('RGB', (max_size, max_size))
        img = im.convert('RGB')
    if max_size is not None and max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img


def _embed_and_write_frames(
    frames: list[Frame], mf: MediaFile, embedder: Embedder, store: Store
) -> int:
    """Embed a batch of video frames, write the rows, and free MLX buffers."""
    vecs = embedder.embed_images([f.image for f in frames])
    metadata = [
        {
            'id': uuid.uuid4().hex,
            'media_path': str(mf.path),
            'media_type': 'video',
            'timestamp': float(f.timestamp),
            'frame_idx': int(f.frame_idx),
        }
        for f in frames
    ]
    store.add_embeddings_from_arrays(metadata, vecs)
    # Clear per batch (not just per file): a single long video can stream
    # thousands of frames, and the Metal cache would otherwise grow for
    # the whole video before being released.
    mx.clear_cache()
    return len(metadata)


def _process(
    mf: MediaFile, config: Config, embedder: Embedder, store: Store
) -> int:
    """
    Process a media file (image or video), write embeddings to *store*
    incrementally, and return the total number of embedded rows.

    For videos, frames are extracted, deduplicated, and embedded in batches
    of config.batch_size so that peak memory is bounded regardless of video
    length.
    """
    if mf.media_type == 'image':
        img = _load_bounded_rgb_image(mf.path, config.image_max_size)
        vec = embedder.embed_images([img])[0]
        store.add_embeddings_from_arrays(
            [
                {
                    'id': uuid.uuid4().hex,
                    'media_path': str(mf.path),
                    'media_type': 'image',
                    'timestamp': 0.0,
                    'frame_idx': 0,
                }
            ],
            np.asarray([vec], dtype=np.float32),
        )
        # Release MLX's Metal buffer cache so it does not accumulate across
        # files over a long run.
        mx.clear_cache()
        return 1

    if mf.media_type != 'video':
        raise ValueError(f"Unsupported media type '{mf.media_type}'")

    # Video: stream frames, embed in batches, write incrementally
    frames_iter = sample_video(
        mf.path,
        config.frame_interval,
        config.dedup_threshold,
        config.frame_max_size,
    )
    batch: list[Frame] = []
    total = 0

    for frame in frames_iter:
        batch.append(frame)
        if len(batch) >= config.batch_size:
            total += _embed_and_write_frames(batch, mf, embedder, store)
            batch.clear()

    # Flush remaining partial batch
    if batch:
        total += _embed_and_write_frames(batch, mf, embedder, store)

    return total


def _unchanged(prev: dict | ManifestStatus, mf: MediaFile) -> bool:
    """
    Check if a file is unchanged compared to its previously indexed state.

    Accepts either a full manifest dict or a compact :class:`ManifestStatus`.
    """
    logger.debug('>>> Check if unchanged')

    prev_mtime = prev['mtime'] if isinstance(prev, dict) else prev.mtime
    prev_size = prev['size'] if isinstance(prev, dict) else prev.size
    # Use math.isclose for mtime because the value round-trips through
    # PyArrow float64 in LanceDB and may differ by 1 ULP from the live
    # os.stat().st_mtime — exact == would cause false-dirty detection.
    # abs_tol of 1 ms is tight enough to still catch genuine modifications
    # on APFS/HFS+ (nanosecond resolution) while tolerating the coarser
    # precision reported by some network filesystems (SMB, NFS).
    return (
        math.isclose(prev_mtime, mf.mtime, rel_tol=1e-9, abs_tol=1e-3)
        and prev_size == mf.size
    )


class _LazyTextEmbedder:
    def __init__(self, text_embedder: Embedder | Callable[[], Embedder]):
        # An Embedder instance satisfies the runtime-checkable Embedder
        # protocol; a zero-arg factory does not, so this cleanly tells them
        # apart without a separate flag.
        self._text_embedder = text_embedder
        self._resolved: Embedder | None = (
            text_embedder if isinstance(text_embedder, Embedder) else None
        )

    def __call__(self) -> Embedder:
        if self._resolved is None:
            self._resolved = self._text_embedder()  # type: ignore[operator]
        return self._resolved


def index(
    config: Config,
    embedder: Embedder,
    text_embedder: Embedder | Callable[[], Embedder],
    store: Store,
    roots: list[str],
    reindex: bool = False,
    progress: Callable[[], None] | None = None,
) -> None:
    """
    Walk directories to find media files and process them into the index.

    *text_embedder* may be an :class:`Embedder` or a zero-arg factory that
    builds one. A factory is resolved lazily on the first video transcript and
    memoised, so an image-only run never loads the text model — keeping it out
    of memory alongside the visual embedder until it is actually needed.
    """
    get_text_embedder = _LazyTextEmbedder(text_embedder)
    manifest = store.manifest_statuses()
    for mf in walk([Path(r) for r in roots]):
        t0 = time.time()

        key = str(mf.path)
        prev = manifest.get(key)

        logger.debug('>>> Pre-processing checks...')
        logger.debug(f'>>> prev={prev}')
        logger.debug(f'>>> current={key}')

        # Skip files that are unchanged AND already settled (done or
        # permanently errored), unless --reindex forces a rebuild.
        # 'pending' is never skipped -> resume.
        if (
            not reindex
            and prev is not None
            and prev.status in ('done', 'error')
            and _unchanged(prev, mf)
        ):
            if progress:
                progress()

            logger.debug('>>> Skipped')
            continue

        store.delete_file(key)  # clear any stale/partial rows
        store.set_file(
            path=key,
            mtime=mf.mtime,
            size=mf.size,
            media_type=mf.media_type,
            status='pending',
        )
        logger.debug(f'>>> Elapsed: {time.time() - t0:.02f}')
        del t0

        logger.debug(f'>>> Processing {key}...')

        try:
            t0 = time.time()
            logger.debug('>>> _process()')
            n_vectors = _process(mf, config, embedder, store)
            logger.debug(f'>>> Elapsed: {time.time() - t0:.02f}')
            del t0

            if mf.media_type == 'video' and config.index_audio:
                t0 = time.time()
                logger.debug('>>> _process_audio()')
                _process_audio(
                    mf,
                    get_text_embedder,
                    config.audio_model,
                    store=store,
                    batch_size=config.batch_size,
                )
                logger.debug(f'>>> Elapsed: {time.time() - t0:.02f}')
                del t0

            store.set_file(
                path=key,
                mtime=mf.mtime,
                size=mf.size,
                media_type=mf.media_type,
                status='done',
                n_vectors=n_vectors,
            )
            logger.debug('Done')
        except Exception as exc:
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
            logger.debug('Error')
        if progress:
            progress()
