import itertools
import logging
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image

from .config import Config
from .embedder import Embedder
from .frames import sample_video
from .store import Store

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()

# Pull extra rows so grouping-by-file still yields top_k distinct files.
# On very small indexes (where total vectors < k * _OVERFETCH) the final
# result set may contain fewer than k files — this is expected and harmless.
_OVERFETCH = 20

_MOD_VISUAL = '[VISUAL]'
_MOD_AUDIO = '[AUDIO]'


def format_timestamp(seconds: float) -> str:
    """Format a timestamp in seconds as an MM:SS string."""
    total = int(round(seconds))
    return f'{total // 60}:{total % 60:02d}'


def _group_by_media(rows: list[dict]) -> list[dict]:
    """Group search result rows by media file, keeping only the best match per file."""
    best: dict[str, dict] = {}
    for r in rows:
        mp = r['media_path']
        if mp not in best or r['score'] > best[mp]['score']:
            best[mp] = r
    return sorted(best.values(), key=lambda r: r['score'], reverse=True)


def _group_by_media_multi(
    visual_rows: list[dict], audio_rows: list[dict]
) -> list[dict]:
    """Aggregate visual and audio results by media path.

    Each file gets a combined score (best_visual + best_audio) and a
    modality tag showing which sources contributed.
    """
    groups: dict[str, dict] = {}
    for r in itertools.chain(visual_rows, audio_rows):
        mp = r['media_path']
        if mp not in groups:
            groups[mp] = {
                'best_visual': 0.0,
                'best_audio': 0.0,
                'has_visual': False,
                'has_audio': False,
                'best_row': None,
            }

        if r['modality'] == _MOD_VISUAL:
            groups[mp]['best_visual'] = max(
                groups[mp]['best_visual'], r['score']
            )
            groups[mp]['has_visual'] = True
        else:
            groups[mp]['best_audio'] = max(
                groups[mp]['best_audio'], r['score']
            )
            groups[mp]['has_audio'] = True

        best = groups[mp]['best_row']
        if best is None or r['score'] > best['score']:
            groups[mp]['best_row'] = dict(r)

    results: list[dict] = []
    for mp, g in groups.items():
        row = g['best_row']
        row['score'] = g['best_visual'] + g['best_audio']

        mods = []
        if g['has_visual']:
            mods.append(_MOD_VISUAL)
        if g['has_audio']:
            mods.append(_MOD_AUDIO)
        row['modality'] = ''.join(mods)
        results.append(row)

    return sorted(results, key=lambda r: r['score'], reverse=True)


def _format(rows: list[dict]) -> list[dict]:
    """Format search result rows for output."""
    out = []
    for rank, r in enumerate(rows, start=1):
        is_timeable = r['media_type'] in ('video', 'transcript')
        ts = float(r.get('timestamp', r.get('start_time', 0.0)))
        d = {
            'rank': rank,
            'score': round(float(r['score']), 4),
            'path': r['media_path'],
            'media_type': r['media_type'],
            'timestamp': ts,
            'time': format_timestamp(ts) if is_timeable else None,
        }
        if 'modality' in r:
            d['modality'] = r['modality']
        out.append(d)
    return out


def search_text(
    query: str,
    config: Config,
    embedder: Embedder,
    text_embedder: Embedder,
    store: Store,
    top_k: int | None = None,
    media_type: str | None = None,
) -> list[dict]:
    """Search for media matching a text query, combining visual and audio similarities."""
    k = top_k or config.top_k

    vec = embedder.embed_texts([query])[0]
    visual_rows = store.search(
        vec, top_k=k * _OVERFETCH, media_type=media_type
    )
    for r in visual_rows:
        r['modality'] = _MOD_VISUAL

    audio_rows: list[dict] = []
    if media_type in (None, 'video', 'transcript'):
        text_vec = text_embedder.embed_texts([query])[0]
        audio_vec_rows = store.search_transcripts_vector(
            text_vec, top_k=k * _OVERFETCH
        )
        audio_fts_rows = store.search_transcripts_fts(
            query, top_k=k * _OVERFETCH
        )

        audio_rows = audio_vec_rows + audio_fts_rows
        for r in audio_rows:
            r['modality'] = _MOD_AUDIO

    ranked = _group_by_media_multi(visual_rows, audio_rows)
    return _format(ranked[:k])


def search_image(
    path: str | Path,
    config: Config,
    embedder: Embedder,
    store: Store,
    top_k: int | None = None,
    media_type: str | None = None,
) -> list[dict]:
    """Search for media visually similar to the provided image."""
    k = top_k or config.top_k
    with Image.open(Path(path)) as im:
        img = im.convert('RGB')
    vec = embedder.embed_images([img])[0]
    rows = store.search(vec, top_k=k * _OVERFETCH, media_type=media_type)
    return _format(_group_by_media(rows)[:k])


def search_clip(
    path: str | Path,
    config: Config,
    embedder: Embedder,
    store: Store,
    top_k: int | None = None,
) -> list[dict]:
    """Search the index for clips similar to *path*.

    To ensure performance on long-form video, sampled frame vectors are
    mean-pooled into a single query vector before searching. This trades
    some temporal precision for a large speed-up, resulting in a single
    LanceDB lookup instead of one per frame.
    """
    k = top_k or config.top_k
    try:
        frames = sample_video(
            Path(path), config.frame_interval, config.dedup_threshold
        )
    except Exception:
        logger.exception('Failed to extract frames from %s', path)
        return []
    if not frames:
        return []
    vecs = embedder.embed_images([f.image for f in frames])
    pooled_vec = np.mean(vecs, axis=0)
    # Clip->clip: a candidate clip scores by its single best-matching moment (max).
    agg: dict[str, dict] = {}
    for r in store.search(
        pooled_vec, top_k=k * _OVERFETCH, media_type='video'
    ):
        mp = r['media_path']
        if mp not in agg or r['score'] > agg[mp]['score']:
            agg[mp] = r
    ranked = sorted(agg.values(), key=lambda r: r['score'], reverse=True)
    return _format(ranked[:k])
