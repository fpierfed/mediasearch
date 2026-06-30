import json as _json
import os
import subprocess
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from . import pipeline, search as search_mod
from .config import Config, DEFAULT_MODEL, DEFAULT_TEXT_MODEL, TEXT_EMBED_DIM
from . import embedder as emb
from .store import Store

app = typer.Typer(add_completion=False, help='Local semantic media search.')


class MediaType(str, Enum):
    image = 'image'
    video = 'video'


def _config(index_path: Optional[Path], model: Optional[str] = None) -> Config:
    """Create a configuration object using the given index path and model."""
    try:
        c = Config(model=model or DEFAULT_MODEL)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if index_path is not None:
        c.index_path = index_path
    return c


def _open_store(config: Config) -> Store:
    """Open (or create) the LanceDB index at *config.index_path*."""
    return Store(
        config.index_path, dim=config.embed_dim, fts_score_k=config.fts_score_k
    )


def _build_embedder(
    config: Config, text_model: Optional[str] = None
) -> emb.Embedder:
    """
    Build and return an embedder for images or text, using a fake if requested.
    """
    if os.environ.get('MEDIASEARCH_FAKE_EMBEDDER') == '1':
        return emb.FakeEmbedder(
            dim=TEXT_EMBED_DIM if text_model else config.embed_dim
        )

    try:
        if text_model:
            return emb.MLXTextEmbedder(
                model_name=text_model, batch_size=config.batch_size
            )
        return emb.MLXSigLIPEmbedder(
            config.model, batch_size=config.batch_size, dim=config.embed_dim
        )
    except Exception as exc:  # noqa: BLE001
        model_name = text_model or config.model
        raise typer.BadParameter(
            f"Could not load model '{model_name}': {exc}. "
            'mediasearch needs Apple Silicon + MLX and downloads the model '
            + 'on first use.'
        ) from exc


def _build_text_embedder(config: Config) -> emb.Embedder:
    return _build_embedder(config, text_model=DEFAULT_TEXT_MODEL)


def _guard_dim(store: Store, config: Config, reindex: bool = False) -> None:
    """
    Validate that the on-disk embeddings table matches the configured visual
    model dimension.
    """
    on_disk = store.index_dim()
    if on_disk != config.embed_dim:
        if reindex:
            store.reset()
        else:
            raise typer.BadParameter(
                f'Index at {config.index_path} stores {on_disk}-d vectors, '
                + f"but model '{config.model}' produces {config.embed_dim}-d. "
                + 'Rebuild with `mediasearch index <dirs> --model '
                + f'{config.model} --reindex`, or delete {config.index_path}.'
            )


def _guard_text_dim(store: Store) -> None:
    """Validate that the on-disk transcripts table dimension matches the
    configured text model.  Currently all supported text models produce
    TEXT_EMBED_DIM-d vectors, so this is a forward-looking safeguard."""
    on_disk = store.text_index_dim()
    if on_disk != store.text_dim:
        raise typer.BadParameter(
            f'Transcripts table stores {on_disk}-d vectors, but the text '
            f'embedder produces {store.text_dim}-d. Rebuild with '
            f'`mediasearch index <dirs> --reindex`, or delete the index.'
        )


def _emit(results: list[dict], as_json: bool, open_top: bool) -> None:
    """
    Output search results to the console, optionally in JSON format, and
    optionally open the top result.
    """
    if as_json:
        typer.echo(_json.dumps(results, indent=2))
    elif not results:
        typer.echo('No results.')
    else:
        for r in results:
            suffix = f'  @ {r["time"]}' if r['time'] else ''
            mod_prefix = f'{r["modality"]} ' if 'modality' in r else ''
            typer.echo(
                f'{r["rank"]:2d}. {r["score"]:.3f}  '
                + f'{mod_prefix}{r["path"]}{suffix}'
            )
    if open_top and results:
        subprocess.run(['open', '-R', results[0]['path']], check=False)


@app.command()
def index(
    dirs: list[Path] = typer.Argument(..., help='Folders to scan recursively'),
    index_path: Optional[Path] = typer.Option(None, '--index-path'),
    reindex: bool = typer.Option(
        False, '--reindex', help='Rebuild from scratch'
    ),
    model: Optional[str] = typer.Option(
        None, '--model', help='embedding model id'
    ),
    batch_size: Optional[int] = typer.Option(
        None, '--batch-size', min=1, help='Embedding batch size'
    ),
    frame_interval: Optional[float] = typer.Option(
        None, '--frame-interval', min=0.1, help='Seconds between video frames'
    ),
    dedup_threshold: Optional[int] = typer.Option(
        None, '--dedup-threshold', min=0, help='Frame dedup hamming distance'
    ),
    image_max_size: Optional[int] = typer.Option(
        None, '--image-max-size', min=1, help='Cap decoded still-image edge'
    ),
    frame_max_size: Optional[int] = typer.Option(
        None, '--frame-max-size', min=1, help='Cap decoded video-frame edge'
    ),
    index_audio: bool = typer.Option(
        True, '--audio/--no-audio', help='Index video transcripts'
    ),
) -> None:
    """Index directories containing media files for semantic search."""

    config = _config(index_path, model)
    if batch_size is not None:
        config.batch_size = batch_size
    if frame_interval is not None:
        config.frame_interval = frame_interval
    if dedup_threshold is not None:
        config.dedup_threshold = dedup_threshold
    if image_max_size is not None:
        config.image_max_size = image_max_size
    if frame_max_size is not None:
        config.frame_max_size = frame_max_size
    config.index_audio = index_audio
    store = _open_store(config)
    _guard_dim(store, config, reindex=reindex)
    _guard_text_dim(store)
    embedder = _build_embedder(config)
    # Pass a factory, not an instance: the text model is only needed for video
    # transcripts, so deferring its construction keeps it out of memory during
    # image-only runs (and until the first video) alongside the visual model.
    text_embedder = partial(_build_text_embedder, config)

    bar = tqdm(unit='file', desc='Indexing')
    pipeline.index(
        config,
        embedder,
        text_embedder,
        store,
        [str(d) for d in dirs],
        reindex=reindex,
        progress=bar.update,
    )
    bar.close()
    st = store.stats()
    typer.echo(
        f'Indexed: {st["done"]} done, {st["error"]} errors, '
        f'{st["vectors"]} vectors across {st["files"]} files.'
    )


@app.command()
def search(
    query: str,
    index_path: Optional[Path] = typer.Option(None, '--index-path'),
    top_k: Optional[int] = typer.Option(None, '--top-k'),
    media_type: Optional[MediaType] = typer.Option(
        None, '--type', help='image|video'
    ),
    json: bool = typer.Option(False, '--json'),
    open: bool = typer.Option(
        False, '--open', help='Reveal top hit in Finder'
    ),
    model: Optional[str] = typer.Option(
        None, '--model', help='embedding model id'
    ),
) -> None:
    """Search the index for media matching a text query."""
    config = _config(index_path, model)
    store = _open_store(config)
    _guard_dim(store, config)
    _guard_text_dim(store)
    embedder = _build_embedder(config)
    text_embedder = _build_embedder(config, text_model=DEFAULT_TEXT_MODEL)
    results = search_mod.search_text(
        query,
        config,
        embedder,
        text_embedder,
        store,
        top_k=top_k,
        media_type=(media_type.value if media_type else None),
    )
    _emit(results, json, open)


@app.command('similar-image')
def similar_image(
    path: Path,
    index_path: Optional[Path] = typer.Option(None, '--index-path'),
    top_k: Optional[int] = typer.Option(None, '--top-k'),
    media_type: Optional[MediaType] = typer.Option(
        None, '--type', help='image|video'
    ),
    json: bool = typer.Option(False, '--json'),
    open: bool = typer.Option(False, '--open'),
    model: Optional[str] = typer.Option(
        None, '--model', help='embedding model id'
    ),
) -> None:
    """Search the index for media visually similar to a provided image."""
    config = _config(index_path, model)
    store = _open_store(config)
    _guard_dim(store, config)
    embedder = _build_embedder(config)
    results = search_mod.search_image(
        path,
        config,
        embedder,
        store,
        top_k=top_k,
        media_type=(media_type.value if media_type else None),
    )
    _emit(results, json, open)


@app.command('similar-clip')
def similar_clip(
    path: Path,
    index_path: Optional[Path] = typer.Option(None, '--index-path'),
    top_k: Optional[int] = typer.Option(None, '--top-k'),
    json: bool = typer.Option(False, '--json'),
    open: bool = typer.Option(False, '--open'),
    model: Optional[str] = typer.Option(
        None, '--model', help='embedding model id'
    ),
) -> None:
    """Search the index for video clips similar to a provided video clip."""
    config = _config(index_path, model)
    store = _open_store(config)
    _guard_dim(store, config)
    embedder = _build_embedder(config)
    results = search_mod.search_clip(
        path, config, embedder, store, top_k=top_k
    )
    _emit(results, json, open)


@app.command()
def status(
    index_path: Optional[Path] = typer.Option(None, '--index-path'),
    model: Optional[str] = typer.Option(
        None, '--model', help='embedding model id'
    ),
) -> None:
    """Show status and statistics about the current index."""
    config = _config(index_path, model)
    store = _open_store(config)
    st = store.stats()
    typer.echo(
        f'files={st["files"]}  done={st["done"]}  pending={st["pending"]}  '
        f'error={st["error"]}  vectors={st["vectors"]}'
    )
    errs = store.errors()
    if errs:
        typer.echo('\nErrors:')
        for e in errs:
            typer.echo(f'  {e["path"]}: {e["error_msg"]}')
