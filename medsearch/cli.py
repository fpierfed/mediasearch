from __future__ import annotations

import json as _json
import os
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import typer

from . import pipeline, search as search_mod
from .config import Config, DEFAULT_MODEL, DEFAULT_TEXT_MODEL
from .store import Store

app = typer.Typer(add_completion=False, help="Local semantic media search.")


class MediaType(str, Enum):
    image = "image"
    video = "video"


def _config(index_path: Optional[Path], model: Optional[str] = None) -> Config:
    c = Config()
    if index_path is not None:
        c.index_path = index_path
    c.model = model or DEFAULT_MODEL
    try:
        c.embed_dim  # validate the model is known (raises ValueError otherwise)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return c


def _open_store(config: Config) -> Store:
    """Open (or create) the LanceDB index at *config.index_path*."""
    return Store(config.index_path, dim=config.embed_dim, fts_score_k=config.fts_score_k)


def _build_embedder(config: Config, text_model: Optional[str] = None) -> Any:
    if os.environ.get("MEDSEARCH_FAKE_EMBEDDER") == "1":
        from .embedder import FakeEmbedder

        return FakeEmbedder(dim=768 if text_model else config.embed_dim)
    from . import embedder as _emb

    try:
        if text_model:
            return _emb.MLXTextEmbedder(
                model_name=text_model, batch_size=config.batch_size
            )
        return _emb.MLXSigLIPEmbedder(
            config.model, batch_size=config.batch_size, dim=config.embed_dim
        )
    except Exception as exc:  # noqa: BLE001
        model_name = text_model or config.model
        raise typer.BadParameter(
            f"Could not load model '{model_name}': {exc}. "
            "medsearch needs Apple Silicon + MLX and downloads the model on first use."
        ) from exc


def _guard_dim(store: Store, config: Config, reindex: bool = False) -> None:
    on_disk = store.index_dim()
    if on_disk != config.embed_dim:
        if reindex:
            store.reset()
        else:
            raise typer.BadParameter(
                f"Index at {config.index_path} stores {on_disk}-d vectors, but model "
                f"'{config.model}' produces {config.embed_dim}-d. Rebuild with "
                f"`medsearch index <dirs> --model {config.model} --reindex`, or delete {config.index_path}."
            )


def _guard_text_dim(store: Store) -> None:
    """Validate that the on-disk transcripts table dimension matches the
    configured text model.  Currently all supported text models produce
    768-d vectors, so this is a forward-looking safeguard."""
    on_disk = store.text_index_dim()
    if on_disk != store.text_dim:
        raise typer.BadParameter(
            f"Transcripts table stores {on_disk}-d vectors, but the text "
            f"embedder produces {store.text_dim}-d. Rebuild with "
            f"`medsearch index <dirs> --reindex`, or delete the index."
        )


def _emit(results: list[dict], as_json: bool, open_top: bool) -> None:
    if as_json:
        typer.echo(_json.dumps(results, indent=2))
    elif not results:
        typer.echo("No results.")
    else:
        for r in results:
            suffix = f"  @ {r['time']}" if r["time"] else ""
            mod_prefix = f"{r['modality']} " if "modality" in r else ""
            typer.echo(f"{r['rank']:2d}. {r['score']:.3f}  {mod_prefix}{r['path']}{suffix}")
    if open_top and results:
        subprocess.run(["open", "-R", results[0]["path"]], check=False)


@app.command()
def index(
    dirs: list[Path] = typer.Argument(..., help="Folders to scan recursively"),
    index_path: Optional[Path] = typer.Option(None, "--index-path"),
    reindex: bool = typer.Option(False, "--reindex", help="Rebuild from scratch"),
    model: Optional[str] = typer.Option(None, "--model", help="embedding model id"),
) -> None:
    from tqdm import tqdm

    config = _config(index_path, model)
    store = _open_store(config)
    _guard_dim(store, config, reindex=reindex)
    _guard_text_dim(store)
    embedder = _build_embedder(config)
    text_embedder = _build_embedder(config, text_model=DEFAULT_TEXT_MODEL)

    bar = tqdm(unit="file", desc="Indexing")
    pipeline.index(
        config, embedder, text_embedder, store, [str(d) for d in dirs],
        reindex=reindex, progress=lambda: bar.update(1),
    )
    bar.close()
    st = store.stats()
    typer.echo(
        f"Indexed: {st['done']} done, {st['error']} errors, "
        f"{st['vectors']} vectors across {st['files']} files."
    )


@app.command()
def search(
    query: str,
    index_path: Optional[Path] = typer.Option(None, "--index-path"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    media_type: Optional[MediaType] = typer.Option(None, "--type", help="image|video"),
    json: bool = typer.Option(False, "--json"),
    open: bool = typer.Option(False, "--open", help="Reveal top hit in Finder"),
    model: Optional[str] = typer.Option(None, "--model", help="embedding model id"),
) -> None:
    config = _config(index_path, model)
    store = _open_store(config)
    _guard_dim(store, config)
    _guard_text_dim(store)
    embedder = _build_embedder(config)
    text_embedder = _build_embedder(config, text_model=DEFAULT_TEXT_MODEL)
    results = search_mod.search_text(
        query, config, embedder, text_embedder, store, top_k=top_k,
        media_type=(media_type.value if media_type else None),
    )
    _emit(results, json, open)


@app.command("similar-image")
def similar_image(
    path: Path,
    index_path: Optional[Path] = typer.Option(None, "--index-path"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    media_type: Optional[MediaType] = typer.Option(None, "--type", help="image|video"),
    json: bool = typer.Option(False, "--json"),
    open: bool = typer.Option(False, "--open"),
    model: Optional[str] = typer.Option(None, "--model", help="embedding model id"),
) -> None:
    config = _config(index_path, model)
    store = _open_store(config)
    _guard_dim(store, config)
    embedder = _build_embedder(config)
    results = search_mod.search_image(
        path, config, embedder, store, top_k=top_k,
        media_type=(media_type.value if media_type else None),
    )
    _emit(results, json, open)


@app.command("similar-clip")
def similar_clip(
    path: Path,
    index_path: Optional[Path] = typer.Option(None, "--index-path"),
    top_k: Optional[int] = typer.Option(None, "--top-k"),
    json: bool = typer.Option(False, "--json"),
    open: bool = typer.Option(False, "--open"),
    model: Optional[str] = typer.Option(None, "--model", help="embedding model id"),
) -> None:
    config = _config(index_path, model)
    store = _open_store(config)
    _guard_dim(store, config)
    embedder = _build_embedder(config)
    results = search_mod.search_clip(path, config, embedder, store, top_k=top_k)
    _emit(results, json, open)


@app.command()
def status(
    index_path: Optional[Path] = typer.Option(None, "--index-path"),
    model: Optional[str] = typer.Option(None, "--model", help="embedding model id"),
) -> None:
    config = _config(index_path, model)
    store = _open_store(config)
    st = store.stats()
    typer.echo(
        f"files={st['files']}  done={st['done']}  pending={st['pending']}  "
        f"error={st['error']}  vectors={st['vectors']}"
    )
    errs = store.errors()
    if errs:
        typer.echo("\nErrors:")
        for e in errs:
            typer.echo(f"  {e['path']}: {e['error_msg']}")
