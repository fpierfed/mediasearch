import json

import pytest
from typer.testing import CliRunner

from mediasearch.cli import app

runner = CliRunner()


def test_index_then_status(tmp_path, make_image, monkeypatch):
    """Test that indexing a directory works and status returns file counts."""
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'

    r = runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ['status', '--index-path', str(idx)])
    assert r.exit_code == 0
    assert 'files' in r.output.lower()


def test_search_json_output(tmp_path, make_image, monkeypatch):
    """Test that the search command with --json returns valid JSON output."""
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'
    runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])

    r = runner.invoke(
        app, ['search', 'a photo', '--index-path', str(idx), '--json']
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert isinstance(data, list)
    assert data and data[0]['path'].endswith('a.png')


def test_similar_image(tmp_path, make_image, monkeypatch):
    """Test that similar-image command finds and returns the closest
    image match.
    """
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png', (1, 2, 3))
    make_image(lib / 'b.png', (9, 9, 9))
    idx = tmp_path / 'idx'
    runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])

    r = runner.invoke(
        app,
        [
            'similar-image',
            str(lib / 'a.png'),
            '--index-path',
            str(idx),
            '--json',
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data[0]['path'].endswith('a.png')


def test_invalid_type_is_rejected(tmp_path, make_image, monkeypatch):
    """Test that providing an invalid --type to search results in an error."""
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'
    runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])
    r = runner.invoke(
        app, ['search', 'q', '--index-path', str(idx), '--type', 'bogus']
    )
    assert r.exit_code != 0
    assert (
        'bogus' in r.output or 'image' in r.output
    )  # message names the bad value or valid choices


def test_model_load_failure_is_friendly(tmp_path, make_image, monkeypatch):
    """Test that failing to load a real model provides a user-friendly
    error message.
    """
    # No fake-embedder env -> real branch; force the real embedder to
    # fail to load.
    monkeypatch.delenv('MEDIASEARCH_FAKE_EMBEDDER', raising=False)
    import mediasearch.embedder as emb

    def boom(*a, **k):
        raise RuntimeError('metal device not found')

    monkeypatch.setattr(emb, 'MLXSigLIPEmbedder', boom)
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'
    r = runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])
    assert r.exit_code != 0
    # friendly: mentions the model or Apple Silicon, and does NOT dump
    # a raw traceback
    assert 'Traceback' not in r.output
    assert ('model' in r.output.lower()) or (
        'apple silicon' in r.output.lower()
    )


BASE_MODEL = 'mlx-community/siglip2-base-patch16-384'


def test_bad_model_is_rejected(tmp_path, monkeypatch):
    """Test that providing an unknown --model flag results in a clear error."""
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    idx = tmp_path / 'idx'
    r = runner.invoke(
        app, ['status', '--index-path', str(idx), '--model', 'nope']
    )
    assert r.exit_code != 0
    assert 'nope' in r.output or 'Unknown model' in r.output


def test_dim_mismatch_without_reindex_errors(
    tmp_path, make_image, monkeypatch
):
    """Test that running with a mismatched model dimension prompts the
    user to reindex.
    """
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'
    # build a 1152-d index with the default model
    assert (
        runner.invoke(
            app, ['index', str(lib), '--index-path', str(idx)]
        ).exit_code
        == 0
    )
    # now query/index with the 768-d base model, no reindex -> friendly error
    r = runner.invoke(
        app, ['search', 'q', '--index-path', str(idx), '--model', BASE_MODEL]
    )
    assert r.exit_code != 0
    assert 'reindex' in r.output.lower() or '768' in r.output


def test_dim_mismatch_with_reindex_rebuilds(tmp_path, make_image, monkeypatch):
    """Test that passing --reindex allows rebuilding the index with a
    new model dimension.
    """
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'
    assert (
        runner.invoke(
            app, ['index', str(lib), '--index-path', str(idx)]
        ).exit_code
        == 0
    )
    # rebuild at 768 with the base model
    r = runner.invoke(
        app,
        [
            'index',
            str(lib),
            '--index-path',
            str(idx),
            '--model',
            BASE_MODEL,
            '--reindex',
        ],
    )
    assert r.exit_code == 0, r.output
    # now searching with the base model works (dims match) and returns
    # the image
    r2 = runner.invoke(
        app,
        [
            'search',
            'q',
            '--index-path',
            str(idx),
            '--model',
            BASE_MODEL,
            '--json',
        ],
    )
    assert r2.exit_code == 0, r2.output
    import json as _j

    assert _j.loads(r2.output)[0]['path'].endswith('a.png')


def test_similar_clip(tmp_path, make_image, sample_video, monkeypatch):
    """similar-clip command indexes and matches a video clip."""
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'clip.mp4').write_bytes(sample_video.read_bytes())
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'
    runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])

    r = runner.invoke(
        app,
        [
            'similar-clip',
            str(lib / 'clip.mp4'),
            '--index-path',
            str(idx),
            '--json',
        ],
    )
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]['path'].endswith('clip.mp4')


def test_search_plain_text_output(tmp_path, make_image, monkeypatch):
    """search without --json emits plain-text formatted output with
    modality.
    """
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'
    runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])

    r = runner.invoke(app, ['search', 'a photo', '--index-path', str(idx)])
    assert r.exit_code == 0, r.output
    assert ' 1.' in r.output  # rank prefix
    assert 'a.png' in r.output  # path present


def test_open_flag(tmp_path, make_image, monkeypatch):
    """--open flag should not crash (subprocess call is best-effort)."""
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    idx = tmp_path / 'idx'
    runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])

    r = runner.invoke(
        app, ['search', 'a photo', '--index-path', str(idx), '--open']
    )
    assert r.exit_code == 0, r.output


def test_status_shows_errors(tmp_path, make_image, monkeypatch):
    """status command lists error entries when the index contains failures."""
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'good.png')
    (lib / 'bad.mp4').write_bytes(b'not a real video')
    idx = tmp_path / 'idx'
    runner.invoke(app, ['index', str(lib), '--index-path', str(idx)])

    r = runner.invoke(app, ['status', '--index-path', str(idx)])
    assert r.exit_code == 0, r.output
    assert 'Errors:' in r.output
    assert 'bad.mp4' in r.output


def test_text_dim_mismatch_errors(tmp_path, monkeypatch):
    """_guard_text_dim raises when on-disk dim differs from configured
    text_dim.
    """
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    import typer
    from mediasearch.cli import _guard_text_dim
    from mediasearch.store import Store

    store = Store(tmp_path / 'idx')
    store.text_dim = 512  # mismatch from the on-disk 768-d schema

    with pytest.raises(typer.BadParameter):
        _guard_text_dim(store)


def test_build_embedder_text_model_success(tmp_path, monkeypatch):
    """_build_embedder with text_model returns MLXTextEmbedder (no
    FAKE_EMBEDDER).
    """
    monkeypatch.delenv('MEDIASEARCH_FAKE_EMBEDDER', raising=False)
    import mediasearch.embedder as emb
    from mediasearch.cli import _build_embedder
    from mediasearch.config import Config, DEFAULT_TEXT_MODEL

    calls = {}

    class MockTextEmbedder:
        def __init__(self, model_name, batch_size):
            calls['model_name'] = model_name
            calls['batch_size'] = batch_size

    monkeypatch.setattr(emb, 'MLXTextEmbedder', MockTextEmbedder, raising=True)
    config = Config()
    result = _build_embedder(config, text_model=DEFAULT_TEXT_MODEL)

    assert isinstance(result, MockTextEmbedder)
    assert calls['model_name'] == DEFAULT_TEXT_MODEL


def test_search_empty_index_emits_no_results(tmp_path, monkeypatch):
    """Searching an empty index prints 'No results.' to stdout."""
    monkeypatch.setenv('MEDIASEARCH_FAKE_EMBEDDER', '1')
    idx = tmp_path / 'idx'
    r = runner.invoke(app, ['search', 'query', '--index-path', str(idx)])
    assert r.exit_code == 0, r.output
    assert 'No results.' in r.output
