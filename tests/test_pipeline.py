from mediasearch.config import Config
from mediasearch.embedder import FakeEmbedder
from mediasearch.pipeline import index
from mediasearch.store import Store


def _index(tmp_path, roots):
    """Helper function to create a config, store, and fake embedder,
    then run indexing.
    """
    config = Config()
    store = Store(tmp_path / 'idx')
    embedder = FakeEmbedder()
    index(config, embedder, embedder, store, roots)
    return store


def test_index_images_and_video(tmp_path, make_image, sample_video):
    """Test that indexing processes both images and videos correctly."""
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png', (200, 10, 10))
    make_image(lib / 'b.png', (10, 200, 10))
    (lib / 'clip.mp4').write_bytes(sample_video.read_bytes())

    store = _index(tmp_path, [lib])
    st = store.stats()
    assert st['files'] == 3
    assert st['done'] == 3
    assert st['vectors'] >= 4  # 2 images + >=2 video frames


def test_reindex_skips_unchanged(tmp_path, make_image):
    """Test that indexing an already indexed and unchanged file skips
    processing.
    """
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    config = Config()
    store = Store(tmp_path / 'idx')
    index(config, FakeEmbedder(), FakeEmbedder(), store, [lib])
    first_indexed_at = store.manifest()['%s' % (lib / 'a.png')]['indexed_at']

    index(
        config, FakeEmbedder(), FakeEmbedder(), store, [lib]
    )  # second run, nothing changed
    second_indexed_at = store.manifest()['%s' % (lib / 'a.png')]['indexed_at']
    assert (
        first_indexed_at == second_indexed_at
    )  # row was NOT rewritten -> skipped


def test_resume_reprocesses_pending(tmp_path, make_image):
    """Test that files marked as pending are reprocessed on the next run."""
    lib = tmp_path / 'lib'
    lib.mkdir()
    p = make_image(lib / 'a.png')
    config = Config()
    store = Store(tmp_path / 'idx')
    index(config, FakeEmbedder(), FakeEmbedder(), store, [lib])

    # Simulate a crash mid-file: vectors gone, manifest left 'pending'
    store.emb.delete("media_path = '%s'" % p)
    mf = store.manifest()['%s' % p]
    store.set_file(
        path=str(p),
        mtime=mf['mtime'],
        size=mf['size'],
        media_type='image',
        status='pending',
    )

    index(config, FakeEmbedder(), FakeEmbedder(), store, [lib])
    assert store.manifest()['%s' % p]['status'] == 'done'
    assert store.count_vectors(str(p)) == 1


def test_bad_video_is_marked_error_and_run_continues(tmp_path, make_image):
    """Test that a corrupt video is marked with an error status but
    doesn't halt indexing.
    """
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'good.png')
    (lib / 'bad.mp4').write_bytes(b'not a real video')

    store = _index(tmp_path, [lib])
    m = store.manifest()
    assert m['%s' % (lib / 'good.png')]['status'] == 'done'
    assert m['%s' % (lib / 'bad.mp4')]['status'] == 'error'


def test_process_audio_success(tmp_path, sample_video, monkeypatch):
    """Test that _process_audio correctly extracts and formats audio
    transcripts.
    """
    import mlx_whisper
    from mediasearch.config import DEFAULT_AUDIO_MODEL
    from mediasearch.pipeline import _process_audio
    from mediasearch.embedder import FakeEmbedder
    from mediasearch.walker import MediaFile

    lib = tmp_path / 'lib'
    lib.mkdir()
    vid = lib / 'clip.mp4'
    vid.write_bytes(sample_video.read_bytes())
    mf = MediaFile(path=vid, mtime=0.0, size=100, media_type='video')

    def mock_transcribe(path, path_or_hf_repo, **kwargs):
        return {
            'segments': [
                {'text': 'Hello world', 'start': 0.0, 'end': 2.0},
                {'text': 'This is a test', 'start': 2.0, 'end': 4.5},
            ]
        }

    monkeypatch.setattr(mlx_whisper, 'transcribe', mock_transcribe)

    text_embedder = FakeEmbedder()
    rows = _process_audio(mf, text_embedder, audio_model=DEFAULT_AUDIO_MODEL)

    assert len(rows) == 2
    assert rows[0]['media_type'] == 'transcript'
    assert rows[0]['text'] == 'Hello world'
    assert rows[0]['start_time'] == 0.0
    assert rows[0]['end_time'] == 2.0
    assert len(rows[0]['vector']) == text_embedder.dim

    assert rows[1]['text'] == 'This is a test'
    assert rows[1]['start_time'] == 2.0
    assert rows[1]['end_time'] == 4.5


def test_process_audio_empty_segments(tmp_path, sample_video, monkeypatch):
    """_process_audio returns [] when transcription yields no segments."""
    import mlx_whisper
    from mediasearch.config import DEFAULT_AUDIO_MODEL
    from mediasearch.pipeline import _process_audio
    from mediasearch.embedder import FakeEmbedder
    from mediasearch.walker import MediaFile

    lib = tmp_path / 'lib'
    lib.mkdir()
    vid = lib / 'clip.mp4'
    vid.write_bytes(sample_video.read_bytes())
    mf = MediaFile(path=vid, mtime=0.0, size=100, media_type='video')

    monkeypatch.setattr(
        mlx_whisper, 'transcribe', lambda *a, **kw: {'segments': []}
    )

    rows = _process_audio(mf, FakeEmbedder(), audio_model=DEFAULT_AUDIO_MODEL)
    assert rows == []


def test_process_audio_failure(tmp_path, sample_video, monkeypatch):
    """_process_audio returns [] and logs error when transcription raises."""
    import mlx_whisper
    from mediasearch.config import DEFAULT_AUDIO_MODEL
    from mediasearch.pipeline import _process_audio
    from mediasearch.embedder import FakeEmbedder
    from mediasearch.walker import MediaFile

    lib = tmp_path / 'lib'
    lib.mkdir()
    vid = lib / 'clip.mp4'
    vid.write_bytes(sample_video.read_bytes())
    mf = MediaFile(path=vid, mtime=0.0, size=100, media_type='video')

    def _raise(*a, **kw):
        raise RuntimeError('boom')

    monkeypatch.setattr(mlx_whisper, 'transcribe', _raise)

    rows = _process_audio(mf, FakeEmbedder(), audio_model=DEFAULT_AUDIO_MODEL)
    assert rows == []  # swallowed, not raised


def test_process_no_frames_for_video(tmp_path, monkeypatch):
    """_process returns 0 when a video yields no frames."""
    from mediasearch.pipeline import _process
    from mediasearch.config import Config
    from mediasearch.embedder import FakeEmbedder
    from mediasearch.store import Store
    from mediasearch.walker import MediaFile

    def _empty_iter(*a, **kw):
        yield from ()  # empty generator

    monkeypatch.setattr('mediasearch.pipeline.sample_video', _empty_iter)

    mf = MediaFile(
        path=tmp_path / 'empty.mp4', mtime=0.0, size=0, media_type='video'
    )
    store = Store(tmp_path / 'idx')
    n = _process(mf, Config(), FakeEmbedder(), store)
    assert n == 0


def test_reindex_skip_calls_progress(tmp_path, make_image):
    """Progress callback fires for unchanged files that are skipped."""
    from mediasearch.config import Config
    from mediasearch.embedder import FakeEmbedder
    from mediasearch.pipeline import index
    from mediasearch.store import Store

    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    config = Config()
    store = Store(tmp_path / 'idx')

    # First run: index the file
    index(config, FakeEmbedder(), FakeEmbedder(), store, [lib])

    # Second run: everything unchanged — progress should fire for each
    # file walked
    skipped = []
    index(
        config,
        FakeEmbedder(),
        FakeEmbedder(),
        store,
        [lib],
        progress=lambda: skipped.append(1),
    )

    assert len(skipped) >= 1  # the unchanged file triggered progress


def test_text_embedder_factory_not_called_for_images(tmp_path, make_image):
    """An image-only run never resolves the lazy text-embedder factory."""
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    make_image(lib / 'b.png')

    calls = []

    def factory():
        calls.append(1)
        return FakeEmbedder()

    config = Config()
    store = Store(tmp_path / 'idx')
    index(config, FakeEmbedder(), factory, store, [lib])

    assert calls == []  # text model never loaded for images
    assert store.stats()['done'] == 2


def test_text_embedder_factory_resolved_once_for_videos(
    tmp_path, sample_video, monkeypatch
):
    """The factory is resolved lazily and memoised across multiple videos."""
    import mlx_whisper

    # Avoid loading a real whisper model; we only care about the factory.
    monkeypatch.setattr(
        mlx_whisper, 'transcribe', lambda *a, **kw: {'segments': []}
    )

    lib = tmp_path / 'lib'
    lib.mkdir()
    (lib / 'a.mp4').write_bytes(sample_video.read_bytes())
    (lib / 'b.mp4').write_bytes(sample_video.read_bytes())

    calls = []

    def factory():
        calls.append(1)
        return FakeEmbedder()

    config = Config()
    store = Store(tmp_path / 'idx')
    index(config, FakeEmbedder(), factory, store, [lib])

    assert len(calls) == 1  # built once, reused for the second video


def test_process_video_writes_incrementally(tmp_path, sample_video):
    """_process writes video embeddings in batches, not all at once."""
    from mediasearch.pipeline import _process
    from mediasearch.config import Config
    from mediasearch.embedder import FakeEmbedder
    from mediasearch.store import Store
    from mediasearch.walker import MediaFile

    lib = tmp_path / 'lib'
    lib.mkdir()
    vid = lib / 'clip.mp4'
    vid.write_bytes(sample_video.read_bytes())

    mf = MediaFile(path=vid, mtime=vid.stat().st_mtime, size=vid.stat().st_size, media_type='video')
    config = Config()
    config.batch_size = 1  # single-frame batches to force multiple writes
    store = Store(tmp_path / 'idx')
    embedder = FakeEmbedder()

    # Track how many times add_embeddings is called
    calls = []
    orig_add = store.add_embeddings

    def _track(rows):
        calls.append(len(rows))
        return orig_add(rows)

    store.add_embeddings = _track

    n = _process(mf, config, embedder, store)
    assert n > 0  # some frames embedded
    assert n == sum(calls)  # total rows = sum of batch sizes
    assert len(calls) >= 2  # at least 2 batches written with batch_size=1
