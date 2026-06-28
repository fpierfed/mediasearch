from mediasearch.config import Config
from mediasearch.embedder import FakeEmbedder
from mediasearch.pipeline import index
from mediasearch.search import (
    format_timestamp,
    search_clip,
    search_image,
    search_text,
)
from mediasearch.store import Store


def test_format_timestamp():
    """Test that format_timestamp correctly formats seconds to MM:SS."""
    assert format_timestamp(0) == '0:00'
    assert format_timestamp(62) == '1:02'
    assert format_timestamp(125.6) == '2:06'


def _library(tmp_path, make_image, sample_video):
    """Helper function to create a populated test library and index it."""
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png', (200, 10, 10))
    make_image(lib / 'b.png', (10, 10, 200))
    (lib / 'clip.mp4').write_bytes(sample_video.read_bytes())
    config = Config()
    store = Store(tmp_path / 'idx')
    index(config, FakeEmbedder(), FakeEmbedder(), store, [lib])
    return config, store, lib


def test_search_image_returns_exact_match_first(
    tmp_path, make_image, sample_video
):
    """Test that an image query returns the identical image as its best result."""
    config, store, lib = _library(tmp_path, make_image, sample_video)
    results = search_image(lib / 'a.png', config, FakeEmbedder(), store)
    assert results[0]['path'] == str(lib / 'a.png')
    assert results[0]['rank'] == 1
    assert results[0]['score'] >= 0.99


def test_results_are_grouped_one_row_per_file(
    tmp_path, make_image, sample_video
):
    """Test that search results contain only one best match per media file."""
    config, store, lib = _library(tmp_path, make_image, sample_video)
    results = search_image(lib / 'a.png', config, FakeEmbedder(), store)
    paths = [r['path'] for r in results]
    assert len(paths) == len(
        set(paths)
    )  # the multi-frame clip appears at most once


def test_video_result_has_timestamp(tmp_path, make_image, sample_video):
    """Test that video search results include a formatted timestamp."""
    config, store, lib = _library(tmp_path, make_image, sample_video)
    results = search_clip(lib / 'clip.mp4', config, FakeEmbedder(), store)
    top = results[0]
    assert top['path'] == str(lib / 'clip.mp4')  # a clip matches itself best
    assert top['media_type'] == 'video'
    assert top['time'] is not None  # "m:ss" string present for video


def test_search_text_runs_and_is_structured(
    tmp_path, make_image, sample_video
):
    """Test that text search yields properly structured result dictionaries."""
    config, store, lib = _library(tmp_path, make_image, sample_video)
    results = search_text(
        'anything',
        config,
        FakeEmbedder(),
        FakeEmbedder(dim=768),
        store,
        top_k=2,
    )
    assert len(results) <= 2
    assert {
        'rank',
        'score',
        'path',
        'media_type',
        'timestamp',
        'time',
        'modality',
    } <= set(results[0])


def test_search_clip_empty_or_corrupt_clip_returns_empty(tmp_path, make_image):
    """Test that search_clip returns an empty list for a bad video file instead of raising."""
    # A file with a video extension but no decodable frames must not crash search_clip.
    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    bad = lib / 'bad.mp4'
    bad.write_bytes(b'not a real video')
    config = Config()
    store = Store(tmp_path / 'idx')
    index(config, FakeEmbedder(), FakeEmbedder(), store, [lib])
    # querying with the corrupt clip should yield no results, not raise
    assert search_clip(bad, config, FakeEmbedder(), store) == []


def test_search_text_aggregates_scores_and_modalities():
    """Verify _group_by_media_multi correctly combines visual and audio scores.

    NOTE: MockStore returns raw scores directly, bypassing _normalize_scores
    (tested separately in test_store.py).  In production, FTS scores are
    normalised via s/(s+k), so the raw 0.2 from search_transcripts_fts would
    land around ~0.02.  This test exercises aggregation, not normalisation.
    """

    class MockStore:
        def search(self, vec, top_k, media_type):
            return [
                {
                    'media_path': 'clip.mp4',
                    'score': 0.5,
                    'media_type': 'video',
                    'timestamp': 10.0,
                }
            ]

        def search_transcripts_vector(self, vec, top_k):
            return [
                {
                    'media_path': 'clip.mp4',
                    'score': 0.4,
                    'media_type': 'transcript',
                    'start_time': 10.0,
                }
            ]

        def search_transcripts_fts(self, query, top_k):
            return [
                {
                    'media_path': 'clip.mp4',
                    'score': 0.2,
                    'media_type': 'transcript',
                    'start_time': 12.0,
                }
            ]

    config = Config()
    store = MockStore()
    results = search_text(
        'query', config, FakeEmbedder(), FakeEmbedder(dim=768), store, top_k=2
    )

    assert len(results) == 1
    top = results[0]
    # best visual is 0.5, best audio is max(0.4, 0.2) = 0.4. Sum is 0.9.
    assert top['score'] == 0.9
    assert top['modality'] == '[VISUAL][AUDIO]'
    assert top['path'] == 'clip.mp4'


def test_search_clip_no_frames(tmp_path, make_image, monkeypatch):
    """search_clip returns [] when sample_video yields no frames (but doesn't raise)."""
    from mediasearch.config import Config
    from mediasearch.embedder import FakeEmbedder
    from mediasearch.store import Store

    monkeypatch.setattr('mediasearch.search.sample_video', lambda *a, **kw: [])

    lib = tmp_path / 'lib'
    lib.mkdir()
    make_image(lib / 'a.png')
    config = Config()
    store = Store(tmp_path / 'idx')
    # Index something so the store exists, then query with a "video" path
    from mediasearch.pipeline import index

    index(config, FakeEmbedder(), FakeEmbedder(), store, [lib])

    result = search_clip(lib / 'dummy.mp4', config, FakeEmbedder(), store)
    assert result == []
