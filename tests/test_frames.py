import pytest
from PIL import Image

from mediasearch.frames import Frame, dedup, extract_frames


def _frame(color, ts, idx):
    """Helper function to create a mock Frame with a solid color image."""
    return Frame(
        image=Image.new('RGB', (32, 32), color), timestamp=ts, frame_idx=idx
    )


def test_dedup_collapses_identical_frames():
    """Test that dedup collapses identical consecutive frames."""
    frames = [_frame((255, 255, 255), float(i), i) for i in range(5)]
    kept = list(dedup(frames, threshold=5))
    assert len(kept) == 1
    assert kept[0].timestamp == 0.0  # keeps the first of a run


def test_dedup_keeps_distinct_frames():
    """Test that dedup keeps frames with distinct colors."""
    frames = [
        _frame((255, 255, 255), 0.0, 0),
        _frame((0, 0, 0), 1.0, 1),
        _frame((255, 255, 255), 2.0, 2),
    ]
    kept = list(dedup(frames, threshold=5))
    assert len(kept) == 3


def test_extract_frames_returns_timestamps(sample_video):
    """Test that extract_frames returns monotonically increasing timestamps."""
    frames = list(extract_frames(sample_video, interval=2.0))
    assert len(frames) >= 2
    ts = [f.timestamp for f in frames]
    assert ts == sorted(ts)  # monotonically increasing
    assert ts[0] >= 0.0
    assert all(f.image.mode == 'RGB' for f in frames)


def test_sample_video_dedups_two_scenes(sample_video):
    """Test that sample_video successfully extracts and deduplicates
    frames from a multi-scene video.
    """
    from mediasearch.frames import sample_video as sample_fn

    kept = list(sample_fn(sample_video, interval=2.0, dedup_threshold=5))
    assert len(kept) >= 2


def test_extract_frames_yields_rgb_frames_after_pool(sample_video):
    """Frames are still valid RGB images after the yield moved past the
    autorelease pool."""
    frames = list(extract_frames(sample_video, interval=2.0, max_size=32))
    assert len(frames) >= 2
    assert all(f.image.mode == 'RGB' for f in frames)
    assert all(max(f.image.size) <= 32 for f in frames)


def test_dedup_is_lazy():
    """dedup does not consume more input than needed for first yield."""
    consumed = []

    def _track(iterable):
        for item in iterable:
            consumed.append(item)
            yield item

    frames = [
        _frame((255, 255, 255), 0.0, 0),
        _frame((0, 0, 0), 1.0, 1),  # distinct — will be yielded
    ]
    gen = dedup(_track(frames), threshold=5)
    first = next(gen)
    assert first.timestamp == 0.0
    # At this point only the first frame has been consumed — dedup yields
    # the first frame immediately (last_hash is None) without peeking ahead.
    assert len(consumed) == 1


def test_cgimage_to_pil_non_32bpp(monkeypatch):
    """_cgimage_to_pil handles non-32bpp images via raw RGB decoder."""
    from mediasearch.frames import _cgimage_to_pil

    # 2x2 RGB image = 12 bytes (2 rows × 6 bpr)
    raw = bytes([255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 255, 0])

    monkeypatch.setattr('mediasearch.frames.CGImageGetWidth', lambda cg: 2)
    monkeypatch.setattr('mediasearch.frames.CGImageGetHeight', lambda cg: 2)
    monkeypatch.setattr(
        'mediasearch.frames.CGImageGetBytesPerRow', lambda cg: 6
    )
    monkeypatch.setattr(
        'mediasearch.frames.CGImageGetBitsPerPixel', lambda cg: 24
    )
    monkeypatch.setattr(
        'mediasearch.frames.CGImageGetDataProvider', lambda cg: object()
    )
    monkeypatch.setattr(
        'mediasearch.frames.CGDataProviderCopyData', lambda p: raw
    )

    result = _cgimage_to_pil('fake_cgimage')
    assert result.mode == 'RGB'
    assert result.size == (2, 2)


def test_extract_frames_respects_max_size(sample_video):
    """max_size caps the longer edge of decoded frames (downscaling)."""
    # The fixture video is 64x64; cap to 32 so scaling must take effect.
    frames = list(extract_frames(sample_video, interval=2.0, max_size=32))
    assert len(frames) >= 2
    for f in frames:
        assert max(f.image.size) <= 32
        assert f.image.mode == 'RGB'


def test_extract_frames_no_max_size_keeps_native(sample_video):
    """Without max_size, frames keep their native resolution."""
    frames = list(extract_frames(sample_video, interval=2.0))
    assert len(frames) >= 2
    assert all(f.image.size == (64, 64) for f in frames)


def test_extract_frames_nil_url(tmp_path, monkeypatch):
    """extract_frames raises ValueError when NSURL creation fails."""
    import mediasearch.frames as mf

    class _NilNSURL:
        @staticmethod
        def fileURLWithPath_(_path):
            return None

    monkeypatch.setattr(mf, 'NSURL', _NilNSURL)
    with pytest.raises(ValueError, match='NSURL'):
        list(mf.extract_frames(tmp_path / 'x.mp4', interval=2.0))


def test_extract_frames_nil_asset(sample_video, monkeypatch):
    """extract_frames raises ValueError when AVURLAsset creation fails."""
    import mediasearch.frames as mf

    class _NilAsset:
        @staticmethod
        def URLAssetWithURL_options_(_url, _opts):
            return None

    monkeypatch.setattr(mf, 'AVURLAsset', _NilAsset)
    with pytest.raises(ValueError, match='asset'):
        list(mf.extract_frames(sample_video, interval=2.0))


def test_extract_frames_nil_generator(sample_video, monkeypatch):
    """extract_frames raises ValueError when image generator creation fails."""
    import mediasearch.frames as mf

    class _NilGenerator:
        @staticmethod
        def assetImageGeneratorWithAsset_(_asset):
            return None

    monkeypatch.setattr(mf, 'AVAssetImageGenerator', _NilGenerator)
    with pytest.raises(ValueError, match='image generator'):
        list(mf.extract_frames(sample_video, interval=2.0))


def test_extract_frames_handles_frame_error(sample_video, monkeypatch):
    """Frame decode errors are caught and logged, not propagated."""
    import mediasearch.frames as mf

    def boom(*a, **kw):
        raise RuntimeError('simulated decode failure')

    monkeypatch.setattr(mf, '_cgimage_to_pil', boom)
    frames = list(mf.extract_frames(sample_video, interval=2.0))
    # Every frame decode failed, so we get zero frames back — but no crash.
    assert frames == []
