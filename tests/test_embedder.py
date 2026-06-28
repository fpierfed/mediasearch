import numpy as np
import pytest
from PIL import Image

from mediasearch.embedder import FakeEmbedder, l2_normalize
from mediasearch.config import EMBED_DIM


def test_l2_normalize_unit_length():
    """Test that l2_normalize correctly scales vectors to unit length."""
    v = np.array([[3.0, 4.0]], dtype=np.float32)
    out = l2_normalize(v)
    assert np.isclose(np.linalg.norm(out[0]), 1.0)


def test_l2_normalize_handles_zero_vector():
    """Test that l2_normalize handles zero vectors without dividing by zero."""
    v = np.zeros((1, 4), dtype=np.float32)
    out = l2_normalize(v)  # must not divide by zero
    assert np.all(np.isfinite(out))


def test_fake_text_is_deterministic_and_normalized():
    """Test that FakeEmbedder produces deterministic and normalized
    text embeddings.
    """
    e = FakeEmbedder()
    a = e.embed_texts(['a cat', 'a dog'])
    b = e.embed_texts(['a cat', 'a dog'])
    assert a.shape == (2, EMBED_DIM)
    assert np.allclose(a, b)  # deterministic
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0)
    assert not np.allclose(a[0], a[1])  # different inputs -> different vectors


def test_fake_same_image_same_vector():
    """Test that FakeEmbedder produces the same embedding for
    identical images.
    """
    e = FakeEmbedder()
    img = Image.new('RGB', (8, 8), (10, 20, 30))
    v1 = e.embed_images([img])
    v2 = e.embed_images([Image.new('RGB', (8, 8), (10, 20, 30))])
    assert np.allclose(v1, v2)


def test_mlx_text_embedder(monkeypatch):
    """Test that MLXTextEmbedder initializes and produces correct
    text embeddings.
    """
    import sys
    from unittest.mock import MagicMock
    from mediasearch.embedder import MLXTextEmbedder

    mock_load = MagicMock()

    class MockOutput:
        def __init__(self, text_embeds):
            self.text_embeds = text_embeds

    mock_model = MagicMock()
    mock_model.return_value = MockOutput(text_embeds=np.ones((2, 768)))
    mock_processor = MagicMock()
    mock_processor.return_value = {'input_ids': [1, 2, 3]}
    mock_load.return_value = (mock_model, mock_processor)

    mock_mlx = MagicMock()
    mock_mlx.load = mock_load
    monkeypatch.setitem(sys.modules, 'mlx_embeddings', mock_mlx)

    embedder = MLXTextEmbedder(dim=768)
    out = embedder.embed_texts(['hello', 'world'])

    assert out.shape == (2, 768)
    from mediasearch.config import DEFAULT_TEXT_MODEL

    mock_load.assert_called_once_with(DEFAULT_TEXT_MODEL)
    mock_processor.assert_called_once()
    mock_model.assert_called_once()
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_mlx_text_embedder_dimension_mismatch(monkeypatch):
    """Test that MLXTextEmbedder raises ValueError when output dimension
    mismatches expected.
    """
    import sys
    import pytest
    from unittest.mock import MagicMock
    from mediasearch.embedder import MLXTextEmbedder

    mock_load = MagicMock()

    class MockOutput:
        def __init__(self, text_embeds):
            self.text_embeds = text_embeds

    mock_model = MagicMock()
    mock_model.return_value = MockOutput(text_embeds=np.ones((2, 512)))
    mock_processor = MagicMock()
    mock_load.return_value = (mock_model, mock_processor)

    mock_mlx = MagicMock()
    mock_mlx.load = mock_load
    monkeypatch.setitem(sys.modules, 'mlx_embeddings', mock_mlx)

    embedder = MLXTextEmbedder(dim=768)

    with pytest.raises(
        ValueError, match='Expected text embedding dimension 768, got 512'
    ):
        embedder.embed_texts(['hello', 'world'])


def test_mlx_text_embedder_batching(monkeypatch):
    """Batch loop is exercised when texts exceed batch_size."""
    import sys
    from unittest.mock import MagicMock
    from mediasearch.embedder import MLXTextEmbedder

    mock_load = MagicMock()

    class MockOutput:
        def __init__(self, text_embeds):
            self.text_embeds = text_embeds

    mock_model = MagicMock()
    # Return 3 embeddings, one per call (batch_size=2 → 2 calls: 2 + 1)
    mock_model.side_effect = [
        MockOutput(text_embeds=np.ones((2, 768))),
        MockOutput(text_embeds=np.ones((1, 768))),
    ]
    mock_processor = MagicMock()
    mock_processor.return_value = {'input_ids': [1, 2, 3]}
    mock_load.return_value = (mock_model, mock_processor)

    mock_mlx = MagicMock()
    mock_mlx.load = mock_load
    monkeypatch.setitem(sys.modules, 'mlx_embeddings', mock_mlx)

    embedder = MLXTextEmbedder(batch_size=2, dim=768)
    out = embedder.embed_texts(['a', 'b', 'c'])  # 3 texts, 2 batches

    assert out.shape == (3, 768)
    assert mock_processor.call_count == 2  # two batches
    assert mock_model.call_count == 2
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_mlx_text_embedder_embed_images_raises(monkeypatch):
    """MLXTextEmbedder.embed_images always raises NotImplementedError."""
    import sys
    from unittest.mock import MagicMock
    from mediasearch.embedder import MLXTextEmbedder

    mock_load = MagicMock()
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_load.return_value = (mock_model, mock_processor)

    mock_mlx = MagicMock()
    mock_mlx.load = mock_load
    monkeypatch.setitem(sys.modules, 'mlx_embeddings', mock_mlx)

    embedder = MLXTextEmbedder(dim=768)
    dummy = Image.new('RGB', (8, 8))
    with pytest.raises(NotImplementedError, match='images'):
        embedder.embed_images([dummy])


def test_mlx_siglip_smoke(monkeypatch):
    """Smoke-test MLXSigLIPEmbedder text and image embedding when MLX
    is available.

    Skipped on machines without a working MLX install (e.g. CI, Intel Macs).
    """

    # Only run when MLX is actually importable and functional.
    pytest.importorskip(
        'mlx_embeddings', reason='mlx-embeddings not installed'
    )
    # Check for Metal / Apple Silicon — mlx needs it.
    try:
        import mlx.core as mx

        if mx.default_device().type != mx.DeviceType.gpu:
            pytest.skip(
                'MLX GPU device not available (likely Intel Mac or CI)'
            )
    except Exception:
        pytest.skip('MLX device check failed')

    from mediasearch.embedder import MLXSigLIPEmbedder
    from mediasearch.config import DEFAULT_MODEL

    embedder = MLXSigLIPEmbedder(model_name=DEFAULT_MODEL, batch_size=2)

    # Text path: ensure we can embed a small batch.
    out = embedder.embed_texts(['a cat', 'a dog'])
    assert out.shape[0] == 2
    assert out.shape[1] == embedder.dim
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)

    # Image path: ensure we can embed a small batch.
    img1 = Image.new('RGB', (64, 64), (200, 10, 10))
    img2 = Image.new('RGB', (64, 64), (10, 10, 200))
    img_out = embedder.embed_images([img1, img2])
    assert img_out.shape[0] == 2
    assert img_out.shape[1] == embedder.dim
    assert np.allclose(np.linalg.norm(img_out, axis=1), 1.0)
    # Different images should produce different embeddings
    assert not np.allclose(img_out[0], img_out[1])


def test_mlx_siglip_dimension_mismatch(monkeypatch):
    """MLXSigLIPEmbedder.embed_texts raises ValueError on dimension
    mismatch.
    """
    import sys
    from unittest.mock import MagicMock
    from mediasearch.embedder import MLXSigLIPEmbedder

    mock_load = MagicMock()
    mock_model = MagicMock()
    mock_processor = MagicMock()

    def _proc_side_effect(**kwargs):
        if 'images' in kwargs:
            return {'pixel_values': [1, 2, 3]}
        return {'input_ids': [1, 2, 3]}

    mock_processor.side_effect = _proc_side_effect
    mock_load.return_value = (mock_model, mock_processor)

    mock_mlx = MagicMock()
    mock_mlx.load = mock_load
    monkeypatch.setitem(sys.modules, 'mlx_embeddings', mock_mlx)

    # Mock _to_numpy to return an array with wrong dimension
    embedder = MLXSigLIPEmbedder(model_name='test', dim=1152)
    embedder._to_numpy = lambda embeds: np.ones((2, 768), dtype=np.float32)

    with pytest.raises(
        ValueError, match='Expected text embedding dimension 1152, got 768'
    ):
        embedder.embed_texts(['hello', 'world'])

    # Also cover the image dimension mismatch guard
    embedder2 = MLXSigLIPEmbedder(model_name='test', dim=1152)
    embedder2._to_numpy = lambda embeds: np.ones((2, 768), dtype=np.float32)

    dummy = Image.new('RGB', (8, 8))
    with pytest.raises(
        ValueError, match='Expected image embedding dimension 1152, got 768'
    ):
        embedder2.embed_images([dummy, dummy])
