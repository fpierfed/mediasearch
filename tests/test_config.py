from pathlib import Path

from mediasearch.config import (
    Config,
    EMBED_DIM,
    classify_ext,
    IMAGE_EXTS,
    VIDEO_EXTS,
)


def test_defaults():
    """Test that Config initializes with expected default values."""
    c = Config()
    assert c.frame_interval == 2.0
    assert c.dedup_threshold == 5
    assert c.batch_size == 16
    assert c.top_k == 20
    assert c.model == 'google/siglip2-base-patch16-256'
    assert c.index_path == Path.home() / '.mediasearch' / 'index'
    assert EMBED_DIM == 768


def test_overrides():
    """Test that Config allows overriding default values."""
    c = Config(frame_interval=1.0, top_k=5)
    assert c.frame_interval == 1.0
    assert c.top_k == 5


def test_classify_ext():
    """Test that classify_ext correctly identifies image and video
    extensions.
    """
    assert classify_ext(Path('a.JPG')) == 'image'
    assert classify_ext(Path('b.mov')) == 'video'
    assert classify_ext(Path('c.txt')) is None
    assert '.heic' in IMAGE_EXTS and '.mp4' in VIDEO_EXTS


def test_embed_dim_property_and_resolver():
    """Test that embed_dim property matches the dimension for known models."""
    from mediasearch.config import Config, embed_dim_for

    assert Config().embed_dim == 768
    # 768-dim (base/medium)
    assert (
        Config(model='google/siglip2-base-patch16-384').embed_dim == 768
    )
    assert embed_dim_for('google/siglip2-base-patch16-384') == 768
    assert (
        Config(model='google/siglip2-base-patch16-256').embed_dim == 768
    )
    assert embed_dim_for('google/siglip2-base-patch16-256') == 768
    # 1024-dim (large)
    assert Config(model='google/siglip2-large-patch16-384').embed_dim == 1024
    assert embed_dim_for('google/siglip2-large-patch16-384') == 1024
    assert (
        Config(model='mlx-community/siglip-large-patch16-384').embed_dim
        == 1024
    )
    # 1152-dim (xlarge / SO400M)
    assert Config(model='google/siglip2-so400m-patch16-256').embed_dim == 1152
    assert embed_dim_for('google/siglip2-so400m-patch16-512') == 1152
    # quantised
    assert (
        Config(model='mlx-community/siglip-large-patch16-384-4bit').embed_dim
        == 1024
    )
    assert (
        Config(model='mlx-community/siglip2-base-patch16-224-8bit').embed_dim
        == 768
    )


def test_embed_dim_unknown_model_raises():
    """Test that embed_dim_for raises ValueError for unknown models."""
    import pytest
    from mediasearch.config import embed_dim_for

    with pytest.raises(ValueError):
        embed_dim_for('not-a-real-model')


def test_model_input_size_for_known_models():
    from mediasearch.config import model_input_size_for

    assert model_input_size_for('google/siglip2-base-patch16-256') == 256
    assert model_input_size_for('google/siglip2-base-patch16-512') == 512
    assert (
        model_input_size_for('mlx-community/siglip2-base-patch16-224-8bit')
        == 224
    )
    assert (
        model_input_size_for('mlx-community/siglip2-so400m-patch16-384') == 384
    )


def test_config_defaults_bound_media_to_default_model_size():
    from mediasearch.config import Config

    c = Config()
    assert c.image_max_size == 256
    assert c.frame_max_size == 256
    assert c.index_audio is True


def test_config_explicit_media_caps_override_model_size():
    from mediasearch.config import Config

    c = Config(image_max_size=384, frame_max_size=512, index_audio=False)
    assert c.image_max_size == 384
    assert c.frame_max_size == 512
    assert c.index_audio is False
