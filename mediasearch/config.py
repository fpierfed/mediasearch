from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ModelSpec:
    dim: int
    input_size: int


MODEL_SPECS = {
    # SigLIP 2 xlarge / SO400M (~400M params, 1152-dim)
    'mlx-community/siglip2-so400m-patch16-384': ModelSpec(
        dim=1152, input_size=384
    ),
    'google/siglip2-so400m-patch16-256': ModelSpec(
        dim=1152, input_size=256
    ),
    'google/siglip2-so400m-patch16-512': ModelSpec(
        dim=1152, input_size=512
    ),
    # SigLIP 2 large (~300M params, 1024-dim)
    'google/siglip2-large-patch16-256': ModelSpec(
        dim=1024, input_size=256
    ),
    'google/siglip2-large-patch16-384': ModelSpec(
        dim=1024, input_size=384
    ),
    'google/siglip2-large-patch16-512': ModelSpec(
        dim=1024, input_size=512
    ),
    # SigLIP 2 base / medium (~86M params, 768-dim)
    'google/siglip2-base-patch16-384': ModelSpec(
        dim=768, input_size=384
    ),
    'google/siglip2-base-patch16-256': ModelSpec(
        dim=768, input_size=256
    ),
    'google/siglip2-base-patch16-512': ModelSpec(
        dim=768, input_size=512
    ),
    'mlx-community/siglip2-base-patch16-224-8bit': ModelSpec(
        dim=768, input_size=224
    ),
    # SigLIP 1 legacy (pre-converted by mlx-community)
    'mlx-community/siglip-so400m-patch14-384': ModelSpec(
        dim=1152, input_size=384
    ),
    'mlx-community/siglip-so400m-patch14-224': ModelSpec(
        dim=1152, input_size=224
    ),
    'mlx-community/siglip-large-patch16-384': ModelSpec(
        dim=1024, input_size=384
    ),
    'mlx-community/siglip-large-patch16-384-4bit': ModelSpec(
        dim=1024, input_size=384
    ),
}
MODEL_DIMS = {model: spec.dim for model, spec in MODEL_SPECS.items()}
MODEL_INPUT_SIZES = {
    model: spec.input_size for model, spec in MODEL_SPECS.items()
}
DEFAULT_MODEL = 'google/siglip2-base-patch16-256'
DEFAULT_TEXT_MODEL = 'mlx-community/multilingual-e5-base-mlx'
# whisper-small (~244M params) keeps the resident transcription model far
# smaller than large-v3-turbo (~809M) — a large RSS saving since this model
# stays loaded alongside the visual and text embedders for the whole run.
DEFAULT_AUDIO_MODEL = 'mlx-community/whisper-small-mlx'


def embed_dim_for(model: str) -> int:
    """Return the embedding dimension for the given model ID."""
    try:
        return MODEL_SPECS[model].dim
    except KeyError:
        known = ', '.join(sorted(MODEL_SPECS))
        raise ValueError(
            f"Unknown model '{model}'. Known models: {known}"
        ) from None


# Native input resolution (px) each visual model embeds at. Used to bound the
# size we decode images/frames to, since decoding larger than the model input
# is wasted memory.


def model_input_size_for(model: str) -> int:
    """Return the native input resolution (px) for the given model ID."""
    try:
        return MODEL_SPECS[model].input_size
    except KeyError:
        known = ', '.join(sorted(MODEL_SPECS))
        raise ValueError(
            f"Unknown model '{model}'. Known models: {known}"
        ) from None


EMBED_DIM = embed_dim_for(
    DEFAULT_MODEL
)  # kept for backward-compatible imports
TEXT_EMBED_DIM = 768
DEFAULT_INDEX_PATH = Path.home() / '.mediasearch' / 'index'

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.m4v', '.mkv', '.avi', '.webm'}


def classify_ext(path: Path) -> str | None:
    """Classify a file by extension as 'image' or 'video'."""

    ext = path.suffix.lower()
    if ext in IMAGE_EXTS:
        return 'image'
    if ext in VIDEO_EXTS:
        return 'video'
    return None


@dataclass
class Config:
    """
    Configuration

    At some point this will be backed by a real config file but for the moment
    it is not the case. Which means that this file is the configuration.
    """

    index_path: Path = DEFAULT_INDEX_PATH
    model: str = DEFAULT_MODEL
    audio_model: str = DEFAULT_AUDIO_MODEL
    frame_interval: float = 2.0
    dedup_threshold: int = 5
    # Cap the longer edge of decoded still images / video frames before
    # embedding. SigLIP downsamples to its native input size regardless, so
    # decoding larger is wasted memory: a 4K RGB frame is ~33 MB vs ~0.2 MB at
    # 256 px, and a batch holds batch_size of them at once. When left None,
    # both default to the selected model's input size in __post_init__.
    image_max_size: int | None = None
    frame_max_size: int | None = None
    batch_size: int = 16
    top_k: int = 20
    # Index audio transcripts for videos. Disabling it (CLI --no-audio) skips
    # loading the Whisper and text-embedding models entirely — the single
    # largest memory reduction for video-heavy libraries.
    index_audio: bool = True
    fts_score_k: float = (
        10.0  # BM25 half-saturation constant for score normalisation
    )

    def __post_init__(self) -> None:
        model_size = model_input_size_for(self.model)
        if self.image_max_size is None:
            self.image_max_size = model_size
        if self.frame_max_size is None:
            self.frame_max_size = model_size

    @property
    def embed_dim(self) -> int:
        return embed_dim_for(self.model)
