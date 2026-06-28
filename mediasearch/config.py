from dataclasses import dataclass
from pathlib import Path

MODEL_DIMS = {
    # ── SigLIP 2 xlarge / SO400M (~400M params, 1152-dim) ──────────────
    'mlx-community/siglip2-so400m-patch16-384': 1152,  # default · 384 px
    'google/siglip2-so400m-patch16-256': 1152,  # faster  · 256 px
    'google/siglip2-so400m-patch16-512': 1152,  # precise · 512 px
    # ── SigLIP 2 large (~300M params, 1024-dim) ─────────────────────────
    'google/siglip2-large-patch16-256': 1024,
    'google/siglip2-large-patch16-384': 1024,
    'google/siglip2-large-patch16-512': 1024,
    # ── SigLIP 2 base / medium (~86M params, 768-dim) ───────────────────
    'mlx-community/siglip2-base-patch16-384': 768,
    'mlx-community/siglip2-base-patch16-256': 768,
    'google/siglip2-base-patch16-512': 768,
    'mlx-community/siglip2-base-patch16-224-8bit': 768,  # 8-bit quantised
    # ── SigLIP 1 legacy (pre-converted by mlx-community) ────────────────
    'mlx-community/siglip-so400m-patch14-384': 1152,
    'mlx-community/siglip-so400m-patch14-224': 1152,
    'mlx-community/siglip-large-patch16-384': 1024,
    'mlx-community/siglip-large-patch16-384-4bit': 1024,  # 4-bit quantised
}
DEFAULT_MODEL = 'mlx-community/siglip2-so400m-patch16-384'
DEFAULT_TEXT_MODEL = 'mlx-community/multilingual-e5-base-mlx'
DEFAULT_AUDIO_MODEL = 'mlx-community/whisper-large-v3-turbo'


def embed_dim_for(model: str) -> int:
    """Return the embedding dimension for the given model ID."""
    try:
        return MODEL_DIMS[model]
    except KeyError:
        known = ', '.join(sorted(MODEL_DIMS))
        raise ValueError(
            f"Unknown model '{model}'. Known models: {known}"
        ) from None


EMBED_DIM = embed_dim_for(
    DEFAULT_MODEL
)  # 1152; kept for backward-compatible imports
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
    index_path: Path = DEFAULT_INDEX_PATH
    model: str = DEFAULT_MODEL
    audio_model: str = DEFAULT_AUDIO_MODEL
    frame_interval: float = 2.0
    dedup_threshold: int = 5
    batch_size: int = 16
    top_k: int = 20
    fts_score_k: float = (
        10.0  # BM25 half-saturation constant for score normalisation
    )

    @property
    def embed_dim(self) -> int:
        return embed_dim_for(self.model)
