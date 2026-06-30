"""
Embedder

We support three embedders:
    * FakeEmbedder, used for tests
    * MLXSigLIPEmbedder, used for images and videos
    * MLXTextEmbedder, used for text
"""

from abc import abstractmethod
import hashlib
from typing import Protocol, runtime_checkable, Any

import numpy as np
from PIL import Image

from .config import DEFAULT_TEXT_MODEL, EMBED_DIM, TEXT_EMBED_DIM


def mlx_load(model_name: str):
    """
    Load an MLX embeddings model when a real embedder is constructed.

    We do this since loading this module can take ~2 seconds...
    """
    from mlx_embeddings import load

    return load(model_name)


def l2_normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vectors in an array to unit length (L2 norm)."""

    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (v / norms).astype(np.float32)


@runtime_checkable
class Embedder(Protocol):
    dim: int

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError

    def _to_numpy(self, embeds: Any) -> np.ndarray:
        """Convert MLX arrays or PyTorch tensors to NumPy arrays."""
        return np.array(embeds, dtype=np.float32)


class FakeEmbedder(Embedder):
    """
    Deterministic, model-free embedder for tests. Same content -> same vector.
    """

    def __init__(self, dim: int = EMBED_DIM):
        """Initialise FakeEmbedder with a specific embedding dimension."""
        self.dim = dim

    def _vec(self, key: bytes) -> np.ndarray:
        """Generate a deterministic random vector from a byte key."""
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Return fake, deterministic embeddings for a list of texts."""
        v = np.stack([self._vec(b'text:' + t.encode('utf-8')) for t in texts])
        return l2_normalize(v)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Return fake, deterministic embeddings for a list of images."""
        v = np.stack(
            [
                self._vec(
                    b'img:'
                    + (im if im.mode == 'RGB' else im.convert('RGB')).tobytes()
                )
                for im in images
            ]
        )
        return l2_normalize(v)


class MLXSigLIPEmbedder(Embedder):
    """
    SigLIP 2 embeddings via mlx-embeddings. Loads the model+processor once.
    """

    def __init__(
        self, model_name: str, batch_size: int = 16, dim: int = EMBED_DIM
    ):
        """Initialise the model and processor from mlx-embeddings."""

        self.model, self.processor = mlx_load(model_name)
        self.batch_size = batch_size
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into normalized vectors using SigLIP 2."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.processor(
                text=batch,
                padding='max_length',
                max_length=64,
                truncation=True,
                return_tensors='mlx',
            )
            embeds = self.model.get_text_features(
                input_ids=inputs['input_ids']
            )
            out.append(self._to_numpy(embeds))
        out_normalized = l2_normalize(np.vstack(out))
        if out_normalized.shape[1] != self.dim:
            raise ValueError(
                f'Expected text embedding dimension {self.dim}, '
                + f'got {out_normalized.shape[1]}'
            )
        return out_normalized

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Embed a list of images into normalized vectors using SigLIP 2."""
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = [
                im if im.mode == 'RGB' else im.convert('RGB')
                for im in images[i : i + self.batch_size]
            ]
            inputs = self.processor(images=batch, return_tensors='mlx')
            embeds = self.model.get_image_features(
                pixel_values=inputs['pixel_values']
            )
            out.append(self._to_numpy(embeds))
        out_normalized = l2_normalize(np.vstack(out))
        if out_normalized.shape[1] != self.dim:
            raise ValueError(
                f'Expected image embedding dimension {self.dim}, '
                + f'got {out_normalized.shape[1]}'
            )
        return out_normalized


class MLXTextEmbedder(Embedder):
    """Text embeddings (e.g. multilingual-e5) via mlx-embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        batch_size: int = 16,
        dim: int = TEXT_EMBED_DIM,
    ):
        """Initialise the text model and processor from mlx-embeddings."""

        self.model, self.processor = mlx_load(model_name)
        self.batch_size = batch_size
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into normalized vectors."""
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.processor(
                text=batch,
                padding='max_length',
                max_length=64,
                truncation=True,
                return_tensors='mlx',
            )
            embeds = self.model(**inputs).text_embeds
            out.append(self._to_numpy(embeds))
        out_normalized = l2_normalize(np.vstack(out))
        if out_normalized.shape[1] != self.dim:
            raise ValueError(
                f'Expected text embedding dimension {self.dim}, '
                + f'got {out_normalized.shape[1]}'
            )
        return out_normalized

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        """Not supported. Raises NotImplementedError."""
        raise NotImplementedError('MLXTextEmbedder does not support images')
