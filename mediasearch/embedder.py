from __future__ import annotations

import hashlib
from typing import Protocol, runtime_checkable, Any

import numpy as np
from PIL import Image

from .config import DEFAULT_TEXT_MODEL, EMBED_DIM


def l2_normalize(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (v / norms).astype(np.float32)


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...
    def embed_images(self, images: list[Image.Image]) -> np.ndarray: ...


class FakeEmbedder:
    """Deterministic, model-free embedder for tests. Same content -> same vector."""

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def _vec(self, key: bytes) -> np.ndarray:
        seed = int.from_bytes(hashlib.sha256(key).digest()[:8], 'little')
        rng = np.random.default_rng(seed)
        return rng.standard_normal(self.dim).astype(np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        v = np.stack([self._vec(b'text:' + t.encode('utf-8')) for t in texts])
        return l2_normalize(v)

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        v = np.stack(
            [self._vec(b'img:' + im.convert('RGB').tobytes()) for im in images]
        )
        return l2_normalize(v)


class MLXSigLIPEmbedder:
    """SigLIP 2 embeddings via mlx-embeddings. Loads the model+processor once."""

    def __init__(
        self, model_name: str, batch_size: int = 16, dim: int = EMBED_DIM
    ):
        from mlx_embeddings import load

        self.model, self.processor = load(model_name)
        self.batch_size = batch_size
        self.dim = dim

    def _to_numpy(self, embeds: Any) -> np.ndarray:
        # mlx arrays, torch tensors, and numpy all support np.array(...)
        return np.array(embeds, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.processor(
                text=batch,
                padding='max_length',
                max_length=64,
                return_tensors='mlx',
            )
            embeds = self.model.get_text_features(
                input_ids=inputs['input_ids']
            )
            out.append(self._to_numpy(embeds))
        out_normalized = l2_normalize(np.vstack(out))
        if out_normalized.shape[1] != self.dim:
            raise ValueError(
                f'Expected text embedding dimension {self.dim}, got {out_normalized.shape[1]}'
            )
        return out_normalized

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        out = []
        for i in range(0, len(images), self.batch_size):
            batch = [
                im.convert('RGB') for im in images[i : i + self.batch_size]
            ]
            inputs = self.processor(images=batch, return_tensors='mlx')
            embeds = self.model.get_image_features(
                pixel_values=inputs['pixel_values']
            )
            out.append(self._to_numpy(embeds))
        out_normalized = l2_normalize(np.vstack(out))
        if out_normalized.shape[1] != self.dim:
            raise ValueError(
                f'Expected image embedding dimension {self.dim}, got {out_normalized.shape[1]}'
            )
        return out_normalized


class MLXTextEmbedder:
    """Text embeddings (e.g. multilingual-e5) via mlx-embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_MODEL,
        batch_size: int = 16,
        dim: int = 768,
    ):
        from mlx_embeddings import load

        self.model, self.processor = load(model_name)
        self.batch_size = batch_size
        self.dim = dim

    def _to_numpy(self, embeds: Any) -> np.ndarray:
        return np.array(embeds, dtype=np.float32)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            inputs = self.processor(
                text=batch,
                padding='max_length',
                max_length=64,
                return_tensors='mlx',
            )
            embeds = self.model(**inputs).text_embeds
            out.append(self._to_numpy(embeds))
        out_normalized = l2_normalize(np.vstack(out))
        if out_normalized.shape[1] != self.dim:
            raise ValueError(
                f'Expected text embedding dimension {self.dim}, got {out_normalized.shape[1]}'
            )
        return out_normalized

    def embed_images(self, images: list[Image.Image]) -> np.ndarray:
        raise NotImplementedError('MLXTextEmbedder does not support images')
