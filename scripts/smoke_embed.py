"""Manual smoke test for the real MLX SigLIP 2 embedder. NOT run in CI.

Usage:  python scripts/smoke_embed.py
Downloads the model on first run, then verifies shape, normalization,
and basic cross-modal sanity (a matching caption beats a mismatched one).
"""

from PIL import Image
import numpy as np

from mediasearch.config import EMBED_DIM, DEFAULT_MODEL
from mediasearch.embedder import MLXSigLIPEmbedder

e = MLXSigLIPEmbedder(DEFAULT_MODEL)

red = Image.new('RGB', (224, 224), (220, 20, 20))
img = e.embed_images([red])
assert img.shape == (1, EMBED_DIM), img.shape
assert np.isclose(np.linalg.norm(img[0]), 1.0, atol=1e-3)

txt = e.embed_texts(['a solid red image', 'a photo of a snowy forest'])
assert txt.shape == (2, EMBED_DIM)

sim_match = float(img[0] @ txt[0])  # red image vs "red" caption
sim_mismatch = float(img[0] @ txt[1])  # red image vs "snowy forest"
print(f'match={sim_match:.3f}  mismatch={sim_mismatch:.3f}')
assert sim_match > sim_mismatch, 'cross-modal sanity check failed'
print('OK: real MLX SigLIP 2 embedder works.')
