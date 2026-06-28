import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
from PIL import Image

import objc
from AVFoundation import AVURLAsset, AVAssetImageGenerator
from CoreMedia import CMTimeMakeWithSeconds, CMTimeGetSeconds
from Foundation import NSURL
from Quartz import (
    CGImageGetDataProvider,
    CGDataProviderCopyData,
    CGImageGetWidth,
    CGImageGetHeight,
    CGImageGetBytesPerRow,
    CGImageGetBitsPerPixel,
)

logger = logging.getLogger(__name__)


@dataclass
class Frame:
    image: Image.Image
    timestamp: float
    frame_idx: int


def _cgimage_to_pil(cg_image: Any) -> Image.Image:
    """Convert an AVFoundation CGImage to a PIL Image."""
    width = CGImageGetWidth(cg_image)
    height = CGImageGetHeight(cg_image)
    bpr = CGImageGetBytesPerRow(cg_image)
    bpp = CGImageGetBitsPerPixel(cg_image)

    provider = CGImageGetDataProvider(cg_image)
    data = CGDataProviderCopyData(provider)

    # AVAssetImageGenerator on macOS produces BGRA data
    # (kCVPixelFormatType_32BGRA), so use the appropriate raw decoder.
    # frombuffer is efficient as it avoids a full copy if possible.
    if bpp == 32:
        img = Image.frombuffer(
            'RGBA', (width, height), data, 'raw', 'BGRA', bpr, 1
        )
    else:
        img = Image.frombuffer(
            'RGB', (width, height), data, 'raw', 'RGB', bpr, 1
        )
    return img.convert('RGB')


def extract_frames(path: Path, interval: float) -> list[Frame]:
    """Decode sequentially using AVFoundation for hardware acceleration."""
    out: list[Frame] = []

    url = NSURL.fileURLWithPath_(str(path))
    if not url:
        raise ValueError(f'Failed to create NSURL from path: {path}')

    asset = AVURLAsset.URLAssetWithURL_options_(url, None)
    if not asset:
        raise ValueError(f'Failed to load asset from URL: {url}')

    generator = AVAssetImageGenerator.assetImageGeneratorWithAsset_(asset)
    if not generator:
        raise ValueError(f'Failed to create image generator for asset: {url}')

    generator.setAppliesPreferredTrackTransform_(True)

    duration_cmtime = asset.duration()
    duration_sec = CMTimeGetSeconds(duration_cmtime)
    if duration_sec <= 0:
        raise ValueError(f'Video has invalid duration: {duration_sec}')

    next_t = 0.0
    idx = 0

    while next_t <= duration_sec:
        # 600 timescale is a standard safe value for video timing
        time = CMTimeMakeWithSeconds(next_t, 600)

        with objc.autorelease_pool():
            try:
                cg_image, error = (
                    generator.copyCGImageAtTime_actualTime_error_(
                        time, None, None
                    )
                )
                if cg_image:
                    pil_image = _cgimage_to_pil(cg_image)
                    out.append(
                        Frame(image=pil_image, timestamp=next_t, frame_idx=idx)
                    )
            except Exception:
                logger.exception(
                    'Failed to extract frame at %ss from %s', next_t, path
                )

        idx += 1
        next_t += interval

    return out


def dedup(frames: list[Frame], threshold: int) -> list[Frame]:
    """Drop frames whose perceptual hash is within `threshold` hamming distance
    of the last KEPT frame. Lower threshold = stricter (keeps more).

    Uses colorhash (not dhash/average_hash): gradient/mean hashes are computed
    from *relative* pixel differences and are therefore identically all-zero for
    any solid-color frame, so they cannot tell distinct flat scenes apart
    (e.g. a fully-red scene vs a fully-blue scene). colorhash is sensitive to
    color content and distinguishes them."""
    kept: list[Frame] = []
    last_hash = None
    for f in frames:
        h = imagehash.colorhash(f.image)
        if last_hash is None or (h - last_hash) > threshold:
            kept.append(f)
            last_hash = h
    return kept


def sample_video(
    path: Path, interval: float, dedup_threshold: int
) -> list[Frame]:
    """Extract and deduplicate frames from a video file."""
    return dedup(extract_frames(path, interval), dedup_threshold)
