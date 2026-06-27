from __future__ import annotations

import av
import numpy as np
import pytest
from PIL import Image


def _make_video(path, seconds=4, fps=10, size=(64, 64)):
    """Write a short 2-scene clip (red then blue) using the LGPL mpeg4 encoder
    that PyAV's binary wheels ship with."""
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = size
        stream.pix_fmt = "yuv420p"
        total = seconds * fps
        for i in range(total):
            color = (200, 0, 0) if i < total // 2 else (0, 0, 200)
            arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture
def sample_video(tmp_path):
    p = tmp_path / "clip.mp4"
    _make_video(p)
    return p


@pytest.fixture
def make_image():
    def _make(path, color=(120, 120, 120), size=(64, 64)):
        Image.new("RGB", size, color).save(path)
        return path
    return _make
