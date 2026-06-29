# mediasearch — Installation

## Requirements

- **macOS** (Apple Silicon — M1/M2/M3/M4)
- **Python 3.11+**
- ~2 GB disk space for the default SigLIP 2 model (downloaded on first use)

## Install

```bash
git clone https://github.com/.../mediasearch.git   # or your fork
cd mediasearch
pip install -e ".[dev]"
```

The `[dev]` extra brings in `pytest` and `pyav` for running the test suite.

## Verify

```bash
# Unit / integration suite (no model required — runs in ~6 s)
pytest -q

# Real model smoke test (downloads ~1 GB on first run)
python scripts/smoke_embed.py
```

## Quick start

```bash
# Build the index (incremental — re-run to pick up new files)
mediasearch index ~/Movies ~/Pictures
```

### Usage examples

**Text search with JSON output and a type filter:**

```bash
# Search across all media
$ mediasearch search "sunset over mountains" --json --top-k 3
[
  {
    "rank": 1, "score": 0.231,
    "path": "/Users/me/Pictures/sunset-yosemite.jpg",
    "media_type": "image", "timestamp": 0.0, "time": null,
    "modality": "[VISUAL]"
  },
  ...
]

# Search only videos (audio transcripts are searched too)
$ mediasearch search "ocean waves" --type video
 1. 0.215  [VISUAL][AUDIO] /Users/me/Movies/beach-walk.mp4  @ 1:18
 2. 0.192  [VISUAL][AUDIO] /Users/me/Movies/surfing.mp4  @ 0:06
```

**Find visually similar images or clips:**

```bash
mediasearch similar-image ~/Pictures/reference.jpg --top-k 3
mediasearch similar-clip ~/Movies/sample.mov --json
mediasearch similar-image ~/Pictures/photo.jpg --type video   # only video matches
```

**Inspect and manage the index:**

```bash
$ mediasearch status
files=342  done=334  pending=0  error=8  vectors=1204

$ mediasearch status --index-path /Volumes/external/mediasearch-index

# Switch to the higher-quality so400m model (requires rebuild)
$ mediasearch index ~/Pictures ~/Movies --model mlx-community/siglip2-so400m-patch16-384 --reindex
```

**Reveal the top hit in Finder:**

```bash
mediasearch search "red bicycle" --open
```

## Model choice

Several SigLIP 2 sizes are supported.  Pick one with `--model`:

| Model | Dim | Quality | Speed / memory |
|---|---|---|---|
| `google/siglip2-base-patch16-256` *(default)* | 768 | Good | Fast, low memory |
| `mlx-community/siglip2-so400m-patch16-384` | 1152 | Best | ~1 GB params, slower |

The model and its embedding dimension are **paired** — switching models requires
`--reindex`.  mediasearch detects mismatches and tells you what to do.

## Memory tuning

Still images and video frames are decoded down to the model's input size (256 px for
the default `siglip2-base-patch16-256`) before embedding, so batches stay bounded even
for 4K/8K media. Override with `--image-max-size` / `--frame-max-size`.

For the lowest indexing memory:

```bash
mediasearch index ~/Pictures ~/Movies \
  --batch-size 1 \
  --frame-interval 5 \
  --dedup-threshold 10 \
  --image-max-size 224 \
  --frame-max-size 224 \
  --no-audio
```

`--no-audio` skips loading the Whisper and text-embedding models entirely — the largest
single memory reduction for video-heavy libraries.

## Troubleshooting

**`Could not load model ... mediasearch needs Apple Silicon + MLX`**
→ You are on an Intel Mac.  This tool requires Apple Silicon.

**`Expected text/image embedding dimension X, got Y`**
→ The on-disk index was built with a different model.  Run:

```bash
mediasearch index <dirs> --model <your-model> --reindex
```

**Search is slow**
→ Switch to the faster `base` model (see above).  It halves the index size
  and runs 3–5× faster with only a modest quality trade-off.

**`No results.` from a fresh index**
→ You may have run `search` before `index`.  Build the index first.
