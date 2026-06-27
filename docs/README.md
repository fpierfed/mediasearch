# medsearch — Installation

## Requirements

- **macOS** (Apple Silicon — M1/M2/M3/M4)
- **Python 3.11+**
- ~2 GB disk space for the default SigLIP 2 model (downloaded on first use)

## Install

```bash
git clone https://github.com/.../medsearch.git   # or your fork
cd medsearch
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
medsearch index ~/Movies ~/Pictures
```

### Usage examples

**Text search with JSON output and a type filter:**

```bash
# Search across all media
$ medsearch search "sunset over mountains" --json --top-k 3
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
$ medsearch search "ocean waves" --type video
 1. 0.215  [VISUAL][AUDIO] /Users/me/Movies/beach-walk.mp4  @ 1:18
 2. 0.192  [VISUAL][AUDIO] /Users/me/Movies/surfing.mp4  @ 0:06
```

**Find visually similar images or clips:**

```bash
$ medsearch similar-image ~/Pictures/reference.jpg --top-k 3
$ medsearch similar-clip ~/Movies/sample.mov --json
$ medsearch similar-image ~/Pictures/photo.jpg --type video   # only video matches
```

**Inspect and manage the index:**

```bash
$ medsearch status
files=342  done=334  pending=0  error=8  vectors=1204

$ medsearch status --index-path /Volumes/external/medsearch-index

# Switch to the faster base model (requires rebuild)
$ medsearch index ~/Pictures ~/Movies --model mlx-community/siglip2-base-patch16-384 --reindex
```

**Reveal the top hit in Finder:**

```bash
$ medsearch search "red bicycle" --open
```

## Model choice

Two SigLIP 2 sizes are supported.  Pick one with `--model`:

| Model | Dim | Quality | Speed / memory |
|---|---|---|---|
| `mlx-community/siglip2-so400m-patch16-384` *(default)* | 1152 | Best | ~1 GB params |
| `mlx-community/siglip2-base-patch16-384` | 768 | Good | ~3–5× faster |

The model and its embedding dimension are **paired** — switching models requires
`--reindex`.  medsearch detects mismatches and tells you what to do.

## Troubleshooting

**`Could not load model ... medsearch needs Apple Silicon + MLX`**
→ You are on an Intel Mac.  This tool requires Apple Silicon.

**`Expected text/image embedding dimension X, got Y`**
→ The on-disk index was built with a different model.  Run:
```bash
medsearch index <dirs> --model <your-model> --reindex
```

**Search is slow**
→ Switch to the faster `base` model (see above).  It halves the index size
  and runs 3–5× faster with only a modest quality trade-off.

**`No results.` from a fresh index**
→ You may have run `search` before `index`.  Build the index first.
