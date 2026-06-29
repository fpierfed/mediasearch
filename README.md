# mediasearch

Local semantic search over a macOS image & video library. Text→media, image→media,
and clip→clip, powered by SigLIP 2 (MLX) and LanceDB. Fully offline after the one-time
model download.

## Install (Apple Silicon, Python 3.11+)

```bash
pip install -e ".[dev]"
```

## Use

```bash
# Build / update the index (incremental + resumable)
mediasearch index ~/Movies ~/Pictures

# Search
mediasearch search "two people hiking at sunset"
mediasearch similar-image ~/Pictures/example.jpg
mediasearch similar-clip ~/Movies/example.mov

# Inspect
mediasearch status
```

Common flags: `--top-k N`, `--type image|video`, `--json`, `--open` (reveal top hit in
Finder), `--index-path PATH` (default `~/.mediasearch/index`), `--model ID`.

### Usage examples

**Build the index, then search:**

```bash
# Index your media folders (incremental — re-run to pick up new files)
$ mediasearch index ~/Pictures ~/Movies
Indexing: 100%|████████████| 342/342 [00:45<00:00, 7.5file/s]
Indexed: 334 done, 8 errors, 1204 vectors across 342 files.
```

**Text search with JSON output:**

```bash
$ mediasearch search "sunset over mountains" --json --top-k 3
[
  {
    "rank": 1,
    "score": 0.231,
    "path": "/Users/me/Pictures/sunset-yosemite.jpg",
    "media_type": "image",
    "timestamp": 0.0,
    "time": null,
    "modality": "[VISUAL]"
  },
  {
    "rank": 2,
    "score": 0.187,
    "path": "/Users/me/Movies/hike-timelapse.mp4",
    "media_type": "video",
    "timestamp": 42.0,
    "time": "0:42",
    "modality": "[VISUAL][AUDIO]"
  },
  ...
]
```

**Search only videos:**

```bash
$ mediasearch search "ocean waves" --type video
 1. 0.215  [VISUAL][AUDIO] /Users/me/Movies/beach-walk.mp4  @ 1:18
 2. 0.192  [VISUAL][AUDIO] /Users/me/Movies/surfing.mp4  @ 0:06
 3. 0.164  [VISUAL] /Users/me/Movies/coastline-drone.mp4  @ 2:30
```

**Find similar images:**

```bash
$ mediasearch similar-image ~/Pictures/reference.jpg --json --top-k 3
# Returns images (and optionally videos) visually similar to reference.jpg
```

**Find similar video clips:**

```bash
$ mediasearch similar-clip ~/Movies/sample.mov
 1. 1.000  /Users/me/Movies/sample.mov  @ 0:00
 2. 0.892  /Users/me/Movies/sample-edit.mov  @ 0:00
 3. 0.745  /Users/me/Movies/outtakes.mov  @ 1:12
```

**Filter by type in similar-image:**

```bash
$ mediasearch similar-image ~/Pictures/photo.jpg --type video
# Finds videos that contain frames visually similar to photo.jpg
```

**Inspect the index:**

```bash
$ mediasearch status
files=342  done=334  pending=0  error=8  vectors=1204

$ mediasearch status --index-path /Volumes/external/mediasearch-index
files=1240  done=1237  pending=0  error=3  vectors=5102
```

**Open the top hit in Finder:**

```bash
$ mediasearch search "red bicycle" --open
# Reveals the best-matching file in Finder
```

**Rebuild with a different model:**

```bash
$ mediasearch index ~/Pictures ~/Movies --model mlx-community/siglip2-so400m-patch16-384 --reindex
# Switches to the higher-quality so400m model and rebuilds the entire index
```

## Interpreting result scores

Each result has a `score` = cosine similarity between the query and the matched
embedding (range roughly −1 to 1; higher = more similar).

**Expect low absolute numbers, especially for text queries.** SigLIP is trained with a
sigmoid loss, so its cosine scores are much smaller than CLIP-style intuition suggests —
a genuinely strong text→image match often lands around 0.2, not 0.5+. What matters is the
*gap* between real matches and the rest, not the raw magnitude. The thresholds below are
starting points for the default `so400m` model; calibrate on your own library.

**Text → media** (lowest scale — text and image are different modalities):

| Score | Interpretation |
|---|---|
| `> ~0.25` | Strong, confident match |
| `~0.12 – 0.25` | Relevant, worth showing |
| `~0.05 – 0.12` | Weak / uncertain |
| `< ~0.05` | Likely noise |

**Image → media and clip → clip** (same modality — much higher scale):

| Score | Interpretation |
|---|---|
| `> 0.9` | Near-duplicate / same scene (an exact self-match scores ~1.0) |
| `0.7 – 0.9` | Strongly similar |
| `0.5 – 0.7` | Loosely related |
| `< 0.5` | Probably unrelated |

Notes:

- **Calibrate empirically.** Run a few queries you know the answers to and look for the
  score *drop-off* — there's usually a visible gap between real matches and noise. Set any
  cutoff at that gap rather than at a fixed global number.
- **Prefer relative cutoffs.** Because per-query magnitudes drift, "keep results within
  ~0.05 of the top hit" is more stable than a fixed absolute threshold.
- **Recalibrate when switching models.** The `base` (768-d) and `so400m` (1152-d) models
  produce different score scales, so a threshold tuned for one won't transfer to the other.
- For a true 0–1 confidence (rather than ranking), SigLIP's calibrated probability is
  `sigmoid(logit_scale · cosine + logit_bias)` using the model's learned parameters; raw
  cosine is sufficient for ranking.

## Choosing a model

Two SigLIP 2 models are supported, with different embedding dimensions:

| Model | Dim | Quality | Speed / memory |
|---|---|---|---|
| `google/siglip2-base-patch16-256` (default) | 768 | Good for everyday queries | Fast, low memory, smaller index |
| `mlx-community/siglip2-so400m-patch16-384` | 1152 | Best — stronger on fine-grained / abstract queries | ~1 GB params, slower ingest |

Pick one with `--model`. **The model and its embedding dimension are paired**, so an
index built with one model cannot be searched or extended with the other — switching
models requires rebuilding the index:

```bash
# Rebuild the whole index with the higher-quality so400m model
mediasearch index ~/Pictures ~/Movies --model mlx-community/siglip2-so400m-patch16-384 --reindex

# Then always pass the same --model for searches against that index
mediasearch search "a red bicycle" --model mlx-community/siglip2-so400m-patch16-384
```

If you query an index with the wrong model, mediasearch detects the dimension mismatch and
tells you to `--reindex` rather than failing cryptically. To make a model the permanent
default, change `DEFAULT_MODEL` in `mediasearch/config.py`.

## Tuning (for slower machines / larger libraries)

Defaults target a 16 GB Apple Silicon Mac: frame interval 2.0s, dedup threshold 5,
batch size 16, model `siglip2-base-patch16-256`. If ingest is too slow, raise
`--frame-interval` or switch models with `--model` as shown above.

Video frames are deduplicated with a perceptual color hash before embedding, so static
clips cost far fewer vectors than their raw frame count.

## Memory tuning

The default visual model is `google/siglip2-base-patch16-256`, so still images and
video frames are decoded down to a 256 px longer edge before embedding (the decoder
scales in hardware; the model would downsample anyway). This keeps batches bounded
even for 4K and 8K source media. The caps default to the selected model's input size
and can be overridden with `--image-max-size` / `--frame-max-size`.

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

Use `--no-audio` for visual-only indexing. Audio transcript indexing loads a Whisper
model and a text-embedding model; keeping audio disabled is the largest single memory
reduction for video-heavy libraries.

## Development

```bash
pytest -q                      # full unit/integration suite — no model required
python scripts/smoke_embed.py  # manual real-model check (downloads weights on first run)
```
