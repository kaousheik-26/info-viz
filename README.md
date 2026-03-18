# VLM Attention Visualizer

Interactive tool for capturing and visualizing per-layer, per-head attention heatmaps from video language models during autoregressive generation. See exactly which video regions each generated token attends to.

![Qwen2.5-VL & VideoLLaMA3 supported](https://img.shields.io/badge/models-Qwen2.5--VL%20%7C%20VideoLLaMA3-blue)

## Supported Models

| Model | Example checkpoint | Notes |
|---|---|---|
| **Qwen2.5-VL** | `Qwen/Qwen2.5-VL-7B-Instruct`, `Qwen/Qwen2.5-VL-32B-Instruct` | Regular spatial grid, temporal merge ×2 |
| **VideoLLaMA3** | `DAMO-NLP-SG/VideoLLaMA3-7B`, `DAMO-NLP-SG/VideoLLaMA3-2B` | SigLIP encoder, optional token compression |

The model family is auto-detected from the config — just change `--model`.

## Project Structure

```
.
├── capture.py              # Step 1: Run inference, hook attention, save data
├── server.py               # Step 2: Serve interactive visualization
├── templates/
│   └── index.html          # Frontend UI (served by Flask)
├── attn_data/              # Output directory (created by capture.py)
│   ├── metadata.json       # Model info, grid geometry, generated tokens
│   ├── attention_avg.npy   # [steps, layers, video_tokens]  float32
│   ├── attention_full.npy  # [steps, layers, heads, video_tokens]  float16 (optional)
│   ├── compression_mask.npy# [full_video_tokens] bool (VideoLLaMA3 only)
│   └── frames/
│       ├── frame_000.png
│       ├── frame_001.png
│       └── ...
└── README.md
```

## How It Works

### capture.py

Runs model inference on a video+prompt and captures attention weights at every autoregressive decode step using PyTorch forward hooks on each self-attention layer.

**What it does:**

1. **Loads the model** with `attn_implementation="eager"` (required to get attention weight tensors — flash attention doesn't expose them)
2. **Prepares inputs** using each model's native processor/chat template
3. **Locates video token positions** in the KV cache — this is the tricky part:
   - *Qwen2.5-VL*: finds `<|video_pad|>` token positions in `input_ids`
   - *VideoLLaMA3*: finds `image_token_index` positions, accounting for token compression (some video tokens may be removed based on inter-frame similarity)
4. **Registers forward hooks** on every `layer.self_attn` module that capture `attn_weights[0, :, 0, video_indices]` during decode steps (when `q_len == 1`)
5. **Runs `model.generate()`** — the hooks fire on each decode step, capturing which video patches the new token attends to
6. **Saves** attention arrays, metadata, video frames, and (for VideoLLaMA3) the compression mask

### server.py

Flask server that loads the captured data and serves a REST API + the interactive frontend.

**Key detail — grid reconstruction:**

The attention arrays store one value per video token. To overlay heatmaps on frames, we need to reshape this 1-D vector into a `[temporal_groups, h_patches, w_patches]` grid.

- *Qwen2.5-VL (no compression)*: direct reshape — `num_video_tokens == TG × H × W`
- *VideoLLaMA3 (with compression)*: scatter surviving tokens back into the full grid using `compression_mask.npy`, with zeros where tokens were removed, then reshape

The API always returns `[TG][H][W]` heatmaps regardless of model, so the frontend doesn't need to know about compression.

**API endpoints:**

| Endpoint | Description |
|---|---|
| `GET /` | Serves the UI |
| `GET /api/metadata` | Model info, grid geometry, generated tokens |
| `GET /api/frames/<idx>` | Video frame PNG |
| `GET /api/attention?step=0&layer=27&head=avg` | Single-step heatmap |
| `GET /api/attention_multi?steps=0,1,2&layer=27&head=avg&agg=mean` | Multi-step aggregated heatmap |

### templates/index.html

Single-page interactive frontend. No build step, no dependencies — just vanilla JS with inline CSS.

**Features:**
- Click/shift-click/ctrl-click generated tokens to select which decode steps to visualize
- Layer slider, head selector, opacity control
- Mean/max aggregation across selected steps
- Six colormaps (inferno, hot, viridis, plasma, magma, turbo)
- Per-frame attention sum scores
- Model family badge and compression indicator

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Core dependencies
pip install torch transformers flask numpy Pillow

# For Qwen2.5-VL
pip install qwen-vl-utils

# For VideoLLaMA3 (custom code model — no extra package needed,
# but the processor requires ffmpeg for video loading)
# Install ffmpeg binary:
conda install -c conda-forge ffmpeg
# or: sudo apt install ffmpeg
# or: module load ffmpeg  (on cluster)

# For frame extraction (at least one required)
pip install decord
# or: pip install opencv-python
```

## Usage

### Step 1: Capture attention data

```bash
# Qwen2.5-VL
python capture.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --video /path/to/video.mp4 \
  --prompt "Describe this video." \
  --output attn_data_qwen

# VideoLLaMA3
python capture.py \
  --model DAMO-NLP-SG/VideoLLaMA3-7B \
  --video /path/to/video.mp4 \
  --prompt "Describe this video." \
  --output attn_data_vllama3

# With per-head data (larger files)
python capture.py \
  --model DAMO-NLP-SG/VideoLLaMA3-2B \
  --video /path/to/video.mp4 \
  --save-full \
  --output attn_data_full
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-VL-32B-Instruct` | HuggingFace model name or local path |
| `--video` | *(required)* | Path to video file |
| `--prompt` | `"Describe this video."` | Text prompt |
| `--fps` | `1.0` | Frames per second to sample |
| `--max-tokens` | `128` | Maximum tokens to generate |
| `--output` | `attn_data` | Output directory |
| `--save-full` | `false` | Also save per-head attention (can be large) |

### Step 2: Launch the visualization server

```bash
python server.py --data attn_data_qwen --port 8888
```

### Step 3: Open in browser

**Local machine:**
```
http://localhost:8888
```

**Remote server (SSH tunnel):**
```bash
# On your local machine:
ssh -L 8888:localhost:8888 user@gpu-server

# If behind a jump host:
ssh -L 8888:localhost:8888 -J user@jumphost user@gpu-server

# Then open http://localhost:8888
```

If port 8888 is already in use locally, pick a different local port:
```bash
ssh -L 9999:localhost:8888 user@gpu-server
# Then open http://localhost:9999
```

## Output Data Format

### metadata.json

```jsonc
{
  "model": "DAMO-NLP-SG/VideoLLaMA3-2B",
  "model_family": "videollama3",        // or "qwen2_5_vl"
  "num_frames": 10,
  "h_patches": 13,                       // spatial height in patches
  "w_patches": 23,                       // spatial width in patches
  "num_temporal_groups": 10,             // number of temporal groups
  "temporal_merge_factor": 1,            // 1 for VideoLLaMA3, 2 for Qwen2.5-VL
  "num_video_tokens": 2990,             // tokens in the KV cache
  "num_video_tokens_full": 2990,        // pre-compression count (VideoLLaMA3)
  "has_compression": true,              // whether compression mask exists
  "num_layers": 28,
  "num_heads": 12,
  "num_gen_steps": 70,
  "generated_tokens": ["The", " video", ...],
  "generated_text": "The video shows..."
}
```

### attention_avg.npy

Shape: `[num_gen_steps, num_layers, num_video_tokens]` — float32

Head-averaged attention weights. For each generated token (step) and each layer, stores the attention weight from that token to every video token position.

### attention_full.npy (optional, `--save-full`)

Shape: `[num_gen_steps, num_layers, num_heads, num_video_tokens]` — float16

Per-head attention weights. Can be large — for Qwen2.5-VL-32B with 127 steps × 64 layers × 40 heads × 1495 tokens, this is ~930 MB.

### compression_mask.npy (VideoLLaMA3 only)

Shape: `[num_temporal_groups × h_patches × w_patches]` — bool

Indicates which positions in the full spatial grid have surviving tokens after VideoLLaMA3's temporal similarity compression. `True` = token kept, `False` = token removed. Used by the server to scatter sparse attention values back into the full grid for visualization.

## Notes

- `attn_implementation="eager"` is required — flash attention and SDPA do not return attention weight tensors. This makes inference slower and uses more memory than normal.
- The hooks only capture decode steps (`q_len == 1`), not the prefill pass.
- VideoLLaMA3's token compression may remove zero tokens if the video has high motion (all frames are sufficiently different).
- Frame counts may differ slightly between what decord/opencv extracts and what the model's processor uses, because each has its own sampling logic. The visualization uses `min(saved_frames, temporal_groups)`.