#!/usr/bin/env python3
"""
Attention Visualization Server
Serves the interactive UI and attention data captured by capture.py.

Handles both model families:
  - Qwen2.5-VL:  regular grid, reshape directly
  - VideoLLaMA3:  possibly compressed grid, scatter back to full grid first

Routes:
  /           — Forward: tokens → frame heatmaps
  /backward   — Backward: patches → token heatmaps

Usage:
    python server.py --data attn_data --port 8888
"""

from flask import Flask, render_template, jsonify, send_file, request
import numpy as np
import json
from pathlib import Path
import argparse

app = Flask(__name__)

# ── Globals (loaded at startup) ──────────────────────────────────────────────
DATA_DIR         = None
metadata         = None
attn_avg         = None      # [steps, layers, video_tokens]
attn_full        = None      # [steps, layers, heads, video_tokens]  (optional)
compression_mask = None      # [full_video_tokens] bool  (optional, VideoLLaMA3)
cumsum_mask      = None      # [full_video_tokens] int   (precomputed for backward)


def to_heatmaps(data_1d):
    """
    Convert a 1-D attention vector into per-temporal-group spatial heatmaps.

    For Qwen2.5-VL (no compression):
        data_1d has length = num_temporal_groups * h * w  -> reshape directly.

    For VideoLLaMA3 with compression:
        data_1d has length = num_surviving_tokens  (< full grid).
        We scatter the values back into the full grid using the saved
        compression mask, placing zeros where tokens were removed.
    """
    tg = metadata["num_temporal_groups"]
    hp = metadata["h_patches"]
    wp = metadata["w_patches"]
    full_len = tg * hp * wp

    if compression_mask is not None:
        full = np.zeros(full_len, dtype=np.float32)
        assert compression_mask.shape[0] == full_len
        assert data_1d.shape[0] == int(compression_mask.sum())
        full[compression_mask] = data_1d.astype(np.float32)
    else:
        assert data_1d.shape[0] == full_len, (
            f"Data length {data_1d.shape[0]} != grid {full_len}"
        )
        full = data_1d.astype(np.float32)

    return full.reshape(tg, hp, wp)


def grid_to_compressed_idx(tg, row, col):
    """
    Map a (tg, row, col) grid position to the index in the compressed
    video token array. Returns None if the patch was removed by compression.

    Without compression, this is simply: tg * spf + row * wp + col.
    """
    wp  = metadata["w_patches"]
    spf = metadata["spatial_per_frame"]
    flat_idx = tg * spf + row * wp + col

    if compression_mask is not None:
        if flat_idx >= len(compression_mask) or not compression_mask[flat_idx]:
            return None  # patch was compressed away
        return int(cumsum_mask[flat_idx])
    else:
        n_vt = metadata["num_video_tokens"]
        if flat_idx >= n_vt:
            return None
        return flat_idx


def normalise(arr):
    """Min-max normalise to [0, 1]."""
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        norm = ((arr - vmin) / (vmax - vmin))
    else:
        norm = np.zeros_like(arr, dtype=np.float32)
    return norm, vmin, vmax


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/backward")
def backward():
    return render_template("backward.html")


@app.route("/api/metadata")
def api_metadata():
    return jsonify(metadata)


@app.route("/api/frames/<int:idx>")
def api_frame(idx):
    path = DATA_DIR / "frames" / f"frame_{idx:03d}.png"
    if not path.exists():
        return jsonify({"error": f"Frame {idx} not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/attention")
def api_attention():
    step  = int(request.args.get("step", 0))
    layer = int(request.args.get("layer", metadata["num_layers"] - 1))
    head  = request.args.get("head", "avg")

    step  = max(0, min(step, attn_avg.shape[0] - 1))
    layer = max(0, min(layer, attn_avg.shape[1] - 1))

    if head == "avg" or attn_full is None:
        data = attn_avg[step, layer]
        head_label = "average"
    else:
        h = max(0, min(int(head), attn_full.shape[2] - 1))
        data = attn_full[step, layer, h].astype(np.float32)
        head_label = h

    heatmaps = to_heatmaps(data)
    norm, vmin, vmax = normalise(heatmaps)

    return jsonify({
        "step": step, "layer": layer, "head": head_label,
        "vmin": vmin, "vmax": vmax,
        "heatmaps": norm.tolist(), "heatmaps_raw": heatmaps.tolist(),
    })


@app.route("/api/attention_multi")
def api_attention_multi():
    steps_str = request.args.get("steps", "0")
    layer = int(request.args.get("layer", metadata["num_layers"] - 1))
    head  = request.args.get("head", "avg")
    agg   = request.args.get("agg", "mean")

    layer = max(0, min(layer, attn_avg.shape[1] - 1))

    steps = []
    for s in steps_str.split(","):
        s = s.strip()
        if s.isdigit():
            steps.append(max(0, min(int(s), attn_avg.shape[0] - 1)))
    if not steps:
        steps = [0]

    collected = []
    for s in steps:
        if head == "avg" or attn_full is None:
            collected.append(attn_avg[s, layer])
        else:
            h = max(0, min(int(head), attn_full.shape[2] - 1))
            collected.append(attn_full[s, layer, h].astype(np.float32))

    stacked = np.stack(collected)
    data = stacked.max(axis=0) if agg == "max" else stacked.mean(axis=0)

    heatmaps = to_heatmaps(data)
    norm, vmin, vmax = normalise(heatmaps)

    return jsonify({
        "steps": steps, "layer": layer,
        "head": head if head == "avg" else int(head),
        "agg": agg, "vmin": vmin, "vmax": vmax,
        "heatmaps": norm.tolist(), "heatmaps_raw": heatmaps.tolist(),
    })


@app.route("/api/attention_backward", methods=["POST"])
def api_attention_backward():
    """
    POST /api/attention_backward
    Body: {
      "patches": [ {"tg": 0, "row": 5, "col": 10}, ... ],
      "layer": 27,
      "head": "avg",
      "agg": "sum"
    }

    For each generation step, aggregates the attention weight to the selected
    video token positions. Returns a score per step.

    Handles compression: patches that were compressed away are silently skipped.
    """
    body = request.get_json()
    patches = body.get("patches", [])
    layer   = int(body.get("layer", metadata["num_layers"] - 1))
    head    = body.get("head", "avg")
    agg     = body.get("agg", "sum")

    num_layers = attn_avg.shape[1]
    layer = max(0, min(layer, num_layers - 1))

    # Convert (tg, row, col) -> compressed video token indices
    video_token_indices = []
    skipped = 0
    for p in patches:
        tg, row, col = int(p["tg"]), int(p["row"]), int(p["col"])
        idx = grid_to_compressed_idx(tg, row, col)
        if idx is not None:
            video_token_indices.append(idx)
        else:
            skipped += 1

    num_steps = attn_avg.shape[0]

    if not video_token_indices:
        return jsonify({
            "scores": [0.0] * num_steps,
            "scores_norm": [0.0] * num_steps,
            "num_patches": 0,
            "num_skipped": skipped,
            "layer": layer,
        })

    idx_arr = np.array(video_token_indices)

    # Extract attention for all steps at this layer to selected tokens
    if head == "avg" or attn_full is None:
        selected = attn_avg[:, layer, :][:, idx_arr]   # [steps, n_selected]
    else:
        h = max(0, min(int(head), attn_full.shape[2] - 1))
        selected = attn_full[:, layer, h, :].astype(np.float32)[:, idx_arr]

    # Aggregate across selected patches per step
    if agg == "mean":
        scores = selected.mean(axis=1)
    else:
        scores = selected.sum(axis=1)

    scores = scores.astype(float)
    norm, vmin, vmax = normalise(scores)

    return jsonify({
        "scores":       scores.tolist(),
        "scores_norm":  norm.tolist(),
        "num_patches":  len(video_token_indices),
        "num_skipped":  skipped,
        "layer":        layer,
        "head":         head,
        "agg":          agg,
        "vmin":         vmin,
        "vmax":         vmax,
    })


# ── Startup ──────────────────────────────────────────────────────────────────

def main():
    global DATA_DIR, metadata, attn_avg, attn_full, compression_mask, cumsum_mask

    parser = argparse.ArgumentParser(description="Attention Visualization Server")
    parser.add_argument("--data", default="attn_data",
                        help="Data directory produced by capture.py")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    DATA_DIR = Path(args.data)
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} does not exist. Run capture.py first.")
        return

    print("Loading data ...")
    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    attn_avg = np.load(DATA_DIR / "attention_avg.npy")
    print(f"  attention_avg:  {attn_avg.shape}  ({attn_avg.nbytes/1e6:.1f} MB)")

    full_path = DATA_DIR / "attention_full.npy"
    if full_path.exists():
        attn_full = np.load(full_path)
        print(f"  attention_full: {attn_full.shape}  ({attn_full.nbytes/1e6:.1f} MB)")
    else:
        print("  (per-head data not saved -- use --save-full in capture.py)")

    mask_path = DATA_DIR / "compression_mask.npy"
    if mask_path.exists():
        compression_mask = np.load(mask_path)
        # Precompute cumulative sum for fast grid->compressed index lookup
        # cumsum_mask[i] = 0-based index in compressed array for position i
        raw_cumsum = np.cumsum(compression_mask).astype(np.int64)
        cumsum_mask = raw_cumsum - 1
        print(f"  compression_mask: {compression_mask.shape}  "
              f"({int(compression_mask.sum())}/{compression_mask.shape[0]} kept)")
    else:
        compression_mask = None
        cumsum_mask = None

    # Sanity check
    n_vt_data = attn_avg.shape[2]
    tg = metadata["num_temporal_groups"]
    hp = metadata["h_patches"]
    wp = metadata["w_patches"]
    full_grid = tg * hp * wp

    if compression_mask is not None:
        expected = int(compression_mask.sum())
        assert n_vt_data == expected, (
            f"attention video-token dim ({n_vt_data}) != "
            f"compression mask True count ({expected})"
        )
        print(f"  Mode: COMPRESSED grid  ({n_vt_data}/{full_grid} tokens)")
    else:
        assert n_vt_data == full_grid, (
            f"attention video-token dim ({n_vt_data}) != "
            f"full grid ({full_grid} = {tg}x{hp}x{wp})"
        )
        print(f"  Mode: REGULAR grid  ({n_vt_data} tokens)")

    fam = metadata.get("model_family", "unknown")
    print(f"\n  Family:    {fam}")
    print(f"  Model:     {metadata['model']}")
    print(f"  Prompt:    {metadata['prompt']}")
    print(f"  Generated: {metadata['generated_text'][:80]}...")
    print(f"  Frames:    {metadata['num_frames']}  |  "
          f"Grid: {hp}x{wp}  |  TG: {tg}  |  "
          f"Steps: {metadata['num_gen_steps']}")

    print(f"\n  Starting server at http://{args.host}:{args.port}")
    print(f"    Forward view:  http://localhost:{args.port}/")
    print(f"    Backward view: http://localhost:{args.port}/backward")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()