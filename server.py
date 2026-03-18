#!/usr/bin/env python3
"""
Attention Visualization Server
Serves the interactive UI and attention data captured by capture.py.

Handles both model families:
  - Qwen2.5-VL:  regular grid, reshape directly
  - VideoLLaMA3:  possibly compressed grid, scatter back to full grid first

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
DATA_DIR        = None
metadata        = None
attn_avg        = None      # [steps, layers, video_tokens]
attn_full       = None      # [steps, layers, heads, video_tokens]  (optional)
compression_mask = None     # [full_video_tokens] bool  (optional, VideoLLaMA3 only)


def to_heatmaps(data_1d):
    """
    Convert a 1-D attention vector into per-temporal-group spatial heatmaps.

    For Qwen2.5-VL (no compression):
        data_1d has length = num_temporal_groups * h * w  → reshape directly.

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
        # Sparse → full grid
        full = np.zeros(full_len, dtype=np.float32)
        # compression_mask is bool, same length as full_len
        assert compression_mask.shape[0] == full_len, (
            f"Mask length {compression_mask.shape[0]} != grid {full_len}"
        )
        assert data_1d.shape[0] == int(compression_mask.sum()), (
            f"Data length {data_1d.shape[0]} != mask-True count "
            f"{int(compression_mask.sum())}"
        )
        full[compression_mask] = data_1d.astype(np.float32)
    else:
        # Regular grid — direct reshape
        assert data_1d.shape[0] == full_len, (
            f"Data length {data_1d.shape[0]} != grid {full_len}"
        )
        full = data_1d.astype(np.float32)

    return full.reshape(tg, hp, wp)


def normalise(heatmaps):
    """Min-max normalise to [0, 1]."""
    vmin = float(heatmaps.min())
    vmax = float(heatmaps.max())
    if vmax > vmin:
        norm = ((heatmaps - vmin) / (vmax - vmin)).tolist()
    else:
        norm = np.zeros_like(heatmaps).tolist()
    return norm, vmin, vmax


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


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
    """
    GET /api/attention?step=0&layer=27&head=avg
    Returns heatmaps shaped [num_temporal_groups][h_patches][w_patches].
    """
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
        "heatmaps": norm, "heatmaps_raw": heatmaps.tolist(),
    })


@app.route("/api/attention_multi")
def api_attention_multi():
    """
    GET /api/attention_multi?steps=0,1,2&layer=27&head=avg&agg=mean
    Aggregate attention across multiple generation steps.
    """
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
        "heatmaps": norm, "heatmaps_raw": heatmaps.tolist(),
    })


# ── Startup ──────────────────────────────────────────────────────────────────

def main():
    global DATA_DIR, metadata, attn_avg, attn_full, compression_mask

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

    print("Loading data …")
    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    attn_avg = np.load(DATA_DIR / "attention_avg.npy")
    print(f"  attention_avg:  {attn_avg.shape}  ({attn_avg.nbytes/1e6:.1f} MB)")

    full_path = DATA_DIR / "attention_full.npy"
    if full_path.exists():
        attn_full = np.load(full_path)
        print(f"  attention_full: {attn_full.shape}  ({attn_full.nbytes/1e6:.1f} MB)")
    else:
        print("  (per-head data not saved — use --save-full in capture.py)")

    mask_path = DATA_DIR / "compression_mask.npy"
    if mask_path.exists():
        compression_mask = np.load(mask_path)
        print(f"  compression_mask: {compression_mask.shape}  "
              f"({int(compression_mask.sum())}/{compression_mask.shape[0]} kept)")
    else:
        compression_mask = None

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
            f"full grid ({full_grid} = {tg}×{hp}×{wp})"
        )
        print(f"  Mode: REGULAR grid  ({n_vt_data} tokens)")

    fam = metadata.get("model_family", "unknown")
    print(f"\n  Family:    {fam}")
    print(f"  Model:     {metadata['model']}")
    print(f"  Prompt:    {metadata['prompt']}")
    print(f"  Generated: {metadata['generated_text'][:80]}…")
    print(f"  Frames:    {metadata['num_frames']}  |  "
          f"Grid: {hp}×{wp}  |  TG: {tg}  |  "
          f"Steps: {metadata['num_gen_steps']}")

    print(f"\n🌐  http://{args.host}:{args.port}")
    print(f"    SSH: ssh -L {args.port}:localhost:{args.port} user@host")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()