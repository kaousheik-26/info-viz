#!/usr/bin/env python3
"""
Attention Visualization Server
Single unified interface with coordinated forward + backward views.

Usage:
    python server.py --data attn_data --port 8888
"""

from flask import Flask, render_template, jsonify, send_file, request
import numpy as np
import json
from pathlib import Path
import argparse

app = Flask(__name__)

DATA_DIR         = None
metadata         = None
attn_avg         = None
attn_full        = None
compression_mask = None
cumsum_mask      = None


def to_heatmaps(data_1d):
    tg = metadata["num_temporal_groups"]
    hp = metadata["h_patches"]
    wp = metadata["w_patches"]
    full_len = tg * hp * wp
    if compression_mask is not None:
        full = np.zeros(full_len, dtype=np.float32)
        full[compression_mask] = data_1d.astype(np.float32)
    else:
        full = data_1d.astype(np.float32)
    return full.reshape(tg, hp, wp)


def grid_to_compressed_idx(tg, row, col):
    wp  = metadata["w_patches"]
    spf = metadata["spatial_per_frame"]
    flat_idx = tg * spf + row * wp + col
    if compression_mask is not None:
        if flat_idx >= len(compression_mask) or not compression_mask[flat_idx]:
            return None
        return int(cumsum_mask[flat_idx])
    else:
        if flat_idx >= metadata["num_video_tokens"]:
            return None
        return flat_idx


def normalise(arr):
    vmin = float(arr.min())
    vmax = float(arr.max())
    if vmax > vmin:
        norm = (arr - vmin) / (vmax - vmin)
    else:
        norm = np.zeros_like(arr, dtype=np.float32)
    return norm, vmin, vmax


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
    body = request.get_json()
    patches = body.get("patches", [])
    layer   = int(body.get("layer", metadata["num_layers"] - 1))
    head    = body.get("head", "avg")
    agg     = body.get("agg", "sum")
    layer   = max(0, min(layer, attn_avg.shape[1] - 1))

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
            "scores": [0.0] * num_steps, "scores_norm": [0.0] * num_steps,
            "num_patches": 0, "num_skipped": skipped, "layer": layer,
        })

    idx_arr = np.array(video_token_indices)
    if head == "avg" or attn_full is None:
        selected = attn_avg[:, layer, :][:, idx_arr]
    else:
        h = max(0, min(int(head), attn_full.shape[2] - 1))
        selected = attn_full[:, layer, h, :].astype(np.float32)[:, idx_arr]

    scores = selected.mean(axis=1) if agg == "mean" else selected.sum(axis=1)
    scores = scores.astype(float)
    norm, vmin, vmax = normalise(scores)

    return jsonify({
        "scores": scores.tolist(), "scores_norm": norm.tolist(),
        "num_patches": len(video_token_indices), "num_skipped": skipped,
        "layer": layer, "head": head, "agg": agg, "vmin": vmin, "vmax": vmax,
    })


def main():
    global DATA_DIR, metadata, attn_avg, attn_full, compression_mask, cumsum_mask

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="attn_data")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    DATA_DIR = Path(args.data)
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} does not exist."); return

    print("Loading data ...")
    with open(DATA_DIR / "metadata.json") as f:
        metadata = json.load(f)

    attn_avg = np.load(DATA_DIR / "attention_avg.npy")
    print(f"  attention_avg: {attn_avg.shape}")

    full_path = DATA_DIR / "attention_full.npy"
    if full_path.exists():
        attn_full = np.load(full_path)
        print(f"  attention_full: {attn_full.shape}")

    mask_path = DATA_DIR / "compression_mask.npy"
    if mask_path.exists():
        compression_mask = np.load(mask_path)
        cumsum_mask = np.cumsum(compression_mask).astype(np.int64) - 1
        print(f"  compression_mask: {int(compression_mask.sum())}/{compression_mask.shape[0]} kept")

    print(f"\n  Model:     {metadata['model']}")
    print(f"  Generated: {metadata['generated_text'][:80]}...")
    print(f"  Frames: {metadata['num_frames']}  Grid: {metadata['h_patches']}x{metadata['w_patches']}  Steps: {metadata['num_gen_steps']}")
    print(f"\n  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()