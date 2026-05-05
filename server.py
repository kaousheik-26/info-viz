#!/usr/bin/env python3
"""
Attention Visualization Server — Multi-example gallery
Scans outputs/ for all captured runs, deduplicates videos across models,
and serves a unified gallery + analysis interface.

Usage:
    python server.py --data outputs --port 8888
"""

from flask import Flask, render_template, jsonify, send_file, request
import numpy as np
import json
from pathlib import Path
import argparse

app = Flask(__name__)

ROOT_DIR = None
_cache = {}


def _load_example(model, video):
    key = f"{model}/{video}"
    if key in _cache:
        return _cache[key]

    data_dir = ROOT_DIR / model / video
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        return None

    with open(meta_path) as f:
        metadata = json.load(f)

    attn_avg = np.load(data_dir / "attention_avg.npy")

    attn_full = None
    full_path = data_dir / "attention_full.npy"
    if full_path.exists():
        attn_full = np.load(full_path)

    compression_mask = None
    cumsum_mask = None
    mask_path = data_dir / "compression_mask.npy"
    if mask_path.exists():
        compression_mask = np.load(mask_path)
        cumsum_mask = np.cumsum(compression_mask).astype(np.int64) - 1

    entry = {
        "data_dir": data_dir,
        "metadata": metadata,
        "attn_avg": attn_avg,
        "attn_full": attn_full,
        "compression_mask": compression_mask,
        "cumsum_mask": cumsum_mask,
    }
    _cache[key] = entry
    print(f"  Loaded: {key}  attn_avg={attn_avg.shape}")
    return entry


def _to_heatmaps(data_1d, meta, compression_mask):
    tg = meta["num_temporal_groups"]
    hp = meta["h_patches"]
    wp = meta["w_patches"]
    full_len = tg * hp * wp
    if compression_mask is not None:
        full = np.zeros(full_len, dtype=np.float32)
        full[compression_mask] = data_1d.astype(np.float32)
    else:
        full = data_1d.astype(np.float32)
    return full.reshape(tg, hp, wp)


def _grid_to_idx(tg, row, col, meta, compression_mask, cumsum_mask):
    wp = meta["w_patches"]
    spf = meta["spatial_per_frame"]
    flat = tg * spf + row * wp + col
    if compression_mask is not None:
        if flat >= len(compression_mask) or not compression_mask[flat]:
            return None
        return int(cumsum_mask[flat])
    else:
        if flat >= meta["num_video_tokens"]:
            return None
        return flat


def _normalise(arr):
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


@app.route("/api/examples")
def api_examples():
    """Return videos grouped by video_key, listing available models for each."""
    videos = {}  # video_key -> { prompt, models: [...], thumb_model }
    for model_dir in sorted(ROOT_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for video_dir in sorted(model_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            meta_path = video_dir / "metadata.json"
            if not meta_path.exists():
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            vk = video_dir.name
            if vk not in videos:
                videos[vk] = {
                    "video_key": vk,
                    "prompt": meta.get("prompt", ""),
                    "num_frames": meta.get("num_frames", 0),
                    "models": [],
                    "thumb_model": model_dir.name,
                }

            has_thumb = (video_dir / "frames" / "frame_000.png").exists()
            if has_thumb and not videos[vk].get("_has_thumb"):
                videos[vk]["thumb_model"] = model_dir.name
                videos[vk]["_has_thumb"] = True

            videos[vk]["models"].append({
                "model_key": model_dir.name,
                "model": meta.get("model", ""),
                "model_family": meta.get("model_family", ""),
                "num_gen_steps": meta.get("num_gen_steps", 0),
                "num_layers": meta.get("num_layers", 0),
                "num_heads": meta.get("num_heads", 0),
            })

    result = []
    for v in videos.values():
        v.pop("_has_thumb", None)
        result.append(v)
    return jsonify(result)


@app.route("/api/example/<model>/<video>/metadata")
def api_metadata(model, video):
    ex = _load_example(model, video)
    if not ex:
        return jsonify({"error": "Not found"}), 404
    return jsonify(ex["metadata"])


@app.route("/api/example/<model>/<video>/frames/<int:idx>")
def api_frame(model, video, idx):
    path = ROOT_DIR / model / video / "frames" / f"frame_{idx:03d}.png"
    if not path.exists():
        return jsonify({"error": f"Frame {idx} not found"}), 404
    return send_file(path, mimetype="image/png")


@app.route("/api/example/<model>/<video>/attention_multi")
def api_attention_multi(model, video):
    ex = _load_example(model, video)
    if not ex:
        return jsonify({"error": "Not found"}), 404

    meta = ex["metadata"]
    attn_avg = ex["attn_avg"]
    attn_full = ex["attn_full"]

    steps_str = request.args.get("steps", "0")
    layer = int(request.args.get("layer", meta["num_layers"] - 1))
    head = request.args.get("head", "avg")
    agg = request.args.get("agg", "mean")
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
    heatmaps = _to_heatmaps(data, meta, ex["compression_mask"])
    norm, vmin, vmax = _normalise(heatmaps)

    return jsonify({
        "steps": steps, "layer": layer,
        "head": head if head == "avg" else int(head),
        "agg": agg, "vmin": vmin, "vmax": vmax,
        "heatmaps": norm.tolist(), "heatmaps_raw": heatmaps.tolist(),
    })


@app.route("/api/example/<model>/<video>/attention_backward", methods=["POST"])
def api_attention_backward(model, video):
    ex = _load_example(model, video)
    if not ex:
        return jsonify({"error": "Not found"}), 404

    meta = ex["metadata"]
    attn_avg = ex["attn_avg"]
    attn_full = ex["attn_full"]

    body = request.get_json()
    patches = body.get("patches", [])
    layer = int(body.get("layer", meta["num_layers"] - 1))
    head = body.get("head", "avg")
    agg = body.get("agg", "sum")
    layer = max(0, min(layer, attn_avg.shape[1] - 1))

    video_token_indices = []
    skipped = 0
    for p in patches:
        tg, row, col = int(p["tg"]), int(p["row"]), int(p["col"])
        idx = _grid_to_idx(tg, row, col, meta, ex["compression_mask"], ex["cumsum_mask"])
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
    norm, vmin, vmax = _normalise(scores)

    return jsonify({
        "scores": scores.tolist(), "scores_norm": norm.tolist(),
        "num_patches": len(video_token_indices), "num_skipped": skipped,
        "layer": layer, "head": head, "agg": agg, "vmin": vmin, "vmax": vmax,
    })


def main():
    global ROOT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    ROOT_DIR = Path(args.data)
    if not ROOT_DIR.exists():
        print(f"Error: {ROOT_DIR} does not exist."); return

    count = 0
    for m in ROOT_DIR.iterdir():
        if not m.is_dir(): continue
        for v in m.iterdir():
            if (v / "metadata.json").exists(): count += 1
    print(f"Found {count} examples in {ROOT_DIR}/")
    print(f"\n  http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()