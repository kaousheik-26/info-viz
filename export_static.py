#!/usr/bin/env python3
"""
Export captured attention data to a fully static site.
No server needed — just open index.html in a browser or host on GitHub Pages.

Usage:
    python export_static.py --data outputs --out docs
"""

import json
import shutil
import numpy as np
from pathlib import Path
import argparse


def export(data_dir, out_dir):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)

    if out_dir.exists():
        print(f"Cleaning {out_dir}/")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    manifest = []
    total_size = 0

    for model_dir in sorted(data_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for video_dir in sorted(model_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            meta_path = video_dir / "metadata.json"
            if not meta_path.exists():
                continue

            mk = model_dir.name
            vk = video_dir.name
            dest = out_dir / "data" / mk / vk
            dest.mkdir(parents=True)

            # Copy metadata
            with open(meta_path) as f:
                meta = json.load(f)
            with open(dest / "metadata.json", "w") as f:
                json.dump(meta, f)

            # Convert attention_avg.npy to raw float32 binary
            avg_path = video_dir / "attention_avg.npy"
            if avg_path.exists():
                arr = np.load(avg_path).astype(np.float32)
                bin_path = dest / "attention_avg.bin"
                arr.tofile(str(bin_path))
                size_mb = bin_path.stat().st_size / 1e6
                total_size += size_mb
                print(f"  {mk}/{vk}: {arr.shape} -> {size_mb:.1f} MB")

            # Copy compression mask if exists
            mask_path = video_dir / "compression_mask.npy"
            if mask_path.exists():
                mask = np.load(mask_path)
                mask.astype(np.uint8).tofile(str(dest / "compression_mask.bin"))

            # Copy frames
            frames_src = video_dir / "frames"
            if frames_src.exists():
                frames_dst = dest / "frames"
                shutil.copytree(frames_src, frames_dst)

            manifest.append({
                "model_key": mk,
                "video_key": vk,
                "model": meta.get("model", ""),
                "model_family": meta.get("model_family", ""),
                "prompt": meta.get("prompt", ""),
                "num_frames": meta.get("num_frames", 0),
                "num_gen_steps": meta.get("num_gen_steps", 0),
                "num_layers": meta.get("num_layers", 0),
                "num_heads": meta.get("num_heads", 0),
                "num_video_tokens": meta.get("num_video_tokens", 0),
                "has_compression": meta.get("has_compression", False),
                "attn_shape": list(np.load(avg_path).shape) if avg_path.exists() else [],
            })

    # Write manifest
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Copy static.html as index.html
    tpl = Path(__file__).parent / "templates" / "static.html"
    if tpl.exists():
        shutil.copy(tpl, out_dir / "index.html")
        print(f"\n  Copied static.html -> {out_dir}/index.html")
    else:
        print(f"\n  WARNING: {tpl} not found.")
        print(f"  Copy templates/static.html to {out_dir}/index.html manually.")

    print(f"\n  Exported {len(manifest)} examples to {out_dir}/")
    print(f"  Total attention data: {total_size:.1f} MB")
    print(f"  Open {out_dir}/index.html in Chrome")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="outputs")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()
    export(args.data, args.out)