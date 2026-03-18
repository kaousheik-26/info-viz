#!/usr/bin/env python3
"""
Unified VLM Attention Capture
Runs inference, captures per-layer attention from generated tokens to video patches,
and exports everything for the visualization server.

Supports:
  - Qwen2.5-VL  (e.g. Qwen/Qwen2.5-VL-7B-Instruct, Qwen/Qwen2.5-VL-32B-Instruct)
  - VideoLLaMA3  (e.g. DAMO-NLP-SG/VideoLLaMA3-7B, DAMO-NLP-SG/VideoLLaMA3-2B)

Usage:
    python capture.py --model Qwen/Qwen2.5-VL-7B-Instruct --video video.mp4
    python capture.py --model DAMO-NLP-SG/VideoLLaMA3-7B --video video.mp4
    python capture.py --model DAMO-NLP-SG/VideoLLaMA3-2B --video video.mp4 --save-full
"""

import torch
import json
import numpy as np
import os
import argparse
from pathlib import Path
from PIL import Image
from transformers import AutoConfig


# ═══════════════════════════════════════════════════════════════════════════════
# AttentionCapture — shared by both model families
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionCapture:
    """
    Registers forward hooks on every self-attention layer to capture
    decode-step attention weights (generated token -> video patch tokens).

    Only captures when q_len == 1 (autoregressive decode), skipping prefill.
    """

    def __init__(self, video_token_indices):
        self.video_indices = video_token_indices   # list of ints
        self.captures = []          # (layer_idx, np.array[heads, num_video_tokens])
        self.hooks = []
        self.num_layers = 0

    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                attn = output[1]                    # [batch, heads, q_len, k_len]
                if attn.shape[2] == 1:              # decode step only
                    video_attn = attn[0, :, 0, self.video_indices] \
                        .detach().cpu().float().numpy()
                    self.captures.append((layer_idx, video_attn))
        return hook

    def register(self, layers):
        """Attach hooks to an iterable of transformer decoder layers."""
        self.num_layers = len(layers)
        for i, layer in enumerate(layers):
            h = layer.self_attn.register_forward_hook(self._make_hook(i))
            self.hooks.append(h)
        print(f"  Registered attention hooks on {self.num_layers} layers")

    def get_data(self):
        """
        Returns
        -------
        attn_avg  : np.ndarray [steps, layers, video_tokens]  (head-averaged)
        attn_full : np.ndarray [steps, layers, heads, video_tokens]
        """
        n = len(self.captures)
        num_steps = n // self.num_layers
        assert n == num_steps * self.num_layers, (
            f"Capture count {n} not a multiple of {self.num_layers} layers"
        )
        print(f"  Captured {num_steps} decode steps x {self.num_layers} layers")

        steps_avg, steps_full = [], []
        for s in range(num_steps):
            layer_avg, layer_full = [], []
            for l in range(self.num_layers):
                idx = s * self.num_layers + l
                layer_idx, attn = self.captures[idx]
                assert layer_idx == l
                layer_full.append(attn)              # [heads, vt]
                layer_avg.append(attn.mean(axis=0))  # [vt]
            steps_full.append(np.stack(layer_full))
            steps_avg.append(np.stack(layer_avg))

        return np.stack(steps_avg), np.stack(steps_full)

    def cleanup(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()
        self.captures.clear()
        print("  Removed attention hooks")


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def save_frames_from_tensors(video_frames, frames_dir):
    """Save a list of frame tensors (CHW, 0-1 or 0-255) as PNGs."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(video_frames):
        if isinstance(frame, torch.Tensor):
            arr = frame.permute(1, 2, 0).numpy() if frame.dim() == 3 else frame.numpy()
        else:
            arr = np.array(frame)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        Image.fromarray(arr).save(frames_dir / f"frame_{i:03d}.png")
    print(f"  Saved {len(video_frames)} frames to {frames_dir}")


def save_frames_from_video(video_path, fps, max_frames, frames_dir):
    """Extract frames from a video file. Returns number of frames saved."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        total = len(vr)
        video_fps = vr.get_avg_fps()
        step = max(1, int(video_fps / fps))
        indices = list(range(0, total, step))[:max_frames]
        batch = vr.get_batch(indices).asnumpy()
        for i, frame in enumerate(batch):
            Image.fromarray(frame).save(frames_dir / f"frame_{i:03d}.png")
        print(f"  Saved {len(batch)} frames (decord)")
        return len(batch)
    except ImportError:
        pass
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        vfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(vfps / fps))
        idx, saved = 0, 0
        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(
                    frames_dir / f"frame_{saved:03d}.png"
                )
                saved += 1
            idx += 1
        cap.release()
        print(f"  Saved {saved} frames (opencv)")
        return saved
    except ImportError:
        pass
    print("  WARNING: install decord or opencv-python to save frames")
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# Model family detection
# ═══════════════════════════════════════════════════════════════════════════════

def detect_model_family(model_name_or_path: str) -> str:
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    mt = getattr(config, "model_type", "").lower()
    ar = " ".join(getattr(config, "architectures", []) or []).lower()
    if any(k in mt for k in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl")):
        return "qwen2_5_vl"
    if any(k in mt for k in ("videollama3", "video_llama_3")):
        return "videollama3"
    if any(k in ar for k in ("qwen2_5_vl", "qwen2vl")):
        return "qwen2_5_vl"
    if any(k in ar for k in ("videollama3", "video_llama_3")):
        return "videollama3"
    raise ValueError(f"Unsupported model: type={mt}, arch={ar}")


# ═══════════════════════════════════════════════════════════════════════════════
# Qwen2.5-VL backend
# ═══════════════════════════════════════════════════════════════════════════════

def run_qwen2_5_vl(args, output_dir):
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print("Loading Qwen2.5-VL …")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto",
        attn_implementation="eager", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)
    num_heads  = model.config.text_config.num_attention_heads
    num_layers = model.config.text_config.num_hidden_layers
    print(f"  {num_layers} layers, {num_heads} heads")

    # ── Inputs ──
    print("Preparing inputs …")
    messages = [{
        "role": "user",
        "content": [
            {"type": "video", "video": args.video, "fps": args.fps},
            {"type": "text",  "text":  args.prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)
    input_ids = inputs.input_ids[0]

    # ── Video token positions ──
    vid_pad_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vid_mask   = (input_ids == vid_pad_id)
    vid_idx    = torch.where(vid_mask)[0]
    n_vt       = int(vid_mask.sum())

    # ── Grid ──
    n_frames   = len(video_inputs[0])
    fh, fw     = video_inputs[0][0].shape[1], video_inputs[0][0].shape[2]
    hp, wp     = fh // 28, fw // 28
    spf        = hp * wp
    tmf        = 2
    n_tg       = n_frames // tmf

    assert n_vt == n_tg * spf, f"{n_vt} != {n_tg}*{spf}"
    print(f"  Frames={n_frames}  Grid={hp}x{wp}  TG={n_tg}  VT={n_vt}")

    save_frames_from_tensors(video_inputs[0], output_dir / "frames")

    # ── Generate ──
    capturer = AttentionCapture(vid_idx.tolist())
    capturer.register(model.model.language_model.layers)

    print("Running inference …")
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=args.max_tokens, output_attentions=True,
        )

    gen_ids  = gen_out[0, input_ids.shape[0]:]
    gen_toks = [processor.tokenizer.decode([t.item()]) for t in gen_ids]
    gen_text = processor.batch_decode(
        [gen_ids], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    meta = dict(
        model=args.model, model_family="qwen2_5_vl",
        video_path=args.video, prompt=args.prompt, fps=args.fps,
        num_frames=n_frames, frame_height=fh, frame_width=fw,
        h_patches=hp, w_patches=wp, spatial_per_frame=spf,
        num_temporal_groups=n_tg, temporal_merge_factor=tmf,
        num_video_tokens=n_vt,
        has_compression=False,
        num_layers=num_layers, num_heads=num_heads,
    )
    return capturer, gen_toks, gen_text, meta


# ═══════════════════════════════════════════════════════════════════════════════
# VideoLLaMA3 backend
# ═══════════════════════════════════════════════════════════════════════════════

def run_videollama3(args, output_dir):
    from transformers import AutoModelForCausalLM, AutoProcessor

    print("Loading VideoLLaMA3 …")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype="auto", attn_implementation="eager", device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

    cfg        = model.config
    num_heads  = cfg.num_attention_heads
    num_layers = cfg.num_hidden_layers
    img_tok    = cfg.image_token_index
    use_comp   = getattr(cfg, "use_token_compression", False)

    print(f"  {num_layers} layers, {num_heads} heads")
    print(f"  image_token_index={img_tok}  compression={use_comp}")

    # ── Inputs ──
    print("Preparing inputs …")
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {
                    "video_path": args.video,
                    "fps": args.fps,
                    "max_frames": 128,
                }},
                {"type": "text", "text": args.prompt},
            ],
        },
    ]
    inputs = processor(conversation=conversation, return_tensors="pt")

    # CPU copies for analysis
    ids_cpu = inputs["input_ids"][0].clone()
    gs_cpu  = inputs["grid_sizes"].clone()  if "grid_sizes"  in inputs else None
    ms_cpu  = inputs["merge_sizes"].clone() if "merge_sizes" in inputs else None
    modals  = inputs.get("modals", ["video"])

    # Move to GPU
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}
    if "pixel_values" in inputs and inputs["pixel_values"] is not None:
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    # ── Grid geometry ──
    if gs_cpu is not None and gs_cpu.numel() >= 3:
        gs = gs_cpu[0] if gs_cpu.dim() > 1 else gs_cpu
        t_grid, h_grid, w_grid = gs.tolist()
    else:
        t_grid, h_grid, w_grid = 1, 1, 1

    ms_val = int(ms_cpu[0].item()) if ms_cpu is not None and ms_cpu.numel() > 0 else 1
    vcfg   = getattr(cfg, "vision_encoder_config", None)
    psz    = getattr(vcfg, "patch_size", 14) if vcfg else 14

    hp  = h_grid // ms_val
    wp  = w_grid // ms_val
    spf = hp * wp
    n_tg = t_grid
    full_vt = n_tg * spf

    print(f"  Grid (T,H,W)=({t_grid},{h_grid},{w_grid})  merge={ms_val}")
    print(f"  Patches {hp}x{wp}={spf}/TG   Full VT={full_vt}")

    # ── Compression mask & video-token KV positions ──
    comp_mask_np = None

    if use_comp and "pixel_values" in inputs and gs_cpu is not None:
        print("  Computing compression mask …")
        bnp = gs_cpu.prod(dim=1).div(ms_cpu ** 2).long()
        comp_mask = model._get_compression_mask(
            inputs["pixel_values"],
            bnp.to(model.device),
            inputs["grid_sizes"],
            inputs["merge_sizes"],
            modals,
        ).cpu()                                   # [full_vt] bool

        comp_mask_np = comp_mask.numpy()
        n_surv = int(comp_mask.sum())
        print(f"  Compression: {full_vt} -> {n_surv}  "
              f"({full_vt - n_surv} removed)")

        # Reconstruct compressed input_ids to find KV positions
        img_sel = (ids_cpu == img_tok)
        keep = ~img_sel.clone()
        sel_idx = torch.where(img_sel)[0]
        n_sel = len(sel_idx)
        usable = min(n_sel, comp_mask.shape[0])
        keep[sel_idx[:usable]] = comp_mask[:usable]
        if usable < n_sel:
            keep[sel_idx[usable:]] = False

        compressed_ids = ids_cpu[keep]
        vid_positions  = torch.where(compressed_ids == img_tok)[0].tolist()
        n_vt           = len(vid_positions)
        embed_len      = compressed_ids.shape[0]
    else:
        vid_positions = torch.where(ids_cpu == img_tok)[0].tolist()
        n_vt          = len(vid_positions)
        embed_len     = ids_cpu.shape[0]

    print(f"  KV video positions: {n_vt}  "
          f"(range {vid_positions[0]}–{vid_positions[-1]})")
    print(f"  Prefill length: {embed_len}")

    # ── Save frames ──
    n_saved = save_frames_from_video(
        args.video, args.fps, 128, output_dir / "frames"
    )
    n_display = min(n_saved, t_grid) if n_saved > 0 else t_grid

    # ── Save compression mask ──
    if comp_mask_np is not None:
        np.save(output_dir / "compression_mask.npy", comp_mask_np)
        print(f"  Saved compression_mask.npy  shape={comp_mask_np.shape}")

    # ── Generate ──
    capturer = AttentionCapture(vid_positions)
    capturer.register(model.model.layers)

    print("Running inference …")
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=args.max_tokens, output_attentions=True,
        )

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # VideoLLaMA3's generate() converts input_ids → inputs_embeds internally,
    # so the returned tensor's prefix length doesn't match embed_len.
    # Use the hook capture count to reliably find the new tokens.
    n_new = len(capturer.captures) // capturer.num_layers
    gen_ids = gen_out[0, -n_new:] if n_new > 0 else gen_out[0, :0]

    gen_toks = [tokenizer.decode([t.item()]) for t in gen_ids]
    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    meta = dict(
        model=args.model, model_family="videollama3",
        video_path=args.video, prompt=args.prompt, fps=args.fps,
        num_frames=n_display,
        frame_height=h_grid * psz, frame_width=w_grid * psz,
        h_patches=hp, w_patches=wp, spatial_per_frame=spf,
        num_temporal_groups=n_tg, temporal_merge_factor=1,
        num_video_tokens=n_vt,
        num_video_tokens_full=full_vt,
        has_compression=(comp_mask_np is not None),
        patch_size=psz, merge_size=ms_val,
        grid_thw=[t_grid, h_grid, w_grid],
        num_layers=num_layers, num_heads=num_heads,
    )
    return capturer, gen_toks, gen_text, meta


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="Unified VLM Attention Capture")
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-32B-Instruct")
    ap.add_argument("--video", required=True)
    ap.add_argument("--prompt", default="Describe this video.")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--output", default="attn_data")
    ap.add_argument("--save-full", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    family = detect_model_family(args.model)
    print(f"Detected model family: {family}\n")

    if family == "qwen2_5_vl":
        capturer, gen_toks, gen_text, meta = run_qwen2_5_vl(args, out)
    elif family == "videollama3":
        capturer, gen_toks, gen_text, meta = run_videollama3(args, out)
    else:
        raise ValueError(family)

    print(f"\n  Generated: {gen_text[:120]}")

    attn_avg, attn_full = capturer.get_data()
    capturer.cleanup()

    np.save(out / "attention_avg.npy", attn_avg.astype(np.float32))
    print(f"  Saved attention_avg.npy  {attn_avg.shape}")
    if args.save_full:
        np.save(out / "attention_full.npy", attn_full.astype(np.float16))
        print(f"  Saved attention_full.npy {attn_full.shape}")

    meta.update(
        num_gen_steps=len(gen_toks),
        generated_tokens=gen_toks,
        generated_text=gen_text,
        has_full_attention=args.save_full,
    )
    with open(out / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅  All data saved to {out}/")
    print(f"    Next: python server.py --data {out}")


if __name__ == "__main__":
    main()