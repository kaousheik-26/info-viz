#!/bin/bash
# Run attention capture on all videos listed in prompts.csv for both models

QWEN="Qwen/Qwen2.5-VL-7B-Instruct"
VLLAMA="DAMO-NLP-SG/VideoLLaMA3-7B"
CSV="prompts.csv"

# Skip header, split only on first comma (in case prompt has commas)
tail -n +2 "$CSV" | while IFS= read -r line; do
    video="${line%%,*}"
    prompt="${line#*,}"

    [ -f "$video" ] || { echo "SKIP: $video not found"; continue; }
    name=$(basename "$video")

    echo "════════════════════════════════════════"
    echo "  Video:  $name"
    echo "  Prompt: $prompt"
    echo "════════════════════════════════════════"

    echo ">> Qwen2.5-VL-7B"
    python3 capture.py --model "$QWEN" --video "$video" --prompt "$prompt"

    echo ""
    echo ">> VideoLLaMA3-7B"
    python3 capture.py --model "$VLLAMA" --video "$video" --prompt "$prompt"

    echo ""
done

echo "Done. Outputs in outputs/"