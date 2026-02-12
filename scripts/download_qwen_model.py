#!/usr/bin/env python3
"""Download a Qwen base model snapshot for local serving.

Usage:
  python scripts/download_qwen_model.py \
    --model Qwen/Qwen1.5-7B-Chat \
    --output-dir ./data/pretrained_models/Qwen1.5-7B-Chat

Notes:
- This script downloads model weights only.
- It does NOT start an inference server.
- qwen_ptuning.py expects a completion API compatible endpoint.
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Download Qwen model snapshot from Hugging Face.")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen1.5-7B-Chat",
        help="Model repo id on Hugging Face (default: Qwen/Qwen1.5-7B-Chat).",
    )
    parser.add_argument(
        "--output-dir",
        default="./data/pretrained_models/Qwen1.5-7B-Chat",
        help="Local output directory.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch/tag/commit revision.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token (or set HF_TOKEN env var).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Missing dependency: huggingface_hub. Install with: pip install huggingface_hub", file=sys.stderr)
        return 2

    token = args.token or os.getenv("HF_TOKEN")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Downloading {args.model} to {args.output_dir} ...")
    local_path = snapshot_download(
        repo_id=args.model,
        local_dir=args.output_dir,
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=token,
    )
    print(f"Download completed: {local_path}")
    print("Next step: start a completion-compatible serving endpoint and set QWEN_API_URL if needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
