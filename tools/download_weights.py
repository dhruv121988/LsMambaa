#!/usr/bin/env python3
"""
Download trained model checkpoints from GitHub Releases.
Usage:
    python tools/download_weights.py [--model boundary_cnn_mamba_unet]
"""
import os
import sys
import argparse
import urllib.request
from tqdm import tqdm


CHECKPOINTS = {
    "boundary_cnn_mamba_unet": {
        "url": "https://github.com/dhruv121988/LsMambaa/releases/download/v1.0.0-weights/boundary_cnn_mamba_unet_resnet34-multiplicative-epoch16.ckpt",
        "dest": "model_weights/loveda/boundary_cnn_mamba_unet_resnet34-multiplicative-epoch16/boundary_cnn_mamba_unet_resnet34-multiplicative-epoch16.ckpt",
        "description": "BoundaryCNNMambaUNet 16-epoch LoveDA Checkpoint (58.86% mIoU)",
    },
    "boundary_cnn_mamba_unet_weights": {
        "url": "https://github.com/dhruv121988/LsMambaa/releases/download/v1.0.0-weights/boundary_cnn_mamba_unet_resnet34-epoch16_state_dict.pth",
        "dest": "model_weights/loveda/boundary_cnn_mamba_unet_resnet34-multiplicative-epoch16/boundary_cnn_mamba_unet_resnet34-epoch16_state_dict.pth",
        "description": "BoundaryCNNMambaUNet 16-epoch state_dict only (87 MB)",
    },
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest: str, desc: str):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        print(f"File already exists: {dest} (Skipping download)")
        return

    print(f"Downloading {desc}...")
    print(f"From: {url}")
    print(f"To:   {dest}")
    with DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, filename=dest, reporthook=t.update_to)
    print(f"Downloaded successfully: {dest}\n")


def main():
    parser = argparse.ArgumentParser(description="Download model weights for LsMamba")
    parser.add_argument("--model", type=str, default="boundary_cnn_mamba_unet",
                        choices=list(CHECKPOINTS.keys()) + ["all"],
                        help="Model checkpoint to download")
    args = parser.parse_args()

    if args.model == "all":
        for k, v in CHECKPOINTS.items():
            download_file(v["url"], v["dest"], v["description"])
    else:
        v = CHECKPOINTS[args.model]
        download_file(v["url"], v["dest"], v["description"])


if __name__ == "__main__":
    main()
