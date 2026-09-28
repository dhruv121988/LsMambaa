#!/usr/bin/env python3
"""
Phase 2: Efficiency Measurement Tool
====================================
Measures parameters, FLOPs (including analytical custom kernel selective-scan correction),
peak GPU memory, and throughput (images/sec) across all models under identical precision.

Usage:
    python tools/measure_efficiency.py
"""

import sys
import os
import time
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

from geoseg.models.BoundaryVMambaUNet import BoundaryVMambaUNet
from geoseg.models.CMTFNet import CMTFNet
from geoseg.models.TransUNet import TransUNet
from geoseg.models.UNetFormer import UNetFormer
from geoseg.models.SSNet import SSNet


def count_parameters(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def calculate_ss2d_flops(h: int, w: int, d_state: int = 16):
    """
    Analytically compute FLOPs performed inside the 4-way selective scan custom kernel
    which standard PyTorch autograd FLOP profilers (fvcore / thop) cannot hook into.
    
    VMamba-Tiny depth per stage: [2, 2, 9, 2], channels: [96, 192, 384, 768]
    Patch size 4 -> Stage 1 spatial: (h/4, w/4).
    Each 2D selective scan runs 4 directions (H->, <-H, V|, |V).
    Each direction: discrete recurrence h_t = A_bar * h_{t-1} + B_bar * x_t; y_t = C_bar * h_t
    Ops per token: 2 * d_state * d_channel (state update) + 2 * d_state * d_channel (output projection) + discretization ~ 9 * d_state * d_channel.
    4 directions = 36 * d_state * d_channel * L.
    """
    stages_depth = [2, 2, 9, 2]
    stages_channels = [96, 192, 384, 768]
    
    total_ss2d_flops = 0
    curr_h, curr_w = h // 4, w // 4
    for depth, d_channel in zip(stages_depth, stages_channels):
        l_tokens = curr_h * curr_w
        flops_per_block = 4 * (9 * d_state * d_channel * l_tokens)
        total_ss2d_flops += depth * flops_per_block
        curr_h = max(1, curr_h // 2)
        curr_w = max(1, curr_w // 2)
    return total_ss2d_flops


def measure_flops(model: nn.Module, input_size=(1, 3, 512, 512), is_vmamba=False):
    device = next(model.parameters()).device
    x = torch.randn(*input_size, device=device)
    
    standard_flops = 0
    if HAS_FVCORE:
        try:
            flops = FlopCountAnalysis(model, x)
            flops.unsupported_ops_warnings(False)
            standard_flops = flops.total()
        except Exception as e:
            standard_flops = 0

    custom_ss2d_flops = 0
    if is_vmamba:
        b, c, h, w = input_size
        custom_ss2d_flops = b * calculate_ss2d_flops(h, w)

    total_flops = standard_flops + custom_ss2d_flops
    return total_flops, standard_flops, custom_ss2d_flops


def measure_throughput_and_memory(model: nn.Module, batch_size: int, resolution: int, warmup: int = 5, reps: int = 15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    x = torch.randn(batch_size, 3, resolution, resolution, device=device, dtype=torch.float32)
    
    # Check if forward pass fits in memory
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    try:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
            for _ in range(warmup):
                _ = model(x)
            torch.cuda.synchronize(device)

            torch.cuda.reset_peak_memory_stats(device)
            start_time = time.perf_counter()
            for _ in range(reps):
                _ = model(x)
            torch.cuda.synchronize(device)
            end_time = time.perf_counter()

        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        total_time = end_time - start_time
        fps = (reps * batch_size) / total_time
        return fps, peak_mem_mb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return "OOM", "OOM"
    except Exception as e:
        torch.cuda.empty_cache()
        return f"ERR: {e}", "ERR"


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA device not available. Phase 2 efficiency measurement requires GPU.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print("=" * 95)
    print(f"  PHASE 2: EFFICIENCY BENCHMARK ON {device_name}")
    print("=" * 95)
    print("Precision: 16-bit Mixed (fp16) | Warmup: 5 iters | Repetitions: 15 iters | GPU Synchronized")
    print("-" * 95)

    num_classes = 7

    models = {
        "BoundaryVMambaUNet (Full)": (
            BoundaryVMambaUNet(num_classes=num_classes, decoder_channels=128, use_boundary_heads=True, gated_levels=(1, 2, 3), gate_type="multiplicative"),
            True
        ),
        "Plain VMamba U-Net": (
            BoundaryVMambaUNet(num_classes=num_classes, decoder_channels=128, use_boundary_heads=False, gated_levels=()),
            True
        ),
        "Aux-Only Baseline": (
            BoundaryVMambaUNet(num_classes=num_classes, decoder_channels=128, use_boundary_heads=True, gated_levels=()),
            True
        ),
        "Single-Stage Gate": (
            BoundaryVMambaUNet(num_classes=num_classes, decoder_channels=128, use_boundary_heads=True, gated_levels=(1,), gate_type="multiplicative"),
            True
        ),
        "CMTFNet": (
            CMTFNet(num_classes=num_classes),
            False
        ),
        "TransUNet": (
            TransUNet(num_classes=num_classes),
            False
        ),
        "UNetFormer": (
            UNetFormer(num_classes=num_classes),
            False
        ),
        "SSNet": (
            SSNet(num_classes=num_classes, patch_size=16, vit_dim=768, num_layers=8, dropout=0.1),
            False
        ),
    }

    results = []

    for idx, (name, (model, is_vmamba)) in enumerate(models.items(), 1):
        print(f"\n[{idx}/{len(models)}] Benchmarking {name}...")
        model.to("cuda")
        total_params, trainable_params = count_parameters(model)
        param_m = total_params / 1e6

        # FLOPs at 512x512
        f512_tot, f512_std, f512_ss2d = measure_flops(model, input_size=(1, 3, 512, 512), is_vmamba=is_vmamba)
        gflops_512 = f512_tot / 1e9

        # FLOPs at 1024x1024
        f1024_tot, f1024_std, f1024_ss2d = measure_flops(model, input_size=(1, 3, 1024, 1024), is_vmamba=is_vmamba)
        gflops_1024 = f1024_tot / 1e9

        # Throughput & Peak Mem at 512x512
        fps_512_b1, mem_512_b1 = measure_throughput_and_memory(model, batch_size=1, resolution=512)
        fps_512_b8, mem_512_b8 = measure_throughput_and_memory(model, batch_size=8, resolution=512)

        # Throughput & Peak Mem at 1024x1024
        fps_1024_b1, mem_1024_b1 = measure_throughput_and_memory(model, batch_size=1, resolution=1024)
        fps_1024_b8, mem_1024_b8 = measure_throughput_and_memory(model, batch_size=8, resolution=1024)

        f_b1_s = f"{fps_512_b1:.1f}" if isinstance(fps_512_b1, (int, float)) else str(fps_512_b1)
        f_b8_s = f"{fps_512_b8:.1f}" if isinstance(fps_512_b8, (int, float)) else str(fps_512_b8)
        f_1024_s = f"{fps_1024_b1:.1f}" if isinstance(fps_1024_b1, (int, float)) else str(fps_1024_b1)
        print(f"  --> Params: {param_m:.2f}M | GFLOPs: {gflops_512:.2f}G (512), {gflops_1024:.2f}G (1024) | FPS(512): {f_b1_s} (B1), {f_b8_s} (B8) | FPS(1024): {f_1024_s}")

        results.append({
            "name": name,
            "params_m": param_m,
            "gflops_512": gflops_512,
            "gflops_1024": gflops_1024,
            "fps_512_b1": fps_512_b1,
            "fps_512_b8": fps_512_b8,
            "mem_512_b8": mem_512_b8,
            "fps_1024_b1": fps_1024_b1,
            "fps_1024_b8": fps_1024_b8,
            "mem_1024_b8": mem_1024_b8,
        })
        torch.cuda.empty_cache()

    # Print Formatted Table
    print("\n" + "=" * 135)
    print("                    PHASE 2: COMPREHENSIVE EFFICIENCY BENCHMARK RESULTS                    ")
    print("=" * 135)
    print(f"{'Model Architecture':<28} | {'Params':<8} | {'GFLOPs 512':<10} | {'GFLOPs 1024':<11} | {'FPS (512, B1)':<13} | {'FPS (512, B8)':<13} | {'FPS (1024, B1)':<14} | {'FPS (1024, B8)':<14} | {'Peak Mem (512, B8)'}")
    print("-" * 135)
    for r in results:
        fps_512_b1_s = f"{r['fps_512_b1']:.1f}" if isinstance(r['fps_512_b1'], (int, float)) else str(r['fps_512_b1'])
        fps_512_b8_s = f"{r['fps_512_b8']:.1f}" if isinstance(r['fps_512_b8'], (int, float)) else str(r['fps_512_b8'])
        fps_1024_b1_s = f"{r['fps_1024_b1']:.1f}" if isinstance(r['fps_1024_b1'], (int, float)) else str(r['fps_1024_b1'])
        fps_1024_b8_s = f"{r['fps_1024_b8']:.1f}" if isinstance(r['fps_1024_b8'], (int, float)) else str(r['fps_1024_b8'])
        mem_512_b8_s = f"{r['mem_512_b8']:.1f} MB" if isinstance(r['mem_512_b8'], (int, float)) else str(r['mem_512_b8'])

        print(f"{r['name']:<28} | {r['params_m']:>6.2f}M | {r['gflops_512']:>9.2f}G | {r['gflops_1024']:>10.2f}G | {fps_512_b1_s:>13} | {fps_512_b8_s:>13} | {fps_1024_b1_s:>14} | {fps_1024_b8_s:>14} | {mem_512_b8_s}")

    print("=" * 135)
    print("Notes on FLOP counting:")
    print("  * Standard autograd profilers hook convolution, linear, and attention ops.")
    print("  * For VMamba models, custom Triton/C++ cross-scan selective recurrence ops (SS2D)")
    print("    are analytically accounted for: 4 directions x 9 x B x L x D x N per SSM block.")
    print("=" * 135 + "\n")

    # Export CSV
    csv_file = "efficiency_benchmark_results.csv"
    with open(csv_file, "w") as f:
        f.write("Model,Params(M),GFLOPs_512,GFLOPs_1024,FPS_512_B1,FPS_512_B8,FPS_1024_B1,FPS_1024_B8,PeakMem_512_B8_MB\n")
        for r in results:
            f.write(f"{r['name']},{r['params_m']:.2f},{r['gflops_512']:.2f},{r['gflops_1024']:.2f},{r['fps_512_b1']},{r['fps_512_b8']},{r['fps_1024_b1']},{r['fps_1024_b8']},{r['mem_512_b8']}\n")
    print(f"Results successfully saved to {csv_file}")


if __name__ == "__main__":
    main()
