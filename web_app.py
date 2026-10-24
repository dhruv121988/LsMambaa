"""
LsMamba Interactive Research Demo Studio
FastAPI service providing live boundary and semantic inference on aerial & satellite imagery.
Supports both Sentinel-2 LULC and LoveDA benchmarks with on-the-fly dataset switching.
"""
import os
import io
import time
import base64
import glob
import sys
from typing import Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tools.cfg import py2cfg
from train_supervision import Supervision_Train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[LSMamba Web App] Running on device: {device}")

# =====================================================================
# Dataset Specifications (Sentinel-2 LULC & LoveDA)
# =====================================================================
DATASETS: Dict[str, Dict[str, Any]] = {
    "sen2_lulc": {
        "id": "sen2_lulc",
        "name": "Sentinel-2 LULC (7 Classes)",
        "badge": "Sentinel-2 10m",
        "benchmark_label": "SEN-2 LULC Ground Truth",
        "ckpt_path": "model_weights/sen2_lulc/boundary_vmamba_unet-sen2_lulc-epoch16/boundary_vmamba_unet-sen2_lulc-epoch16.ckpt",
        "config_path": "config/sen2_lulc/boundary_vmamba_unet.py",
        "target_size": (128, 128),
        "classes": [
            'Water',
            'Dense Forest',
            'Sparse Forest',
            'Barren Land',
            'Built-up',
            'Agriculture Land',
            'Fallow Land'
        ],
        "class_hex": {
            'Water': '#0000ff',
            'Dense Forest': '#006400',
            'Sparse Forest': '#90ee90',
            'Barren Land': '#a52a2a',
            'Built-up': '#ff0000',
            'Agriculture Land': '#ffff00',
            'Fallow Land': '#d2b48c'
        },
        "palette": np.array([
            [0, 0, 255],       # 0: Water (Blue)
            [0, 100, 0],       # 1: Dense Forest (Dark Green)
            [144, 238, 144],   # 2: Sparse Forest (Light Green)
            [165, 42, 42],     # 3: Barren Land (Brown)
            [255, 0, 0],       # 4: Built-up (Red)
            [255, 255, 0],     # 5: Agriculture Land (Yellow)
            [210, 180, 140]    # 6: Fallow Land (Tan)
        ], dtype=np.uint8),
        "metrics": {
            "model_name": "BoundaryVMambaUNet (Proposed)",
            "backbone": "VMamba-Tiny (SS2D)",
            "decoder": "Progressive Boundary-Gated Decoder",
            "best_epoch": 16,
            "completed_epochs": 16,
            "val_mIoU": "29.16%",
            "val_F1": "41.05%",
            "val_OA": "66.31%",
            "boundary_mIoU": "12.80%",
            "boundary_F1": "31.25%",
            "baseline_val_mIoU": "28.07%",
            "improvement": "+1.09% mIoU",
            "resolution": "128×128 px (10m/px)"
        },
        "default_bnd_thresh": 0.35,
        "slider_min": 10,
        "slider_max": 70,
        "mask_offset": -1  # SEN-2 raw masks are 1-7, shifted by -1 to 0-6
    },
    "loveda": {
        "id": "loveda",
        "name": "LoveDA High-Res (7 Classes)",
        "badge": "LoveDA 0.3m",
        "benchmark_label": "LoveDA Ground Truth",
        "ckpt_path": "model_weights/loveda/boundary_vmamba_unet-multiplicative-aux0.4-bnd0.3-epoch16/last.ckpt",
        "config_path": "config/loveda/boundary_vmamba_unet.py",
        "target_size": (512, 512),
        "classes": [
            'Background / Border', 
            'Agricultural / Field', 
            'Building', 
            'Road', 
            'Water', 
            'Barren', 
            'Forest'
        ],
        "class_hex": {
            'Background / Border': '#ffffff',
            'Agricultural / Field': '#84cc16',
            'Building': '#ef4444',
            'Road': '#facc15',
            'Water': '#06b6d4',
            'Barren': '#a855f7',
            'Forest': '#15803d'
        },
        "palette": np.array([
            [255, 255, 255],   # 0: Background / Border (white)
            [132, 204, 22],    # 1: Agricultural / Field (lime/olive green)
            [239, 68, 68],     # 2: Building (vibrant red)
            [250, 204, 21],    # 3: Road (yellow)
            [6, 182, 212],     # 4: Water (cyan / river blue)
            [168, 85, 247],    # 5: Barren (purple)
            [21, 128, 61],     # 6: Forest (emerald green)
        ], dtype=np.uint8),
        "metrics": {
            "model_name": "BoundaryVMambaUNet (Proposed)",
            "backbone": "VMamba-Tiny (SS2D)",
            "decoder": "Progressive Boundary-Gated Decoder",
            "best_epoch": 15,
            "completed_epochs": 16,
            "val_mIoU": "62.19%",
            "val_F1": "75.20%",
            "val_OA": "76.93%",
            "boundary_mIoU": "10.25%",
            "boundary_F1": "29.46%",
            "baseline_val_mIoU": "59.71%",
            "improvement": "+2.48% mIoU",
            "resolution": "1024×1024 px (0.3m/px)"
        },
        "default_bnd_thresh": 0.08,
        "slider_min": 2,
        "slider_max": 25,
        "mask_offset": 0
    }
}

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =====================================================================
# Model Cache & Lazy Loader
# =====================================================================
LOADED_MODELS = {}
ACTIVE_DATASET = "sen2_lulc"  # Defaults directly to Sentinel-2 LULC!

def get_model(dataset_key: str):
    if dataset_key not in DATASETS:
        dataset_key = "sen2_lulc"
    if dataset_key not in LOADED_MODELS:
        ds = DATASETS[dataset_key]
        print(f"[LSMamba Web App] Loading {dataset_key} model: {ds['ckpt_path']}...")
        old_argv = sys.argv
        sys.argv = [sys.argv[0], "--max_train_samples", "1", "--max_val_samples", "1"]
        try:
            cfg = py2cfg(ds["config_path"])
        finally:
            sys.argv = old_argv

        m = Supervision_Train.load_from_checkpoint(ds["ckpt_path"], config=cfg)
        m = m.to(device)
        m.eval()
        LOADED_MODELS[dataset_key] = m
        print(f"[LSMamba Web App] {dataset_key} loaded successfully!")
    return LOADED_MODELS[dataset_key]


# =====================================================================
# Discover Sample Tiles for Datasets
# =====================================================================
DATASET_SAMPLES: Dict[str, list] = {"sen2_lulc": [], "loveda": []}

# Sentinel-2 Samples
sen2_img_dir = "/home/admin/Downloads/SEN-2 LULC/val_images/val"
sen2_mask_dir = "/home/admin/Downloads/SEN-2 LULC/val_masks/val"
if not os.path.exists(sen2_img_dir):
    sen2_img_dir = "data/SEN-2 LULC/val_images/val"
    sen2_mask_dir = "data/SEN-2 LULC/val_masks/val"

if os.path.exists(sen2_img_dir):
    # Diverse representative samples
    selected_ids = ["10004", "100043", "100044", "100047", "100067", "100097", "100112", "100115"]
    for sid in selected_ids:
        img_f = os.path.join(sen2_img_dir, f"{sid}.png")
        mask_f = os.path.join(sen2_mask_dir, f"{sid}.tif")
        if os.path.exists(img_f):
            DATASET_SAMPLES["sen2_lulc"].append({
                "id": sid,
                "label": f"SEN-2 #{sid}",
                "area": "Sentinel-2",
                "img_path": img_f,
                "mask_path": mask_f if os.path.exists(mask_f) else None
            })

# LoveDA Samples
for area in ["Urban", "Rural"]:
    p_img = f"data/LoveDA/Val/Val/{area}/images_png"
    p_mask = f"data/LoveDA/Val/Val/{area}/masks_png"
    if os.path.exists(p_img):
        for f in sorted(os.listdir(p_img))[:4]:
            tile_id = f.replace(".png", "")
            mask_file = os.path.join(p_mask, f)
            DATASET_SAMPLES["loveda"].append({
                "id": tile_id,
                "label": f"{area} #{tile_id}",
                "area": area,
                "img_path": os.path.join(p_img, f),
                "mask_path": mask_file if os.path.exists(mask_file) else None
            })


# =====================================================================
# Preload Default Dataset (Sentinel-2 LULC)
# =====================================================================
print("[LSMamba Web App] Preloading Sentinel-2 LULC model...")
get_model("sen2_lulc")

app = FastAPI(title="LSMamba Research Demo Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")


def colorize_mask(mask: np.ndarray, dataset_key: str = "sen2_lulc") -> Image.Image:
    ds = DATASETS.get(dataset_key, DATASETS["sen2_lulc"])
    palette = ds["palette"]
    h, w = mask.shape
    colored = np.full((h, w, 3), 255, dtype=np.uint8)
    clean_mask = mask.copy()

    if dataset_key == "loveda":
        clean_mask[clean_mask == 7] = 1  # Map 7 (agricultural ignore) to Agricultural Field
    elif dataset_key == "sen2_lulc":
        # Keep within valid 0..6
        clean_mask = np.clip(clean_mask, 0, len(palette) - 1)

    for c_idx in range(len(palette)):
        colored[clean_mask == c_idx] = palette[c_idx]
    return Image.fromarray(colored)


def make_plain_black_boundary(prob_map: np.ndarray, seg_pred: np.ndarray, thresh: Optional[float] = None, dataset_key: str = "loveda", style: str = "white") -> Image.Image:
    h, w = prob_map.shape
    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Resolve calibrated threshold per dataset
    default_t = DATASETS.get(dataset_key, {}).get("default_bnd_thresh", 0.08)
    actual_thresh = thresh if thresh is not None and thresh > 0 else default_t

    # 1. Exact 4-connected boundary between different semantic classes (pinpoint crisp parcel borders)
    padded = np.pad(seg_pred, 1, mode='edge')
    parcel_edges = (
        (seg_pred != padded[:-2, 1:-1]) |
        (seg_pred != padded[2:, 1:-1]) |
        (seg_pred != padded[1:-1, :-2]) |
        (seg_pred != padded[1:-1, 2:])
    )

    # 2. Vision Mamba boundary head prediction at calibrated threshold
    head_edges = (prob_map > actual_thresh)

    # Clean, accurate combined edges
    edges = parcel_edges | head_edges

    if style == "amber":
        black_canvas[edges] = [251, 191, 36]
    elif style == "cyan":
        black_canvas[edges] = [56, 189, 248]
    else:
        black_canvas[edges] = [255, 255, 255]

    return Image.fromarray(black_canvas)


def run_inference(
    image: Image.Image,
    dataset_key: str = "sen2_lulc",
    bnd_thresh: Optional[float] = None,
    bnd_style: str = "white",
    gt_mask_path: Optional[str] = None
):
    ds = DATASETS.get(dataset_key, DATASETS["sen2_lulc"])
    model = get_model(dataset_key)
    target_size = ds["target_size"]

    # Adapt threshold automatically to dataset operating distribution
    default_t = ds.get("default_bnd_thresh", 0.08)
    if bnd_thresh is None or bnd_thresh <= 0:
        actual_thresh = default_t
    elif dataset_key == "loveda" and bnd_thresh > 0.25:
        # Scale to LoveDA range if high threshold was passed
        actual_thresh = default_t
    elif dataset_key == "sen2_lulc" and bnd_thresh < 0.15:
        # Scale to SEN-2 range
        actual_thresh = default_t
    else:
        actual_thresh = bnd_thresh

    img_resized = image.resize(target_size, Image.BILINEAR)
    img_np = np.array(img_resized, dtype=np.float32) / 255.0

    # Normalize
    img_norm = (img_np - MEAN) / STD
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)

    t0 = time.time()
    with torch.no_grad():
        res1, res2, res3, res4 = model.net.backbone(tensor)
        out = model.net.decoder(res1, res2, res3, res4, h=target_size[1], w=target_size[0])
        seg_logits = out['seg_logits']
        bnd_logits = out['final_boundary']

        seg_pred = torch.argmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        bnd_prob = torch.sigmoid(bnd_logits).squeeze().cpu().numpy()
    latency_ms = round((time.time() - t0) * 1000, 1)

    # Segmentation mask
    pred_mask_img = colorize_mask(seg_pred, dataset_key)

    # Ground truth mask if available
    gt_b64 = None
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt_raw = np.array(Image.open(gt_mask_path).resize(target_size, Image.NEAREST))
        if ds.get("mask_offset") == -1:
            gt_raw = np.clip(gt_raw - 1, 0, len(ds["classes"]) - 1)
        gt_color_img = colorize_mask(gt_raw, dataset_key)
        gt_b64 = pil_to_base64(gt_color_img)

    # Crisp boundary on black canvas
    bnd_black_img = make_plain_black_boundary(bnd_prob, seg_pred, thresh=actual_thresh, dataset_key=dataset_key, style=bnd_style)

    # Overlay
    orig_np = np.array(img_resized)
    mask_np = np.array(pred_mask_img)
    overlay_np = cv2.addWeighted(orig_np, 0.45, mask_np, 0.55, 0)
    overlay_img = Image.fromarray(overlay_np)

    # Class percentage breakdown
    total_px = seg_pred.size
    class_stats = []
    classes = ds["classes"]
    class_hex = ds["class_hex"]
    for c_idx, c_name in enumerate(classes):
        count = int(np.sum(seg_pred == c_idx))
        pct = round((count / total_px) * 100, 2)
        class_stats.append({
            "name": c_name,
            "pct": pct,
            "hex": class_hex[c_name]
        })

    return {
        "dataset": dataset_key,
        "dataset_name": ds["name"],
        "benchmark_label": ds["benchmark_label"],
        "resolution": f"{target_size[0]}×{target_size[1]} px",
        "latency_ms": latency_ms,
        "orig_b64": pil_to_base64(img_resized),
        "gt_b64": gt_b64,
        "mask_b64": pil_to_base64(pred_mask_img),
        "bnd_b64": pil_to_base64(bnd_black_img),
        "overlay_b64": pil_to_base64(overlay_img),
        "class_stats": class_stats,
    }


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/api/datasets")
def get_datasets():
    global ACTIVE_DATASET
    res = []
    for k, v in DATASETS.items():
        res.append({
            "id": k,
            "name": v["name"],
            "badge": v["badge"],
            "active": (k == ACTIVE_DATASET)
        })
    return res


@app.post("/api/select_dataset")
def select_dataset(dataset: str = Query(...)):
    global ACTIVE_DATASET
    if dataset in DATASETS:
        ACTIVE_DATASET = dataset
        # Warm up model
        get_model(dataset)
        return {"status": "success", "active_dataset": dataset}
    return JSONResponse(status_code=400, content={"error": f"Unknown dataset {dataset}"})


@app.get("/api/samples")
def get_samples(dataset: Optional[str] = None):
    ds_key = dataset or ACTIVE_DATASET
    samples = DATASET_SAMPLES.get(ds_key, [])
    return [
        {
            "id": s["id"],
            "label": s.get("label", s["id"]),
            "area": s["area"],
            "has_gt": s["mask_path"] is not None
        }
        for s in samples
    ]


@app.get("/api/predict_sample")
def predict_sample(
    tile_id: str = Query(...),
    dataset: Optional[str] = None,
    bnd_thresh: float = Query(0.40),
    bnd_style: str = Query("white")
):
    ds_key = dataset or ACTIVE_DATASET
    samples = DATASET_SAMPLES.get(ds_key, [])
    match = next((s for s in samples if s["id"] == tile_id), None)
    if not match:
        return JSONResponse(status_code=404, content={"error": f"Tile {tile_id} not found in {ds_key}"})

    img = Image.open(match["img_path"]).convert("RGB")
    data = run_inference(
        img,
        dataset_key=ds_key,
        bnd_thresh=bnd_thresh,
        bnd_style=bnd_style,
        gt_mask_path=match["mask_path"]
    )
    data["tile_id"] = tile_id
    data["area"] = match["area"]
    return data


@app.post("/api/predict_upload")
async def predict_upload(
    file: UploadFile = File(...),
    dataset: Optional[str] = None,
    bnd_thresh: float = Query(0.40),
    bnd_style: str = Query("white")
):
    ds_key = dataset or ACTIVE_DATASET
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    data = run_inference(
        img,
        dataset_key=ds_key,
        bnd_thresh=bnd_thresh,
        bnd_style=bnd_style
    )
    data["filename"] = file.filename
    return data


@app.get("/api/metrics")
def get_metrics(dataset: Optional[str] = None):
    ds_key = dataset or ACTIVE_DATASET
    ds = DATASETS.get(ds_key, DATASETS["sen2_lulc"])
    return {
        "dataset": ds_key,
        "metrics": ds["metrics"],
        "classes": ds["classes"],
        "class_hex": ds["class_hex"],
        "benchmark_label": ds["benchmark_label"],
    }


@app.get("/", response_class=HTMLResponse)
def serve_index():
    html_file = "stitch_lsmamba_research_demo_studio/lsmamba_inference_demo/index_live.html"
    if os.path.exists(html_file):
        with open(html_file, "r") as f:
            return f.read()
    return "<h1>LSMamba Web App</h1><p>Frontend template not found.</p>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_app:app", host="0.0.0.0", port=8000, reload=False)
