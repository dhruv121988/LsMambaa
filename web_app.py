import os
import io
import time
import base64
import glob
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Query, Form
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tools.cfg import py2cfg
from train_supervision import Supervision_Train

# =====================================================================
# Configuration & LoveDA Specs (Accurate Class & Color Indexing)
# =====================================================================
CKPT_PATH = "model_weights/loveda/boundary_vmamba_unet-multiplicative-aux0.4-bnd0.3-epoch16/last.ckpt"
CONFIG_PATH = "config/loveda/boundary_vmamba_unet.py"

# In the model training:
# Label 1: Field / Agriculture (Ground)
# Label 2: Building
# Label 3: Road
# Label 4: Water
# Label 5: Barren
# Label 6: Forest
# Label 0: Background / Unlabeled
CLASSES = [
    'Unlabeled', 
    'Agricultural / Field', 
    'Building', 
    'Road', 
    'Water', 
    'Barren', 
    'Forest'
]

CLASS_HEX = {
    'Unlabeled': '#475569',
    'Agricultural / Field': '#eab308',
    'Building': '#ef4444',
    'Road': '#f8fafc',
    'Water': '#06b6d4',
    'Barren': '#a855f7',
    'Forest': '#22c55e'
}

PALETTE = np.array([
    [71, 85, 105],    # 0: Unlabeled (slate gray)
    [234, 179, 8],    # 1: Agricultural / Field (gold / yellow-tan)
    [239, 68, 68],    # 2: Building (vibrant red)
    [248, 250, 252],  # 3: Road (clean white / asphalt line)
    [6, 182, 212],    # 4: Water (cyan / river blue)
    [168, 85, 247],   # 5: Barren (purple)
    [34, 197, 94],    # 6: Forest (emerald green)
], dtype=np.uint8)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# =====================================================================
# Initialize Model
# =====================================================================
print("[LSMamba Web App] Loading BoundaryVMambaUNet model checkpoint...")
cfg = py2cfg(CONFIG_PATH)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Supervision_Train.load_from_checkpoint(CKPT_PATH, config=cfg)
model = model.to(device)
model.eval()
print(f"[LSMamba Web App] Model loaded successfully on {device}!")

app = FastAPI(title="LSMamba Research Demo Studio")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Find sample tiles in LoveDA dataset
SAMPLE_TILES = []
for area in ["Urban", "Rural"]:
    p_img = f"data/LoveDA/Val/Val/{area}/images_png"
    p_mask = f"data/LoveDA/Val/Val/{area}/masks_png"
    if os.path.exists(p_img):
        files = sorted(os.listdir(p_img))[:8]
        for f in files:
            tile_id = f.replace(".png", "")
            mask_file = os.path.join(p_mask, f)
            SAMPLE_TILES.append({
                "id": tile_id,
                "area": area,
                "img_path": os.path.join(p_img, f),
                "mask_path": mask_file if os.path.exists(mask_file) else None
            })

def pil_to_base64(img: Image.Image, format="PNG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")

def colorize_mask(mask: np.ndarray) -> Image.Image:
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for c_idx in range(len(PALETTE)):
        colored[mask == c_idx] = PALETTE[c_idx]
    return Image.fromarray(colored)

def make_plain_black_boundary(prob_map: np.ndarray, seg_pred: np.ndarray, thresh: float = 0.20, style: str = "white") -> Image.Image:
    """
    Produces a crisp, pure-black background (#000000) where ONLY boundaries are visible.
    Combines the model's boundary head prediction with edge contours.
    """
    h, w = prob_map.shape
    black_canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Morphological contour extraction from predicted segmentation mask
    seg_u8 = seg_pred.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(seg_u8, kernel)
    eroded = cv2.erode(seg_u8, kernel)
    mask_edges = (dilated != eroded)

    # 2. Model boundary head confidence above threshold
    head_edges = (prob_map > thresh)

    # Combined sharp edge map
    edges = mask_edges | head_edges

    if style == "amber":
        # Electric Amber on pitch black
        black_canvas[edges] = [251, 191, 36]
    elif style == "cyan":
        # Electric Cyan on pitch black
        black_canvas[edges] = [56, 189, 248]
    else:
        # Crisp Pure White on pitch black (#FFFFFF on #000000)
        black_canvas[edges] = [255, 255, 255]

    return Image.fromarray(black_canvas)

def run_inference(image: Image.Image, target_size=(512, 512), bnd_thresh: float = 0.20, bnd_style: str = "white"):
    orig_w, orig_h = image.size
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
    pred_mask_img = colorize_mask(seg_pred)

    # Crisp boundary on pure black background
    bnd_black_img = make_plain_black_boundary(bnd_prob, seg_pred, thresh=bnd_thresh, style=bnd_style)

    # Overlay (blend of original image and segmentation mask)
    orig_np = np.array(img_resized)
    mask_np = np.array(pred_mask_img)
    overlay_np = cv2.addWeighted(orig_np, 0.45, mask_np, 0.55, 0)
    overlay_img = Image.fromarray(overlay_np)

    # Class percentage breakdown
    total_px = seg_pred.size
    class_stats = []
    for c_idx, c_name in enumerate(CLASSES):
        count = int(np.sum(seg_pred == c_idx))
        pct = round((count / total_px) * 100, 2)
        if pct > 0 or c_idx in [1, 2, 3, 4, 6]:
            class_stats.append({
                "name": c_name,
                "pct": pct,
                "hex": CLASS_HEX[c_name]
            })

    return {
        "latency_ms": latency_ms,
        "orig_b64": pil_to_base64(img_resized),
        "mask_b64": pil_to_base64(pred_mask_img),
        "bnd_b64": pil_to_base64(bnd_black_img),
        "overlay_b64": pil_to_base64(overlay_img),
        "class_stats": class_stats,
    }


# =====================================================================
# API Endpoints
# =====================================================================

@app.get("/api/samples")
def get_samples():
    res = []
    for s in SAMPLE_TILES:
        res.append({
            "id": s["id"],
            "area": s["area"],
            "has_gt": s["mask_path"] is not None
        })
    return res

@app.get("/api/predict_sample")
def predict_sample(
    tile_id: str = Query(...),
    bnd_thresh: float = Query(0.20),
    bnd_style: str = Query("white")
):
    match = next((s for s in SAMPLE_TILES if s["id"] == tile_id), None)
    if not match:
        return JSONResponse(status_code=404, content={"error": "Tile not found"})
    
    img = Image.open(match["img_path"]).convert("RGB")
    data = run_inference(img, bnd_thresh=bnd_thresh, bnd_style=bnd_style)
    data["tile_id"] = tile_id
    data["area"] = match["area"]
    return data

@app.post("/api/predict_upload")
async def predict_upload(
    file: UploadFile = File(...),
    bnd_thresh: float = Query(0.20),
    bnd_style: str = Query("white")
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    data = run_inference(img, bnd_thresh=bnd_thresh, bnd_style=bnd_style)
    data["filename"] = file.filename
    return data

@app.get("/api/metrics")
def get_metrics():
    return {
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
        "improvement": "+2.48% mIoU"
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
