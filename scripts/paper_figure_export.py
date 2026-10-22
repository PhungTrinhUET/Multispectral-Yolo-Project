#!/usr/bin/env python3
"""
paper_figure_export.py — Export paper-quality figure images for GMD-YOLO.

Outputs (paper_figures/):
  input_rgnir.png   — false-color CIR composite (square, native res, 300 DPI)
  input_rgnir.svg   — same image embedded as base64
  input_ndvi.png    — NDVI RdYlGn colormap (300 DPI)
  input_ndvi.svg    — same image embedded as base64
  output_pred.png   — CIR composite + rasterised detection boxes
  output_pred.svg   — CIR base (base64) + VECTOR <rect>/<text> boxes

Run from repo root:
  python scripts/paper_figure_export.py
"""

import sys
import os
import subprocess
import base64
from pathlib import Path

# ─── auto-install ─────────────────────────────────────────────────────────────
def _ensure(pkg, imp=None):
    try:
        __import__(imp or pkg)
    except ImportError:
        print(f"[pip] installing {pkg} …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("pyyaml",       "yaml")
_ensure("numpy",        "numpy")
_ensure("opencv-python","cv2")
_ensure("Pillow",       "PIL")
_ensure("matplotlib",   "matplotlib")
_ensure("ultralytics",  "ultralytics")

import yaml
import numpy as np
import cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
import torch

# ─── paths ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
YAML_PATH = REPO / "dataset_melon.yaml"
WEIGHTS   = Path(
    "/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics/"
    "Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt"
)
OUT_DIR = REPO / "paper_figures"

IMGSZ = 1280    # model inference resolution
CONF  = 0.10    # lower threshold for paper figure (objects are small)
IOU   = 0.45
DPI   = 300

# ─── label mapping (English, for paper) ───────────────────────────────────────
EN = {
    "truoc_thu_phan": "pre-pollination",
    "sau_thu_phan":   "post-pollination",
    "cay40":          "net formation",
    "tao_ngot":       "sweetening stage",
}

# per-class colors — BGR (OpenCV) and hex RGB (SVG)
COLORS_BGR = [
    ( 30,  80, 220),   # 0 → blue
    ( 30, 180,  30),   # 1 → green
    ( 30, 130, 220),   # 2 → orange
    ( 30,  30, 200),   # 3 → red
]
COLORS_HEX = ["#dc501e", "#1eb41e", "#dc821e", "#c81e1e"]  # RGB hex

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  READ YAML
# ═══════════════════════════════════════════════════════════════════════════════
if not YAML_PATH.exists():
    sys.exit(f"[ERROR] {YAML_PATH} not found")

cfg      = yaml.safe_load(YAML_PATH.read_text())
raw_root = Path(cfg["path"])
ndvi_root = Path(str(raw_root).replace("dataset_raw", "dataset_ndvi"))
nc       = int(cfg["nc"])
# names from yaml: {0: str, 1: str, ...}
names    = {int(k): str(v) for k, v in cfg["names"].items()}
names_en = {k: EN.get(v, v) for k, v in names.items()}

# choose split (prefer test)
for candidate in ["test", "val", "train"]:
    key = candidate
    img_rel = cfg.get(key, f"images/{key}")
    if (raw_root / img_rel).is_dir():
        split = candidate
        img_dir   = raw_root  / img_rel
        ndvi_dir  = ndvi_root / img_rel
        label_dir = raw_root  / "labels" / candidate
        break
else:
    sys.exit("[ERROR] No valid image split found in yaml")

print(f"[yaml]   split={split}  nc={nc}  names={names_en}")
print(f"         img:  {img_dir}")
print(f"         ndvi: {ndvi_dir}")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  PICK BEST IMAGE (max class diversity, then max box count)
# ═══════════════════════════════════════════════════════════════════════════════
best_stem, best_score = None, -1
label_dir = Path(str(label_dir))   # ensure Path

if label_dir.is_dir():
    for lf in sorted(label_dir.glob("*.txt")):
        rows = [r for r in lf.read_text().splitlines() if r.strip()]
        if not rows:
            continue
        cls_set = {int(r.split()[0]) for r in rows}
        # Class diversity is the primary criterion; box count breaks ties
        score = len(cls_set) * 1000 + len(rows)
        if score > best_score:
            best_score = score
            best_stem  = lf.stem

if best_stem is None:
    # fallback: first image in directory
    imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not imgs:
        sys.exit(f"[ERROR] No images found in {img_dir}")
    best_stem = imgs[0].stem

# resolve RGB path
rgb_path = None
for ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
    p = img_dir / f"{best_stem}{ext}"
    if p.exists():
        rgb_path = p; break
if rgb_path is None:
    sys.exit(f"[ERROR] RGB image not found for stem {best_stem} in {img_dir}")

# resolve NDVI path
ndvi_path = None
for ext in [".png", ".tif", ".tiff", ".jpg", ".jpeg"]:
    p = ndvi_dir / f"{best_stem}{ext}"
    if p.exists():
        ndvi_path = p; break

print(f"\n[image]  {best_stem}")
print(f"         rgb:  {rgb_path}")
print(f"         ndvi: {ndvi_path or '(not found — will compute from channels)'}")
if not WEIGHTS.exists():
    sys.exit(f"[ERROR] Weights not found: {WEIGHTS}")
print(f"         wts:  {WEIGHTS}")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  LOAD IMAGES
# ═══════════════════════════════════════════════════════════════════════════════
bgr = cv2.imread(str(rgb_path))          # [H, W, 3]  B=NIR, G=Green, R=Red
if bgr is None:
    sys.exit(f"[ERROR] cannot read {rgb_path}")

H, W = bgr.shape[:2]
print(f"         size: {W}×{H} px")

if ndvi_path is not None:
    ndvi = cv2.imread(str(ndvi_path), cv2.IMREAD_GRAYSCALE)   # [H, W] uint8
    if ndvi is None:
        ndvi_path = None

if ndvi_path is None:
    # derive from channels: NDVI = (NIR - Red) / (NIR + Red + eps)
    # bgr[:,:,0] = NIR,  bgr[:,:,2] = Red
    nir_f = bgr[:,:,0].astype(np.float32)
    red_f = bgr[:,:,2].astype(np.float32)
    ndvi_f = (nir_f - red_f) / (nir_f + red_f + 1e-6)        # [-1, 1]
    ndvi_f = np.clip((ndvi_f + 1.0) / 2.0, 0, 1)             # [0, 1]
    ndvi   = (ndvi_f * 255).astype(np.uint8)
    print("         ndvi: computed from NIR/Red channels")

if ndvi.shape[:2] != bgr.shape[:2]:
    ndvi = cv2.resize(ndvi, (W, H), interpolation=cv2.INTER_LINEAR)

# ═══════════════════════════════════════════════════════════════════════════════
# 4.  SQUARE CROP (centre)
# ═══════════════════════════════════════════════════════════════════════════════
SZ = min(H, W)
x0 = (W - SZ) // 2
y0 = (H - SZ) // 2
bgr_sq  = bgr [y0:y0+SZ, x0:x0+SZ].copy()
ndvi_sq = ndvi[y0:y0+SZ, x0:x0+SZ].copy()
print(f"         crop: {SZ}×{SZ} (x0={x0}, y0={y0})")

# ═══════════════════════════════════════════════════════════════════════════════
# 5.  INPUT 1 — RGNIR false-colour (CIR: display R=NIR, G=Red, B=Green)
#     bgr_sq channels: [0]=NIR  [1]=Green  [2]=Red
#     CIR output BGR:  [0]=Green [1]=Red   [2]=NIR
# ═══════════════════════════════════════════════════════════════════════════════
rgnir_bgr = cv2.merge([
    bgr_sq[:,:,1],   # display B = Green
    bgr_sq[:,:,2],   # display G = Red
    bgr_sq[:,:,0],   # display R = NIR  (vegetation → bright red)
])

# ═══════════════════════════════════════════════════════════════════════════════
# 6.  INPUT 2 — NDVI colormap (RdYlGn)
# ═══════════════════════════════════════════════════════════════════════════════
ndvi_norm = ndvi_sq.astype(np.float32) / 255.0           # [0, 1]
cmap      = plt.get_cmap("RdYlGn")
ndvi_rgba = (cmap(ndvi_norm) * 255).astype(np.uint8)     # [H,W,4] RGBA
ndvi_bgr  = cv2.cvtColor(ndvi_rgba[:,:,:3], cv2.COLOR_RGB2BGR)

# ═══════════════════════════════════════════════════════════════════════════════
# 7.  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[model]  loading …")
from ultralytics import YOLO
# import NMS from correct location in this local repo
try:
    from ultralytics.utils.ops import non_max_suppression
except ImportError:
    from ultralytics.utils.nms import non_max_suppression

model    = YOLO(str(WEIGHTS))
model_nn = model.model
model_nn.eval()
device   = next(model_nn.parameters()).device
print(f"         device={device}")

# build 4-ch tensor [B,G,NIR,NDVI] — same order as training loader
img4_sq    = np.concatenate([bgr_sq, ndvi_sq[:,:,None]], axis=2)    # [SZ,SZ,4]
img4_1280  = cv2.resize(img4_sq, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)
tensor     = torch.from_numpy(img4_1280.transpose(2,0,1)).float() / 255.0
tensor     = tensor.unsqueeze(0).to(device)                          # [1,4,1280,1280]

with torch.no_grad():
    raw = model_nn(tensor)   # (y, x) in eval mode; y=[1, 4+nc, num_anchors]

y = raw[0] if isinstance(raw, (tuple, list)) else raw
# safe: if shape is [1, num_anchors, 4+nc] transpose to [1, 4+nc, num_anchors]
if y.shape[1] > y.shape[2]:
    y = y.transpose(1, 2)

dets_nms = non_max_suppression(y, conf_thres=CONF, iou_thres=IOU, max_det=300)
dets     = dets_nms[0].cpu().numpy()   # [N, 6]: x1,y1,x2,y2,conf,cls

# scale from 1280×1280 → SZ×SZ
scale = SZ / IMGSZ
dets[:, :4] *= scale

class_counts = {}
for det in dets:
    c = int(det[5])
    class_counts[c] = class_counts.get(c, 0) + 1

print(f"         {len(dets)} detections")
for c, n in sorted(class_counts.items()):
    print(f"           cls {c} ({names_en.get(c,'?')}): {n}")

# ═══════════════════════════════════════════════════════════════════════════════
# 8.  OUTPUT PNG — CIR + rasterised boxes (for .png version)
# ═══════════════════════════════════════════════════════════════════════════════
out_bgr   = rgnir_bgr.copy()
THICK     = max(5, SZ // 500)
FS        = max(1.0, SZ / 1800)
FT        = max(2, SZ // 900)
PAD       = 6

for det in dets:
    x1, y1, x2, y2, conf, cls_f = det
    cls   = int(cls_f)
    color = COLORS_BGR[cls % len(COLORS_BGR)]
    label = f"{names_en.get(cls, str(cls))}  {conf:.2f}"

    cv2.rectangle(out_bgr, (int(x1), int(y1)), (int(x2), int(y2)), color, THICK)

    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FS, FT)
    ty = max(int(y1) - PAD, th + PAD)
    cv2.rectangle(out_bgr,
                  (int(x1) - PAD, ty - th - PAD),
                  (int(x1) + tw + PAD, ty + bl + PAD),
                  color, -1)
    cv2.putText(out_bgr, label, (int(x1), ty),
                cv2.FONT_HERSHEY_SIMPLEX, FS, (255, 255, 255), FT, cv2.LINE_AA)

# ═══════════════════════════════════════════════════════════════════════════════
# 9.  SAVE PNG  (lossless, 300 DPI, uint8)
# ═══════════════════════════════════════════════════════════════════════════════
OUT_DIR.mkdir(parents=True, exist_ok=True)

def save_png(bgr_img, stem):
    path = OUT_DIR / f"{stem}.png"
    rgb  = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil  = Image.fromarray(rgb)
    pil.save(str(path), format="PNG", optimize=False, dpi=(DPI, DPI))
    kb = path.stat().st_size // 1024
    print(f"  [png] {path.name}  ({kb} KB)")
    return path

print("\n[save PNGs]")
p_rgnir = save_png(rgnir_bgr, "input_rgnir")
p_ndvi  = save_png(ndvi_bgr,  "input_ndvi")
p_pred  = save_png(out_bgr,   "output_pred")

# ═══════════════════════════════════════════════════════════════════════════════
# 10.  SAVE SVG  (base64-embedded raster + vector elements for output)
# ═══════════════════════════════════════════════════════════════════════════════

def b64_png(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")

def svg_wrap(png_path: Path, size: int, vector: str = "") -> str:
    b64 = b64_png(png_path)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg"\n'
        f'     xmlns:xlink="http://www.w3.org/1999/xlink"\n'
        f'     width="{size}" height="{size}" viewBox="0 0 {size} {size}">\n'
        f'  <image x="0" y="0" width="{size}" height="{size}"\n'
        f'         xlink:href="data:image/png;base64,{b64}"\n'
        f'         preserveAspectRatio="xMidYMid meet"/>\n'
        f'{vector}'
        f'</svg>\n'
    )

def hex_to_rgb(h):
    h = h.lstrip("#")
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

FS_SVG     = max(28, SZ // 80)      # font size in px
SW_SVG     = max(6,  SZ // 450)     # stroke-width
PAD_SVG    = 6

def make_vector_boxes(dets_arr):
    parts = []
    for det in dets_arr:
        x1, y1, x2, y2, conf, cls_f = det
        cls   = int(cls_f)
        color = COLORS_HEX[cls % len(COLORS_HEX)]
        label = f"{names_en.get(cls, str(cls))}  {conf:.2f}"
        bw, bh = x2 - x1, y2 - y1

        # bounding-box rect
        parts.append(
            f'  <rect x="{x1:.1f}" y="{y1:.1f}" '
            f'width="{bw:.1f}" height="{bh:.1f}" '
            f'fill="none" stroke="{color}" '
            f'stroke-width="{SW_SVG}" stroke-linejoin="round"/>'
        )

        # label pill
        char_w   = FS_SVG * 0.60
        txt_w    = len(label) * char_w + PAD_SVG * 2
        txt_h    = FS_SVG * 1.4
        lbl_y    = max(y1 - PAD_SVG, txt_h + PAD_SVG)
        parts.append(
            f'  <rect x="{x1:.1f}" y="{lbl_y - txt_h:.1f}" '
            f'width="{txt_w:.1f}" height="{txt_h:.1f}" '
            f'fill="{color}" rx="4" ry="4"/>'
        )

        # label text
        parts.append(
            f'  <text x="{x1 + PAD_SVG:.1f}" '
            f'y="{lbl_y - FS_SVG * 0.18:.1f}" '
            f'font-family="DejaVu Sans,Arial,sans-serif" '
            f'font-size="{FS_SVG}" font-weight="bold" '
            f'fill="white">{label}</text>'
        )
    return "\n".join(parts) + "\n"

print("\n[save SVGs]")

def save_svg(content: str, stem: str):
    path = OUT_DIR / f"{stem}.svg"
    path.write_text(content, encoding="utf-8")
    kb = path.stat().st_size // 1024
    print(f"  [svg] {path.name}  ({kb} KB)")
    return path

save_svg(svg_wrap(p_rgnir, SZ),                          "input_rgnir")
save_svg(svg_wrap(p_ndvi,  SZ),                          "input_ndvi")
# output_pred SVG uses CLEAN rgnir base (no raster boxes) + vector boxes on top
save_svg(svg_wrap(p_rgnir, SZ, make_vector_boxes(dets)), "output_pred")

# ═══════════════════════════════════════════════════════════════════════════════
# 11.  SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print(f"""
╔══════════════════════════════════════════════════════════════╗
  DONE  ·  {OUT_DIR}
  Image : {best_stem}
  Size  : {SZ}×{SZ} px (center crop from {W}×{H})
  Dets  : {len(dets)} boxes  conf≥{CONF}  iou≤{IOU}
  ──────────────────────────────────────────────────────────
  Files :
    input_rgnir.png  — CIR false-colour (NIR→R, Red→G, Green→B)
    input_rgnir.svg  — same, base64-embedded
    input_ndvi.png   — NDVI RdYlGn colourmap
    input_ndvi.svg   — same, base64-embedded
    output_pred.png  — CIR + rasterised boxes
    output_pred.svg  — CIR base + VECTOR <rect>/<text> boxes
╚══════════════════════════════════════════════════════════════╝""")
