#!/usr/bin/env python3
"""
gen_paper_figs.py  —  Generate all paper figures for GMD-YOLO (dual-stream melon detection).

Output: paper_figures/{fig_dataset_samples, fig_comparison, fig_platform,
                        fig_confusion_matrix, fig_failcase}.{pdf,svg}
        paper_figures/confusion_matrix_values.txt

Run from the ultralytics repo root, or execute directly (paths are absolute).
"""
import os, sys, glob, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

random.seed(42)
np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
REPO_DIR  = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics'
OUT_DIR   = os.path.join(REPO_DIR, 'paper_figures')
RAW_DIR   = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
NDVI_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi'
DATA_YAML = os.path.join(RAW_DIR, 'data.yaml')

MODEL_CMX = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
MODEL_V11 = '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolo11s_1280_balanced_b4/weights/best.pt'
MODEL_EF  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp1_NoPretrain_1280px/weights/best.pt')

TEST_IMG_DIR = os.path.join(RAW_DIR, 'images', 'test')
TEST_LBL_DIR = os.path.join(RAW_DIR, 'labels', 'test')

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════
CLASS_NAMES  = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
CLASS_SHORT  = ['Pre-pol.', 'Post-pol.', 'Net form.', 'Sweetening']
NC = 4

# Per-class colors — high contrast on green leaf background
# RGB hex (matplotlib) and BGR (cv2); avoid green
BOX_RGB_HEX = ['#FF00FF', '#00BFFF', '#FFD700', '#FF4040']   # mag/skyblue/gold/red
BOX_BGR     = [(255, 0, 255), (255, 191, 0), (0, 215, 255), (64, 64, 255)]

GT_BGR      = (255, 255, 255)   # white GT boxes
GT_HEX      = '#FFFFFF'

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════════════════════
from ultralytics import YOLO

print("Loading models ...")
m_cmx = YOLO(MODEL_CMX)
m_v11 = YOLO(MODEL_V11)
m_ef  = YOLO(MODEL_EF)
print("  All 3 loaded.\n")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: IMAGE LOADING
# ══════════════════════════════════════════════════════════════════════════════
def load_4ch(img_path):
    """Return (bgr_3ch, bgr_ndvi_4ch) both HWC.  Returns (None,None) on error."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None, None
    fname = Path(img_path).stem + '.png'
    ndvi  = None
    for sp in ['test', 'val', 'train']:
        p = os.path.join(NDVI_DIR, 'images', sp, fname)
        if os.path.exists(p):
            ndvi = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            break
    if ndvi is None:
        nir = bgr[:, :, 2].astype(np.float32)
        red = bgr[:, :, 1].astype(np.float32)
        ndvi = (((nir - red) / (nir + red + 1e-6) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    if ndvi.shape != bgr.shape[:2]:
        ndvi = cv2.resize(ndvi, (bgr.shape[1], bgr.shape[0]))
    return bgr, np.dstack([bgr, ndvi])

def load_gt(img_path):
    """Load YOLO-format labels -> list of (cls, x1,y1,x2,y2)."""
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None, []
    H, W = bgr.shape[:2]
    stem = Path(img_path).stem
    boxes = []
    for sp in ['test', 'val', 'train']:
        lp = os.path.join(RAW_DIR, 'labels', sp, stem + '.txt')
        if os.path.exists(lp):
            for line in open(lp):
                p = line.strip().split()
                if not p:
                    continue
                cls = int(p[0])
                cx, cy, bw, bh = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                x1 = int((cx - bw / 2) * W); y1 = int((cy - bh / 2) * H)
                x2 = int((cx + bw / 2) * W); y2 = int((cy + bh / 2) * H)
                boxes.append((cls, x1, y1, x2, y2))
            break
    return bgr, boxes


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
def run_infer(model, img_path, is_4ch, conf=0.25, iou=0.6):
    """Run model.  Returns (bgr, boxes) where boxes=[(cls,x1,y1,x2,y2,conf),...]."""
    bgr, img4 = load_4ch(img_path)
    if bgr is None:
        return None, []
    src = img4 if is_4ch else img_path
    res = model.predict(source=src, conf=conf, iou=iou, imgsz=1280, verbose=False)
    boxes = []
    for box in res[0].boxes:
        cls = int(box.cls[0]); cf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append((cls, x1, y1, x2, y2, cf))
    return bgr, boxes


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: DRAWING
# ══════════════════════════════════════════════════════════════════════════════
def draw_pred(bgr, boxes, lw=6, font_scale=0.55):
    """Draw prediction boxes (per-class colors) on a copy; return RGB."""
    vis = bgr.copy()
    for (cls, x1, y1, x2, y2, cf) in boxes:
        color = BOX_BGR[cls]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, lw)
        lbl = f'{CLASS_SHORT[cls]} {cf:.2f}'
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        tc = (0, 0, 0) if sum(color) > 400 else (255, 255, 255)
        cv2.putText(vis, lbl, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, tc, 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def draw_gt_vis(bgr, gt_boxes, lw=6, font_scale=0.55):
    """Draw GT boxes (white) on a copy; return RGB."""
    vis = bgr.copy()
    for (cls, x1, y1, x2, y2) in gt_boxes:
        color = GT_BGR
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, lw)
        lbl = CLASS_SHORT[cls]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(vis, lbl, (x1 + 3, y1 - 3),
                    cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

def crop_to_roi(img_rgb, gt_boxes, pad_ratio=0.4, min_size=300):
    """Crop img_rgb to region of interest defined by gt_boxes, then resize to fixed height."""
    if not gt_boxes:
        H, W = img_rgb.shape[:2]
        return img_rgb
    xs = [b[1] for b in gt_boxes] + [b[3] for b in gt_boxes]
    ys = [b[2] for b in gt_boxes] + [b[4] for b in gt_boxes]
    x1c, x2c = min(xs), max(xs)
    y1c, y2c = min(ys), max(ys)
    bw, bh = x2c - x1c, y2c - y1c
    pad_x = max(int(bw * pad_ratio), min_size // 4)
    pad_y = max(int(bh * pad_ratio), min_size // 4)
    H, W = img_rgb.shape[:2]
    c1 = max(0, x1c - pad_x); c2 = min(W, x2c + pad_x)
    r1 = max(0, y1c - pad_y); r2 = min(H, y2c + pad_y)
    crop = img_rgb[r1:r2, c1:c2]
    if crop.size == 0:
        return img_rgb
    return crop

def resize_h(img, target_h=480):
    H, W = img.shape[:2]
    W2 = int(W * target_h / H)
    return cv2.resize(img, (W2, target_h), interpolation=cv2.INTER_AREA)

def save_fig(fig, name):
    for ext in ['pdf', 'svg']:
        p = os.path.join(OUT_DIR, f'{name}.{ext}')
        fig.savefig(p, format=ext, bbox_inches='tight', dpi=300)
        print(f"  Saved: {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# IoU utility (for CM and failcase)
# ══════════════════════════════════════════════════════════════════════════════
def box_iou(b1, b2):
    """b1, b2: (x1,y1,x2,y2). Returns IoU."""
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


# ══════════════════════════════════════════════════════════════════════════════
# BATCH INFERENCE — run GMD-YOLO on ALL test images (for CM + failcase)
# ══════════════════════════════════════════════════════════════════════════════
print("Running GMD-YOLO on all test images (conf=0.25) ...")
all_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg')))
print(f"  {len(all_test)} images found")

# all_results[img_path] = (bgr, pred_boxes, gt_boxes)
all_results = {}
for i, ip in enumerate(all_test):
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(all_test)}")
    bgr, pred = run_infer(m_cmx, ip, is_4ch=True, conf=0.25, iou=0.6)
    _, gt    = load_gt(ip)
    all_results[ip] = (bgr, pred, gt)

print(f"  Done. {len(all_results)} images processed.\n")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD CONFUSION MATRIX  (5×5 = NC + background)
# ══════════════════════════════════════════════════════════════════════════════
print("Building confusion matrix (conf=0.25, iou=0.5) ...")
IOU_THRES_CM = 0.5
cm = np.zeros((NC + 1, NC + 1), dtype=np.int64)   # [gt_cls/bg, pred_cls/bg]

for ip, (bgr, pred, gt) in all_results.items():
    gt_matched   = [False] * len(gt)
    pred_matched = [False] * len(pred)

    for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gt):
        best_iou, best_pi = 0.0, -1
        for pi, (pc, px1, py1, px2, py2, _) in enumerate(pred):
            if pred_matched[pi]:
                continue
            iou_v = box_iou((gx1,gy1,gx2,gy2), (px1,py1,px2,py2))
            if iou_v > best_iou:
                best_iou, best_pi = iou_v, pi
        if best_pi >= 0 and best_iou >= IOU_THRES_CM:
            pred_matched[best_pi] = True
            gt_matched[gi] = True
            cm[gc][pred[best_pi][0]] += 1          # gt→pred class
        else:
            cm[gc][NC] += 1                        # FN: gt_cls → background

    for pi, (pc, *_rest) in enumerate(pred):
        if not pred_matched[pi]:
            cm[NC][pc] += 1                        # FP: background → pred_cls

print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIND FAILURE CASES  (for fig_failcase)
# ══════════════════════════════════════════════════════════════════════════════
print("Collecting failure cases ...")

fn_cases  = []   # (img_path, gt_box_missed,   all_pred, all_gt)
fp_cases  = []   # (img_path, fp_pred_box,       all_pred, all_gt)
conf_cases = []  # (img_path, gt_box, wrong_pred, all_pred, all_gt)
# For glare/occlusion: images with small average box area AND low recall
glare_cases = []

for ip, (bgr, pred, gt) in all_results.items():
    gt_matched   = [False] * len(gt)
    pred_matched = [False] * len(pred)
    match_pairs  = []

    for gi, (gc, gx1, gy1, gx2, gy2) in enumerate(gt):
        best_iou, best_pi = 0.0, -1
        for pi, (pc, px1, py1, px2, py2, _) in enumerate(pred):
            if pred_matched[pi]:
                continue
            iou_v = box_iou((gx1,gy1,gx2,gy2), (px1,py1,px2,py2))
            if iou_v > best_iou:
                best_iou, best_pi = iou_v, pi
        if best_pi >= 0 and best_iou >= 0.5:
            pred_matched[best_pi] = True
            gt_matched[gi] = True
            match_pairs.append((gi, best_pi))
            if pred[best_pi][0] != gc:
                conf_cases.append((ip, gt[gi], pred[best_pi], pred, gt))
        else:
            fn_cases.append((ip, gt[gi], pred, gt))

    for pi, pbox in enumerate(pred):
        if not pred_matched[pi]:
            fp_cases.append((ip, pbox, pred, gt))

    # Glare/occlusion proxy: many GT boxes, low recall (< 50%)
    if len(gt) >= 3:
        recall = sum(gt_matched) / len(gt)
        if recall < 0.5:
            avg_area = np.mean([(b[3]-b[1])*(b[4]-b[2]) for b in gt]) if gt else 0
            glare_cases.append((recall, avg_area, ip, pred, gt))

glare_cases.sort(key=lambda x: x[0])

print(f"  FN cases: {len(fn_cases)}")
print(f"  FP cases: {len(fp_cases)}")
print(f"  Class-confusion cases: {len(conf_cases)}")
print(f"  Glare/occlusion candidates: {len(glare_cases)}\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — DATASET SAMPLES (4 rows × 2 cols)
# Each row = one stage; each cell = cropped zoom around one GT box
# ══════════════════════════════════════════════════════════════════════════════
print("[1/5] fig_dataset_samples ...")

# Per class: find images with that class, pick boxes with decent area
def pick_sample_crops(cls_target, n=2, min_area=1500, max_area=400000):
    """Return list of (bgr_crop_rgb, cls) — crops around single GT boxes."""
    candidates = []
    for ip in all_test:
        _, gt = load_gt(ip)
        bgr = cv2.imread(ip)
        if bgr is None: continue
        H, W = bgr.shape[:2]
        for (cls, x1, y1, x2, y2) in gt:
            if cls != cls_target: continue
            area = (x2 - x1) * (y2 - y1)
            if not (min_area <= area <= max_area): continue
            # Prefer boxes not too close to edge
            margin = 50
            if x1 < margin or y1 < margin or x2 > W-margin or y2 > H-margin: continue
            candidates.append((area, ip, (cls, x1, y1, x2, y2)))
    # Sort by moderate area (not too small, not too large → clearest zoom)
    # Prefer boxes around 5000-30000 px²
    def score(tup):
        area = tup[0]
        return abs(np.log(area) - np.log(12000))   # penalty for deviation from ideal
    candidates.sort(key=score)
    seen_imgs = set()
    crops = []
    for _, ip, box in candidates:
        if ip in seen_imgs: continue
        bgr = cv2.imread(ip)
        cls, x1, y1, x2, y2 = box
        H, W = bgr.shape[:2]
        bw, bh = x2 - x1, y2 - y1
        pad = max(int(max(bw, bh) * 0.8), 80)
        c1 = max(0, x1 - pad); c2 = min(W, x2 + pad)
        r1 = max(0, y1 - pad); r2 = min(H, y2 + pad)
        crop_bgr = bgr[r1:r2, c1:c2].copy()
        # Recalculate box coords in crop
        nx1, ny1 = x1 - c1, y1 - r1
        nx2, ny2 = x2 - c1, y2 - r1
        # Draw box
        color = BOX_BGR[cls]
        cv2.rectangle(crop_bgr, (nx1, ny1), (nx2, ny2), color, 5)
        crops.append(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        seen_imgs.add(ip)
        if len(crops) >= n:
            break
    return crops

CROP_H = 420   # display height per cell in pixels
fig, axes = plt.subplots(4, 2, figsize=(7.5, 10.0))
fig.subplots_adjust(wspace=0.03, hspace=0.04, left=0.13, right=0.99, top=0.97, bottom=0.01)

for cls in range(4):
    crops = pick_sample_crops(cls, n=2)
    for col in range(2):
        ax = axes[cls][col]
        if col < len(crops):
            img = crops[col]
            # Square-ish crop
            h, w = img.shape[:2]
            if w > h:
                start = (w - h) // 2
                img = img[:, start:start+h]
            ax.imshow(img)
        else:
            ax.set_facecolor('#222')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(BOX_RGB_HEX[cls]); spine.set_linewidth(2.5)

    # Stage label on left of row
    axes[cls][0].set_ylabel(CLASS_NAMES[cls], fontsize=9.5, fontweight='bold',
                             rotation=90, va='center', labelpad=8)

# Column headers
axes[0][0].set_title('Sample 1', fontsize=9, pad=4)
axes[0][1].set_title('Sample 2', fontsize=9, pad=4)

legend_handles = [mpatches.Patch(color=BOX_RGB_HEX[i], label=CLASS_NAMES[i]) for i in range(4)]
fig.legend(handles=legend_handles, loc='lower center', ncol=2, fontsize=8.5,
           frameon=True, bbox_to_anchor=(0.55, -0.02))

save_fig(fig, 'fig_dataset_samples')
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — QUALITATIVE COMPARISON  (rows=scenes, cols=GT|V11|EF|Ours)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/5] fig_comparison (running V11 & EF inference on 4 scenes) ...")

# Curated 4 scenes: pre-pol / post-pol / net / sweetening
COMP_IMGS = [
    os.path.join(TEST_IMG_DIR, '2019_0101_001345_409.jpg'),   # pre-pol (7 boxes)
    os.path.join(TEST_IMG_DIR, '2023_0610_144821_322.jpg'),   # post-pol (15, hardest cls)
    os.path.join(TEST_IMG_DIR, '2023_0620_161134_014.jpg'),   # net formation (15)
    os.path.join(TEST_IMG_DIR, '2023_0620_174748_521.jpg'),   # sweetening (106)
]
COMP_LBLS = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']

COL_TITLES = ['Ground Truth', 'YOLOv11s\n(R/G/NIR)', 'EarlyFusion\n(4ch)', 'GMD-YOLO\n(Ours)']
CROP_H_COMP = 380

fig, axes = plt.subplots(4, 4, figsize=(14.0, 12.5))
fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.10, right=0.99, top=0.95, bottom=0.03)

for col, title in enumerate(COL_TITLES):
    axes[0][col].set_title(title, fontsize=9.5, fontweight='bold', pad=6)

for row, ip in enumerate(COMP_IMGS):
    if not os.path.exists(ip):
        print(f"  WARNING: {ip} not found, skipping row {row}")
        for col in range(4):
            axes[row][col].axis('off')
        continue

    bgr, gt = load_gt(ip)

    # Select crop region from first 5 GT boxes (avoid 100+ boxes filling whole image)
    gt_for_crop = gt[:8] if len(gt) > 8 else gt

    # Run V11 and EF
    _, pred_v11 = run_infer(m_v11, ip, is_4ch=False, conf=0.25, iou=0.6)
    _, pred_ef  = run_infer(m_ef,  ip, is_4ch=True,  conf=0.25, iou=0.6)
    _, pred_cmx = run_infer(m_cmx, ip, is_4ch=True,  conf=0.25, iou=0.6)

    # Build 4 annotated images
    imgs_rgb = [
        draw_gt_vis(bgr,  gt,       lw=7),
        draw_pred(bgr,    pred_v11, lw=7),
        draw_pred(bgr,    pred_ef,  lw=7),
        draw_pred(bgr,    pred_cmx, lw=7),
    ]

    # Shared crop region (from GT boxes subset)
    crop_box = gt_for_crop
    for col, im in enumerate(imgs_rgb):
        crop = crop_to_roi(im, crop_box, pad_ratio=0.35)
        crop_resized = resize_h(crop, target_h=CROP_H_COMP)
        axes[row][col].imshow(crop_resized)
        axes[row][col].set_xticks([]); axes[row][col].set_yticks([])

    axes[row][0].set_ylabel(COMP_LBLS[row], fontsize=9.5, fontweight='bold',
                             rotation=90, va='center', labelpad=8)

# Legend
handles = ([mpatches.Patch(color='white', label='GT', ec='gray')] +
           [mpatches.Patch(color=BOX_RGB_HEX[i], label=CLASS_NAMES[i]) for i in range(4)])
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8.5,
           frameon=True, bbox_to_anchor=(0.55, -0.01))

save_fig(fig, 'fig_comparison')
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — PLATFORM  (placeholder with 3 panels)
# ══════════════════════════════════════════════════════════════════════════════
print("[3/5] fig_platform (placeholder layout) ...")

fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.88, bottom=0.04)

panel_titles = [
    '(a)  UAV with ModalAI VOXL 2',
    '(b)  MapIR Survey3 Camera',
    '(c)  Reflectance Calibration Panel',
]
panel_notes = [
    'PLACEHOLDER\nProvide photo of UAV\ncarrying VOXL 2 payload',
    'PLACEHOLDER\nProvide close-up photo\nof MapIR Survey3 camera',
    'PLACEHOLDER\nProvide photo of\nMapIR calibration panel',
]
placeholder_colors = ['#E8EEF4', '#EAF4EA', '#F4F0EA']

for ax, title, note, color in zip(axes, panel_titles, panel_notes, placeholder_colors):
    ax.set_facecolor(color)
    ax.text(0.5, 0.55, note, ha='center', va='center',
            fontsize=10, color='#555555', style='italic',
            fontfamily='DejaVu Sans', transform=ax.transAxes)
    ax.set_title(title, fontsize=9.5, fontweight='bold', pad=5)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(1.2); sp.set_edgecolor('#999')

save_fig(fig, 'fig_platform')
print("  Done.  *** NEEDS REAL PHOTOS for panels (a), (b), (c) ***\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — CONFUSION MATRIX  (5×5, row-normalized)
# ══════════════════════════════════════════════════════════════════════════════
print("[4/5] fig_confusion_matrix ...")

labels_5 = CLASS_NAMES + ['Background']

# Row-normalize (recall-normalized)
cm_f   = cm.astype(np.float64)
row_sum = cm_f.sum(axis=1, keepdims=True).clip(min=1)
cm_norm = cm_f / row_sum

# Save raw values
with open(os.path.join(OUT_DIR, 'confusion_matrix_values.txt'), 'w') as f:
    f.write(f"GMD-YOLO Confusion Matrix  (test-383, conf=0.25, iou_match=0.5)\n")
    f.write(f"Rows = Ground Truth (including Background/FP row)\n")
    f.write(f"Cols = Predicted    (including Background/FN col)\n\n")
    header = '  '.join(f'{l[:6]:>8}' for l in labels_5)
    f.write(f"{'GT/Pred':>18}  {header}\n")
    for i, row_lbl in enumerate(labels_5):
        row_str = '  '.join(f'{int(cm[i,j]):>8d}' for j in range(NC+1))
        f.write(f"  {row_lbl:>16}  {row_str}\n")
    f.write("\n--- Normalized (row sum = 1.0) ---\n")
    f.write(f"{'GT/Pred':>18}  {header}\n")
    for i, row_lbl in enumerate(labels_5):
        row_str = '  '.join(f'{cm_norm[i,j]:>8.3f}' for j in range(NC+1))
        f.write(f"  {row_lbl:>16}  {row_str}\n")
print(f"  Saved: {os.path.join(OUT_DIR, 'confusion_matrix_values.txt')}")

# Plot
fig, ax = plt.subplots(figsize=(6.5, 5.8))

# Colorblind-friendly colormap (sequential, perceptually uniform)
import matplotlib.colors as mcolors
cmap_name = 'Blues'
im = ax.imshow(cm_norm, cmap=cmap_name, vmin=0, vmax=1, aspect='auto')

# Annotate each cell
for i in range(NC + 1):
    for j in range(NC + 1):
        v   = cm_norm[i, j]
        raw = int(cm[i, j])
        tc  = 'white' if v > 0.55 else 'black'
        # Show normalized value; raw count in smaller text below
        ax.text(j, i - 0.12, f'{v:.2f}', ha='center', va='center',
                fontsize=10.5, color=tc, fontweight='bold',
                fontfamily='DejaVu Sans')
        ax.text(j, i + 0.22, f'({raw})', ha='center', va='center',
                fontsize=7.5, color=tc, fontfamily='DejaVu Sans')

tick_lbls = ['\n'.join(l.split()) for l in labels_5]   # wrap long labels
ax.set_xticks(range(NC + 1)); ax.set_yticks(range(NC + 1))
ax.set_xticklabels(tick_lbls, fontsize=8.5, fontfamily='DejaVu Sans')
ax.set_yticklabels(tick_lbls, fontsize=8.5, fontfamily='DejaVu Sans')
ax.set_xlabel('Predicted label', fontsize=9.5, labelpad=8)
ax.set_ylabel('Ground truth label', fontsize=9.5, labelpad=8)
ax.set_title(f'Confusion Matrix — GMD-YOLO on test-383\n(conf=0.25, IoU match≥0.50, row-normalized)',
             fontsize=9, pad=8)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=8)
cb.set_label('Recall fraction', fontsize=8.5)

fig.tight_layout()
save_fig(fig, 'fig_confusion_matrix')
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — FAILURE CASES  (4–6 panels, each one error type)
# ══════════════════════════════════════════════════════════════════════════════
print("[5/5] fig_failcase ...")

FAIL_H = 420

def make_failcase_panel(ip, gt_boxes, pred_boxes,
                        highlight_gt=None, highlight_pred=None, title=''):
    """
    Draw an annotated panel for a failure case.
    - GT boxes: white (thin dashed effect via double-draw)
    - Predicted boxes: class colors
    - highlight_gt: specific GT box to circle (the missed one or confused)
    - highlight_pred: specific pred box to circle (the FP or confused)
    Returns RGB numpy crop.
    """
    bgr = cv2.imread(ip)
    vis = bgr.copy()

    # Draw all GT (white, thin)
    for (cls, x1, y1, x2, y2) in gt_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (180, 180, 180), 3)

    # Draw all predictions (class color, thick)
    for (cls, x1, y1, x2, y2, cf) in pred_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_BGR[cls], 5)

    # Highlight missed GT (bright red outline + label "MISSED")
    if highlight_gt is not None:
        cls, x1, y1, x2, y2 = highlight_gt[:5]
        cv2.rectangle(vis, (x1-4, y1-4), (x2+4, y2+4), (0, 0, 255), 6)
        lbl = f'GT: {CLASS_SHORT[cls]}'
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        cv2.rectangle(vis, (x1, y2+2), (x1+tw+6, y2+th+12), (0, 0, 200), -1)
        cv2.putText(vis, lbl, (x1+3, y2+th+5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Highlight FP / wrong prediction (orange outline + label "FP" or "WRONG CLS")
    if highlight_pred is not None:
        cls, x1, y1, x2, y2, cf = highlight_pred
        cv2.rectangle(vis, (x1-4, y1-4), (x2+4, y2+4), (0, 128, 255), 6)
        lbl = f'PRED: {CLASS_SHORT[cls]} {cf:.2f}'
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_DUPLEX, 0.65, 2)
        cv2.rectangle(vis, (x1, y1-th-12), (x1+tw+6, y1), (0, 128, 200), -1)
        cv2.putText(vis, lbl, (x1+3, y1-3),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    if highlight_gt is not None:
        roi_box = [highlight_gt]
    elif highlight_pred is not None:
        roi_box = [(highlight_pred[0], highlight_pred[1], highlight_pred[2],
                    highlight_pred[3], highlight_pred[4])]
    else:
        roi_box = gt_boxes[:3] if gt_boxes else []

    crop = crop_to_roi(vis_rgb, roi_box, pad_ratio=0.6, min_size=200)
    return crop


def pick_best(cases, key_fn, n=1, min_area=800, max_area=500000):
    """Pick n diverse cases from a list, filtered by area."""
    filtered = []
    for c in cases:
        if len(c) >= 3:
            box = c[1]
            area = (box[3] - box[1]) * (box[4] - box[2]) if len(box) >= 5 else 0
            if min_area <= area <= max_area:
                filtered.append(c)
    seen = set()
    result = []
    for c in sorted(filtered, key=key_fn):
        ip = c[0]
        if ip not in seen:
            seen.add(ip)
            result.append(c)
        if len(result) >= n:
            break
    return result

# Pick one example per failure type
# Type 1: Missed detection (FN)
fn_pick = pick_best(fn_cases,
                    key_fn=lambda c: -(c[1][3]-c[1][1])*(c[1][4]-c[1][2]),
                    n=1)
# Type 2: False positive
fp_pick = pick_best(fp_cases,
                    key_fn=lambda c: -c[1][5],   # highest conf FP
                    n=1)
# Type 3a: net↔sweetening confusion (cls 2 → 3 or 3 → 2)
net_sweet_conf = [c for c in conf_cases
                  if {c[1][0], c[2][0]} == {2, 3}]
nsc_pick = pick_best(net_sweet_conf,
                     key_fn=lambda c: -(c[1][3]-c[1][1])*(c[1][4]-c[1][2]),
                     n=1)
# Type 3b: any class confusion (fallback if 2↔3 not found)
if not nsc_pick:
    nsc_pick = pick_best(conf_cases,
                         key_fn=lambda c: -(c[1][3]-c[1][1])*(c[1][4]-c[1][2]),
                         n=1)

# Type 4: Glare / heavy occlusion (low recall images)
glare_pick = []
for (recall, avg_a, ip, pred, gt) in glare_cases[:10]:
    if avg_a < 500: continue
    glare_pick.append((ip, pred, gt))
    if len(glare_pick) >= 1:
        break

fail_panels = []
fail_titles = []

if fn_pick:
    ip, missed_gt, pred, gt = fn_pick[0]
    crop = make_failcase_panel(ip, gt, pred, highlight_gt=missed_gt)
    fail_panels.append(crop)
    fail_titles.append(f'(1) Missed detection\n(GT: {CLASS_SHORT[missed_gt[0]]})')

if fp_pick:
    ip, fp_box, pred, gt = fp_pick[0]
    crop = make_failcase_panel(ip, gt, pred, highlight_pred=fp_box)
    fail_panels.append(crop)
    fail_titles.append(f'(2) False positive\n(Pred: {CLASS_SHORT[fp_box[0]]} {fp_box[5]:.2f})')

if nsc_pick:
    ip, gt_box, pred_box, pred, gt = nsc_pick[0]
    crop = make_failcase_panel(ip, gt, pred, highlight_gt=gt_box, highlight_pred=pred_box)
    fail_panels.append(crop)
    gt_cls, pred_cls = gt_box[0], pred_box[0]
    fail_titles.append(f'(3) Class confusion\n{CLASS_SHORT[gt_cls]} → {CLASS_SHORT[pred_cls]}')

if glare_pick:
    ip, pred, gt = glare_pick[0]
    crop = make_failcase_panel(ip, gt, pred)
    fail_panels.append(crop)
    fail_titles.append('(4) Low recall scene\n(occlusion / glare)')

# Ensure we have at least 4 panels; pad with extra FN/FP if needed
if len(fn_cases) > 1 and len(fail_panels) < 4:
    for c in fn_cases[1:]:
        if len(fail_panels) >= 4: break
        ip, missed_gt, pred, gt = c
        crop = make_failcase_panel(ip, gt, pred, highlight_gt=missed_gt)
        fail_panels.append(crop)
        fail_titles.append(f'(5) Missed detection\n(GT: {CLASS_SHORT[missed_gt[0]]})')

n_panels = len(fail_panels)
n_cols = min(n_panels, 3); n_rows = (n_panels + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.8 * n_rows))
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = axes[np.newaxis, :]
elif n_cols == 1:
    axes = axes[:, np.newaxis]

legend_handles = [
    mpatches.Patch(color='white', label='GT box (gray outline)', ec='gray'),
    mpatches.Patch(color='red',   label='Missed GT (red outline)'),
    mpatches.Patch(color='#FF8000', label='Highlighted Pred (orange)'),
]
for i, cls in enumerate(range(4)):
    legend_handles.append(mpatches.Patch(color=BOX_RGB_HEX[cls], label=f'Pred: {CLASS_SHORT[cls]}'))

for idx, (crop, title) in enumerate(zip(fail_panels, fail_titles)):
    r, c = idx // n_cols, idx % n_cols
    ax = axes[r][c]
    crop_disp = resize_h(crop, target_h=FAIL_H)
    ax.imshow(crop_disp)
    ax.set_title(title, fontsize=9, fontweight='bold', pad=5, fontfamily='DejaVu Sans')
    ax.set_xticks([]); ax.set_yticks([])

# Hide unused axes
for idx in range(n_panels, n_rows * n_cols):
    axes[idx // n_cols][idx % n_cols].axis('off')

fig.legend(handles=legend_handles, loc='lower center', ncol=3, fontsize=8.5,
           frameon=True, bbox_to_anchor=(0.5, -0.04))
fig.suptitle('GMD-YOLO Failure Cases — test-383, conf=0.25', fontsize=10, y=1.01)
fig.tight_layout(pad=0.5)

save_fig(fig, 'fig_failcase')
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print(f"All figures written to: {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    size_kb = os.path.getsize(fpath) // 1024
    print(f"  {f:45s}  {size_kb:5d} KB")
print()
print("NOTES:")
print("  fig_platform.pdf/svg  —  NEEDS REAL PHOTOS:")
print("    (a) UAV + ModalAI VOXL 2  (photograph)")
print("    (b) MapIR Survey3 camera  (close-up photograph)")
print("    (c) MapIR reflectance calibration panel  (photograph)")
print("  Replace each placeholder by providing the photo files.")
print("  Confusion matrix raw counts + normalized values:")
print(f"    {os.path.join(OUT_DIR, 'confusion_matrix_values.txt')}")
