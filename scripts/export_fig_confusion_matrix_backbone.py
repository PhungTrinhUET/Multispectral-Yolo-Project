#!/usr/bin/env python3
"""
Confusion matrix for CMX_Backbone (Exp20) — test-383, conf=0.25, iou_infer=0.6, iou_match=0.5.
Row-normalised 5×5 (4 classes + Background). Overwrites paper_figures/fig_confusion_matrix.{pdf,svg}.
Style: serif Latin Modern, x-labels rotated 30°, Blues colormap.
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_DIR = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics'
OUT_DIR  = os.path.join(REPO_DIR, 'paper_figures')
RAW_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
NDVI_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi'
CKPT     = ('/home/tower2080/Documents/YOLO-Knowledge-Distillation-TinyML'
            '/ultralytics/runs/detect/runs/ablation_position_backbone'
            '/CMX_Backbone/weights/best.pt')

CONF_INF  = 0.25
IOU_INF   = 0.6
IOU_MATCH = 0.5
NC        = 4
CLASS_NAMES = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
LABELS_5    = CLASS_NAMES + ['Background']

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['CMU Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.size':        10,
    'axes.titlesize':   10,
    'axes.labelsize':   10,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
})

# ── helpers ──────────────────────────────────────────────────────────────────
def load_4ch(ip):
    bgr = cv2.imread(ip)
    if bgr is None:
        return None, None
    stem = Path(ip).stem
    ndvi = None
    for sp in ['test', 'val', 'train']:
        p = os.path.join(NDVI_DIR, 'images', sp, stem + '.png')
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

def load_gt(ip):
    bgr = cv2.imread(ip)
    if bgr is None:
        return []
    H, W = bgr.shape[:2]
    stem = Path(ip).stem
    boxes = []
    for sp in ['test', 'val', 'train']:
        lp = os.path.join(RAW_DIR, 'labels', sp, stem + '.txt')
        if os.path.exists(lp):
            for line in open(lp):
                p = line.strip().split()
                if not p:
                    continue
                c = int(p[0])
                cx_, cy_, bw_, bh_ = float(p[1]), float(p[2]), float(p[3]), float(p[4])
                x1 = int((cx_ - bw_ / 2) * W); y1 = int((cy_ - bh_ / 2) * H)
                x2 = int((cx_ + bw_ / 2) * W); y2 = int((cy_ + bh_ / 2) * H)
                boxes.append((c, x1, y1, x2, y2))
            break
    return boxes

def box_iou(b1, b2):
    xi1 = max(b1[0], b2[0]); yi1 = max(b1[1], b2[1])
    xi2 = min(b1[2], b2[2]); yi2 = min(b1[3], b2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter == 0:
        return 0.0
    return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter)

# ── load model ───────────────────────────────────────────────────────────────
from ultralytics import YOLO
print(f"Loading checkpoint: {CKPT}")
model = YOLO(CKPT)
print("  Done.\n")

# ── run inference ─────────────────────────────────────────────────────────────
TEST_IMG_DIR = os.path.join(RAW_DIR, 'images', 'test')
all_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg')))
print(f"Running inference on {len(all_test)} test images (conf={CONF_INF}, iou_infer={IOU_INF}) ...")

all_results = {}
for i, ip in enumerate(all_test):
    if (i + 1) % 50 == 0 or i == 0:
        print(f"  {i+1}/{len(all_test)}")
    _, img4 = load_4ch(ip)
    if img4 is None:
        continue
    res = model.predict(source=img4, conf=CONF_INF, iou=IOU_INF, imgsz=1280, verbose=False)
    preds = [(int(b.cls[0]), int(b.xyxy[0][0]), int(b.xyxy[0][1]),
              int(b.xyxy[0][2]), int(b.xyxy[0][3]), float(b.conf[0]))
             for b in res[0].boxes]
    gt = load_gt(ip)
    all_results[ip] = (preds, gt)

print(f"  Done. {len(all_results)} images.\n")

# ── build confusion matrix ────────────────────────────────────────────────────
print(f"Building confusion matrix (iou_match={IOU_MATCH}) ...")
cm = np.zeros((NC + 1, NC + 1), dtype=np.int64)   # [gt_cls/bg_row, pred_cls/bg_col]

for ip, (pred, gt) in all_results.items():
    pred_matched = [False] * len(pred)

    for (gc, gx1, gy1, gx2, gy2) in gt:
        best_iou, best_pi = 0.0, -1
        for pi, (pc, px1, py1, px2, py2, _) in enumerate(pred):
            if pred_matched[pi]:
                continue
            iou_v = box_iou((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
            if iou_v > best_iou:
                best_iou, best_pi = iou_v, pi
        if best_pi >= 0 and best_iou >= IOU_MATCH:
            pred_matched[best_pi] = True
            cm[gc][pred[best_pi][0]] += 1     # GT cls → predicted cls
        else:
            cm[gc][NC] += 1                   # FN → background col

    for pi, (pc, *_) in enumerate(pred):
        if not pred_matched[pi]:
            cm[NC][pc] += 1                   # FP → background row

# ── normalise (row = recall) ──────────────────────────────────────────────────
cm_f    = cm.astype(np.float64)
row_sum = cm_f.sum(axis=1, keepdims=True).clip(min=1)
cm_norm = cm_f / row_sum

# ── print report ─────────────────────────────────────────────────────────────
print("\n=== Raw confusion matrix (rows=GT, cols=Pred) ===")
hdr = f"{'':>18} " + '  '.join(f'{l[:8]:>9}' for l in LABELS_5)
print(hdr)
for i, rl in enumerate(LABELS_5):
    row = '  '.join(f'{int(cm[i,j]):>9d}' for j in range(NC+1))
    print(f"  {rl:>16}  {row}")

print("\n=== Normalised (row-sum = 1.0) ===")
print(hdr)
for i, rl in enumerate(LABELS_5):
    row = '  '.join(f'{cm_norm[i,j]:>9.3f}' for j in range(NC+1))
    print(f"  {rl:>16}  {row}")

print("\n=== KEY NUMBERS FOR PAPER ===")
print(f"Per-class recall (diagonal):")
for i, cls in enumerate(CLASS_NAMES):
    print(f"  {cls:<22} recall={cm_norm[i,i]:.3f}  ({int(cm[i,i])}/{int(cm_f[i].sum())})")

print(f"\nOff-diagonal (notable confusions):")
for i in range(NC):
    for j in range(NC):
        if i != j and cm_norm[i, j] >= 0.02:
            print(f"  {CLASS_NAMES[i]} → {CLASS_NAMES[j]}: {cm_norm[i,j]:.3f}  ({int(cm[i,j])})")

fp_row = cm_f[NC]   # background row = FP distribution
fp_total = fp_row.sum()
print(f"\nFalse Positive distribution (total FP={int(fp_total)}):")
for j in range(NC):
    print(f"  Predicted as {CLASS_NAMES[j]:<22}: {int(fp_row[j]):>4}  ({fp_row[j]/max(fp_total,1)*100:.1f}%)")

# ── save raw numbers ──────────────────────────────────────────────────────────
txt_path = os.path.join(OUT_DIR, 'confusion_matrix_values.txt')
with open(txt_path, 'w') as f:
    f.write(f"CMX_Backbone (Exp20) Confusion Matrix\n")
    f.write(f"test-383, conf={CONF_INF}, iou_infer={IOU_INF}, iou_match={IOU_MATCH}\n")
    f.write(f"Rows=Ground Truth  Cols=Predicted  (last row/col=Background)\n\n")
    hdr2 = '  '.join(f'{l[:9]:>10}' for l in LABELS_5)
    f.write(f"{'GT/Pred':>18}  {hdr2}\n")
    for i, rl in enumerate(LABELS_5):
        row = '  '.join(f'{int(cm[i,j]):>10d}' for j in range(NC+1))
        f.write(f"  {rl:>16}  {row}\n")
    f.write("\n--- Normalised ---\n")
    f.write(f"{'GT/Pred':>18}  {hdr2}\n")
    for i, rl in enumerate(LABELS_5):
        row = '  '.join(f'{cm_norm[i,j]:>10.3f}' for j in range(NC+1))
        f.write(f"  {rl:>16}  {row}\n")
print(f"\nSaved: {txt_path}")

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 5.8))

im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1, aspect='auto')

for i in range(NC + 1):
    for j in range(NC + 1):
        v   = cm_norm[i, j]
        raw = int(cm[i, j])
        tc  = 'white' if v > 0.55 else 'black'
        ax.text(j, i - 0.12, f'{v:.2f}', ha='center', va='center',
                fontsize=10.5, color=tc, fontweight='bold')
        ax.text(j, i + 0.22, f'({raw})', ha='center', va='center',
                fontsize=7.5, color=tc)

# x-labels rotated 30°, ha='right' to avoid overlap
tick_lbls_x = ['\n'.join(l.split()) for l in LABELS_5]
tick_lbls_y = tick_lbls_x

ax.set_xticks(range(NC + 1))
ax.set_yticks(range(NC + 1))
ax.set_xticklabels(tick_lbls_x, fontsize=8.5, rotation=30, ha='right', rotation_mode='anchor')
ax.set_yticklabels(tick_lbls_y, fontsize=8.5)
ax.set_xlabel('Predicted label', fontsize=10, labelpad=10)
ax.set_ylabel('Ground truth label', fontsize=10, labelpad=8)
ax.set_title(
    r'(conf$=0.25$, IoU match$\geq 0.50$, row-normalised)',
    fontsize=9.5, pad=8)

cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.ax.tick_params(labelsize=8)
cb.set_label('Recall fraction', fontsize=9)

fig.tight_layout()

for ext in ['pdf', 'svg']:
    p = os.path.join(OUT_DIR, f'fig_confusion_matrix.{ext}')
    fig.savefig(p, format=ext, bbox_inches='tight', dpi=300)
    print(f"Saved: {p}  ({os.path.getsize(p)//1024} KB)")
plt.close(fig)

print("\nDone.")
