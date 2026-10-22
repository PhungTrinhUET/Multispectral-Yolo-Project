"""
Export 3 architecture-panel images (arch_rgnir, arch_ndvi, arch_pred) from one test image.
Same false-color pipeline as gen_figures.py: cv2.BGR -> RGB = CIR composite (NIR in R channel).
"""
import os, sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as cm
import matplotlib.colors as mcolors

os.chdir('/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')
sys.path.insert(0, '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')

RAW_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
NDVI_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi'
FIG_DIR  = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics/paper_figures'
MODEL_CMX = 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt'

STEM = '2023_0620_174738_516'
IMG_PATH = os.path.join(RAW_DIR, 'images', 'test', STEM + '.jpg')
LBL_PATH = os.path.join(RAW_DIR, 'labels', 'test', STEM + '.txt')
NDVI_PATH = os.path.join(NDVI_DIR, 'images', 'test', STEM + '.png')

# Box colours chosen to contrast against the CIR magenta/red background.
# CIR background is dominated by red-pink (NIR in R-channel), so we use
# yellow and cyan which pop clearly against that tone.
# Format: (R, G, B) for drawing on the RGB canvas.
CLASS_COLORS_RGB = [
    (255, 220,   0),  # bright yellow  – pre-pollination
    ( 50, 220, 255),  # cyan           – post-pollination
    ( 50, 255, 130),  # lime green     – net-formation
    (255, 220,   0),  # bright yellow  – sweetening (dominant class; yellow on magenta is vivid)
]
CLASS_NAMES = ['Pre-poll.', 'Post-poll.', 'Net form.', 'Sweetening']

# ── 1. Load full image & label ────────────────────────────────────────────────
bgr = cv2.imread(IMG_PATH)
assert bgr is not None, f'Cannot read {IMG_PATH}'
H, W = bgr.shape[:2]   # 3000 x 4000

# Parse boxes (normalised YOLO)
boxes = []
for line in open(LBL_PATH):
    parts = line.strip().split()
    if not parts: continue
    c, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1 = int((cx - bw/2) * W); y1 = int((cy - bh/2) * H)
    x2 = int((cx + bw/2) * W); y2 = int((cy + bh/2) * H)
    boxes.append((c, x1, y1, x2, y2))

# ── 2. Find best square crop centred on box density ───────────────────────────
CROP_SIZE = min(H, W)   # 3000 px square from 3000x4000 image

if boxes:
    cx_mean = int(np.mean([(b[1]+b[3])//2 for b in boxes]))
    cy_mean = int(np.mean([(b[2]+b[4])//2 for b in boxes]))
else:
    cx_mean, cy_mean = W//2, H//2

# clamp so crop stays inside image
half = CROP_SIZE // 2
x0 = max(0, min(cx_mean - half, W - CROP_SIZE))
y0 = max(0, min(cy_mean - half, H - CROP_SIZE))
x1c, y1c = x0, y0
x2c, y2c = x0 + CROP_SIZE, y0 + CROP_SIZE

print(f'Image: {STEM}.jpg  ({W}x{H})')
print(f'Boxes: {len(boxes)}  |  crop: ({x1c},{y1c}) -> ({x2c},{y2c})  size={CROP_SIZE}x{CROP_SIZE}')

# ── 3. arch_rgnir.jpg — CIR false-colour (same pipeline as gen_figures.py) ───
bgr_crop = bgr[y1c:y2c, x1c:x2c]
rgb_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)   # NIR→R, Red→G, Green→B
out_rgnir = os.path.join(FIG_DIR, 'arch_rgnir.jpg')
cv2.imwrite(out_rgnir, cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 97])
print(f'arch_rgnir.jpg: {rgb_crop.shape[1]}x{rgb_crop.shape[0]} px')

# ── 4. arch_ndvi.jpg — NDVI viridis, percentile-normalised ───────────────────
ndvi_uint8 = cv2.imread(NDVI_PATH, cv2.IMREAD_UNCHANGED)
assert ndvi_uint8 is not None, f'Cannot read NDVI: {NDVI_PATH}'
ndvi_float = ndvi_uint8.astype(np.float32) / 255.0 * 2.0 - 1.0
ndvi_crop = ndvi_float[y1c:y2c, x1c:x2c]

# Auto-scale to p2–p98 so contrast fills the full colormap range
p2, p98 = np.percentile(ndvi_crop, 2), np.percentile(ndvi_crop, 98)
norm = mcolors.Normalize(vmin=p2, vmax=p98)
cmap = matplotlib.colormaps['viridis']
ndvi_rgba = (cmap(norm(ndvi_crop)) * 255).astype(np.uint8)
ndvi_bgr  = cv2.cvtColor(ndvi_rgba[:, :, :3], cv2.COLOR_RGB2BGR)
out_ndvi = os.path.join(FIG_DIR, 'arch_ndvi.jpg')
cv2.imwrite(out_ndvi, ndvi_bgr, [cv2.IMWRITE_JPEG_QUALITY, 97])
print(f'arch_ndvi.jpg:  {ndvi_bgr.shape[1]}x{ndvi_bgr.shape[0]} px  (NDVI range {p2:.3f}–{p98:.3f})')

# ── 5. arch_pred.jpg — run model inference, draw boxes, crop same region ──────
from ultralytics import YOLO
model = YOLO(MODEL_CMX)
results = model.predict(IMG_PATH, conf=0.25, iou=0.6, imgsz=1280, verbose=False)

pred_vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
thick     = max(6, CROP_SIZE // 250)        # box line thickness
font_sc   = max(1.4, CROP_SIZE / 2000)
BLACK     = (0, 0, 0)

for box in results[0].boxes:
    cls      = int(box.cls[0])
    conf_val = float(box.conf[0])
    bx1, by1, bx2, by2 = [int(v) for v in box.xyxy[0].tolist()]
    color = CLASS_COLORS_RGB[cls]

    # Black shadow outline (thick+4) → class colour box on top
    cv2.rectangle(pred_vis, (bx1, by1), (bx2, by2), BLACK,  thick + 4)
    cv2.rectangle(pred_vis, (bx1, by1), (bx2, by2), color,  thick)

    label = f'{CLASS_NAMES[cls]} {conf_val:.2f}'
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_sc, 2)
    lx1, ly1 = bx1, max(by1 - th - 8, 0)
    lx2, ly2 = bx1 + tw + 8, max(by1, th + 8)
    cv2.rectangle(pred_vis, (lx1, ly1), (lx2, ly2), color,  -1)   # class-colour bg
    cv2.putText(pred_vis, label, (lx1 + 4, ly2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_sc, (0, 0, 0), 3)

pred_crop = pred_vis[y1c:y2c, x1c:x2c]
out_pred = os.path.join(FIG_DIR, 'arch_pred.jpg')
cv2.imwrite(out_pred, cv2.cvtColor(pred_crop, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 97])
n_boxes_in_crop = sum(
    1 for box in results[0].boxes
    if x1c <= (box.xyxy[0][0]+box.xyxy[0][2]).item()//2 <= x2c
    and y1c <= (box.xyxy[0][1]+box.xyxy[0][3]).item()//2 <= y2c
)
print(f'arch_pred.jpg:  {pred_crop.shape[1]}x{pred_crop.shape[0]} px  '
      f'({n_boxes_in_crop} boxes visible in crop, '
      f'{len(results[0].boxes)} total detected)')

# ── 6. Final check ────────────────────────────────────────────────────────────
for path in [out_rgnir, out_ndvi, out_pred]:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    size_kb = os.path.getsize(path) // 1024
    print(f'  {os.path.basename(path)}: {w}x{h} px  {size_kb} KB  square={w==h}')
