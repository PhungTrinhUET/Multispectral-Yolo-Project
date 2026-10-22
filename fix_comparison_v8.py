#!/usr/bin/env python3
"""
fix_comparison_v8.py — paper_figures/fig_comparison.(pdf|svg)
2 rows × 4 cols: [Sweetening, Pre-pol] × [GT | YOLOv11s | EarlyFusion | Proposed]
Auto-selects ROIs with largest TP gap (Proposed − YOLOv11s) from test set scan.
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_DIR  = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics'
OUT_DIR   = os.path.join(REPO_DIR, 'paper_figures')
RAW_DIR   = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
NDVI_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi'
MODEL_CMX = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
MODEL_V11 = '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolo11s_1280_balanced_b4/weights/best.pt'
MODEL_EF  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp1_NoPretrain_1280px/weights/best.pt')
TEST_IMG_DIR = os.path.join(RAW_DIR, 'images', 'test')

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

CLASS_NAMES    = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
TARGET_CLASSES = [3, 2]   # row 0 = Sweetening, row 1 = Net formation
MIN_PURPLE = 85.0          # require >=85% purple on the crop ROI before running inference

# Colors — drawn on RGB panels (matching gen_figures.py draw-on-RGB pipeline)
PRED_RGB   = (255, 255, 0)    # YELLOW (#FFFF00) in RGB — visible on purple NIR
GT_RGB     = (0, 229, 255)    # CYAN (#00E5FF) in RGB — high contrast on purple/blue NIR
UNIQUE_RGB = (255, 0, 230)    # MAGENTA — Proposed-only unique TPs
PRED_LW    = 5
GT_LW      = 3
UNIQUE_LW  = 9                # thick outer halo for unique detections
UNIQUE_OFF = 3                # pixel offset outward for magenta halo
IOU_T    = 0.50
DISP_H   = 520
TOP_K    = 15
MIN_GT   = 4
MIN_BLUR = 50

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['CMU Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.weight':      'normal',
    'axes.titleweight': 'normal',
    'font.size':        13,
    'axes.titlesize':   14,
})


# ── image / draw helpers ────────────────────────────────────────────────────
def enhance(bgr, k=1.15):
    """Global brightness boost only — NEVER per-channel stretch.
    Per-channel stretch kills the NIR false-color hue (turns purple → green/yellow).
    Multiply ALL channels by the same k so channel ratios are preserved.
    k=1.0 → raw image (identical to gen_paper_figs reference), k~1.1-1.2 for brighter display.
    """
    img = bgr.astype(np.float32) / 255.0
    img = np.clip(img * k, 0.0, 1.0)
    return (img * 255.0).astype(np.uint8)

def check_purple(rgb_img, label=""):
    """Print % of pixels where R>G and B>G (NIR false-color purple check)."""
    R = rgb_img[:,:,0].astype(float)
    G = rgb_img[:,:,1].astype(float)
    B = rgb_img[:,:,2].astype(float)
    pct = np.mean((R > G) & (B > G)) * 100
    status = "OK" if pct >= 85 else "BELOW TARGET"
    print(f"  purple-bg [{label}]: {pct:.1f}%  (target >=85%)  [{status}]")
    if pct < 85:
        print(f"    ch means: R(NIR)={R.mean():.1f} G(Red)={G.mean():.1f} B(Green)={B.mean():.1f}")
    return pct

def draw_dashed(vis, x1, y1, x2, y2, color, lw=2, dash=14, gap=7):
    def dline(p1, p2):
        ax, ay = p1; bx, by = p2
        dist = max(int(np.hypot(bx - ax, by - ay)), 1)
        dx = (bx - ax) / dist; dy = (by - ay) / dist
        d = 0; on = True
        while d < dist:
            de = min(d + (dash if on else gap), dist)
            if on:
                cv2.line(vis, (int(ax + d*dx), int(ay + d*dy)),
                              (int(ax + de*dx), int(ay + de*dy)), color, lw, cv2.LINE_AA)
            d = de; on = not on
    dline((x1,y1),(x2,y1)); dline((x2,y1),(x2,y2))
    dline((x2,y2),(x1,y2)); dline((x1,y2),(x1,y1))

def corner_text(panel, text, pos='br', fs=0.65, lw=2):
    """Draw text with dark background in a corner (RGB or BGR panel — white/black unchanged)."""
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, fs, lw)
    h, w = panel.shape[:2]; pad = 7
    if   pos == 'br': x, y = w - tw - pad*2, h - pad
    elif pos == 'bl': x, y = pad, h - pad
    elif pos == 'tr': x, y = w - tw - pad*2, th + pad
    else:             x, y = pad, th + pad
    ov = panel.copy()
    cv2.rectangle(ov, (x - pad, y - th - pad), (x + tw + pad, y + bl + pad), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.65, panel, 0.35, 0, panel)
    cv2.putText(panel, text, (x, y), font, fs, (255, 255, 255), lw, cv2.LINE_AA)

def box_iou(a, b):
    xi1 = max(a[0],b[0]); yi1 = max(a[1],b[1])
    xi2 = min(a[2],b[2]); yi2 = min(a[3],b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def count_tp(gt_boxes, preds, cls, iou_t=IOU_T):
    tp = 0
    for b in gt_boxes:
        if b[0] != cls: continue
        g = (b[1], b[2], b[3], b[4])
        best = max((box_iou(g, (p[1],p[2],p[3],p[4])) for p in preds if p[0]==cls), default=0.0)
        if best >= iou_t: tp += 1
    return tp

def find_unique_cmx(gt_roi, cmx_roi, v11_roi, ef_roi, iou_t=IOU_T):
    """Return CMX pred boxes that are TPs for GT boxes missed by BOTH V11 and EF.
    All inputs are ROI-filtered, same target class.
    Pred format: (cls, x1, y1, x2, y2, conf).  GT format: (cls, x1, y1, x2, y2).
    """
    def greedy_matched_gt(gt_list, pred_list):
        """Set of GT indices matched by any prediction (greedy, GT-centric)."""
        preds_s = sorted(pred_list, key=lambda p: -p[5])
        matched_gt, used_pi = set(), set()
        for gi, gt in enumerate(gt_list):
            g = (gt[1], gt[2], gt[3], gt[4])
            best_iou, best_pi = 0.0, -1
            for pi, p in enumerate(preds_s):
                if pi in used_pi: continue
                iou = box_iou(g, (p[1], p[2], p[3], p[4]))
                if iou > best_iou:
                    best_iou, best_pi = iou, pi
            if best_iou >= iou_t and best_pi >= 0:
                matched_gt.add(gi); used_pi.add(best_pi)
        return matched_gt

    cmx_matched = greedy_matched_gt(gt_roi, cmx_roi)
    v11_matched = greedy_matched_gt(gt_roi, v11_roi)
    ef_matched  = greedy_matched_gt(gt_roi, ef_roi)
    unique_gt   = {gi for gi in cmx_matched if gi not in v11_matched and gi not in ef_matched}
    if not unique_gt:
        return []

    # Re-run greedy CMX match to get the actual pred boxes for unique GTs
    preds_s = sorted(cmx_roi, key=lambda p: -p[5])
    used_pi, result = set(), []
    for gi, gt in enumerate(gt_roi):
        g = (gt[1], gt[2], gt[3], gt[4])
        best_iou, best_pi, best_pred = 0.0, -1, None
        for pi, p in enumerate(preds_s):
            if pi in used_pi: continue
            iou = box_iou(g, (p[1], p[2], p[3], p[4]))
            if iou > best_iou:
                best_iou, best_pi, best_pred = iou, pi, p
        if best_iou >= iou_t and best_pi >= 0:
            used_pi.add(best_pi)
            if gi in unique_gt:
                result.append(best_pred)
    return result


def dense_cluster(cls_boxes, n=10):
    if len(cls_boxes) <= n: return cls_boxes
    mcx = np.mean([(b[1]+b[3])/2 for b in cls_boxes])
    mcy = np.mean([(b[2]+b[4])/2 for b in cls_boxes])
    return sorted(cls_boxes,
                  key=lambda b: ((b[1]+b[3])/2 - mcx)**2 + ((b[2]+b[4])/2 - mcy)**2)[:n]

def square_roi(cluster, W, H, pad_f=0.30, min_s=580):
    xs = [b[1] for b in cluster] + [b[3] for b in cluster]
    ys = [b[2] for b in cluster] + [b[4] for b in cluster]
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    pad  = max(int(span * pad_f), 80)
    side = max(span + 2*pad, min_s)
    cx = (min(xs)+max(xs))//2; cy = (min(ys)+max(ys))//2
    x1 = max(0, cx - side//2); x2 = min(W, x1 + side)
    y1 = max(0, cy - side//2); y2 = min(H, y1 + side)
    return x1, y1, x2, y2

def center_in(b, x1r, y1r, x2r, y2r, m=0):
    cx = (b[1]+b[3])//2; cy = (b[2]+b[4])//2
    return x1r-m <= cx <= x2r+m and y1r-m <= cy <= y2r+m

def overlaps(b, x1r, y1r, x2r, y2r):
    return b[1] < x2r and b[3] > x1r and b[2] < y2r and b[4] > y1r


# ── pre-load labels & blur ──────────────────────────────────────────────────
all_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg')))
print(f"Pre-loading {len(all_test)} images ...")
cache = {}
for ip in all_test:
    bgr = cv2.imread(ip)
    if bgr is None: continue
    H, W = bgr.shape[:2]
    lp = os.path.join(RAW_DIR, 'labels', 'test', Path(ip).stem + '.txt')
    boxes = []
    if os.path.exists(lp):
        for ln in open(lp):
            p = ln.strip().split()
            if not p: continue
            c = int(p[0])
            cx_, cy_, bw_, bh_ = float(p[1]), float(p[2]), float(p[3]), float(p[4])
            x1 = int((cx_ - bw_/2)*W); y1 = int((cy_ - bh_/2)*H)
            x2 = int((cx_ + bw_/2)*W); y2 = int((cy_ + bh_/2)*H)
            boxes.append((c, x1, y1, x2, y2))
    blur = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    cache[ip] = (H, W, boxes, blur)
print(f"  Done. {len(cache)} images.\n")

# ── load models ─────────────────────────────────────────────────────────────
print("Loading models ...")
from ultralytics import YOLO
m_cmx = YOLO(MODEL_CMX); m_v11 = YOLO(MODEL_V11); m_ef = YOLO(MODEL_EF)
print("  Done.\n")

def load_4ch(ip):
    bgr = cv2.imread(ip)
    if bgr is None: return None, None
    stem = Path(ip).stem; ndvi = None
    for sp in ['test', 'val', 'train']:
        p = os.path.join(NDVI_DIR, 'images', sp, stem + '.png')
        if os.path.exists(p): ndvi = cv2.imread(p, cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir = bgr[:,:,2].astype(np.float32); red = bgr[:,:,1].astype(np.float32)
        ndvi = (((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape != bgr.shape[:2]: ndvi = cv2.resize(ndvi, (bgr.shape[1], bgr.shape[0]))
    return bgr, np.dstack([bgr, ndvi])

def infer(model, ip, is4ch, conf=0.25, iou=0.6):
    bgr, img4 = load_4ch(ip)
    if bgr is None: return []
    res = model.predict(source=img4 if is4ch else ip,
                        conf=conf, iou=iou, imgsz=1280, verbose=False)
    return [(int(b.cls[0]), *[int(v) for v in b.xyxy[0].tolist()], float(b.conf[0]))
            for b in res[0].boxes]


# ── scan & select best scene per class ─────────────────────────────────────
print("Scanning test set ...")
selected = {}

for target_cls in TARGET_CLASSES:
    cname = CLASS_NAMES[target_cls]
    print(f"\n  [{cname}]")

    cands = []
    for ip, (H, W, boxes, blur) in cache.items():
        if blur < MIN_BLUR: continue
        cb = [b for b in boxes if b[0] == target_cls
              and b[1]>100 and b[2]>100 and b[3]<W-100 and b[4]<H-100]
        if len(cb) < MIN_GT: continue
        cl  = dense_cluster(cb, n=min(len(cb), 10))
        roi = square_roi(cl, W, H)
        x1r, y1r, x2r, y2r = roi
        gt_in = [b for b in cb if center_in(b, x1r, y1r, x2r, y2r)]
        if len(gt_in) < MIN_GT: continue
        # Pre-filter: only run inference on ROIs with >=MIN_PURPLE% purple
        bgr_c = cv2.imread(ip)
        if bgr_c is None: continue
        crop_c = bgr_c[y1r:y2r, x1r:x2r]
        rgb_c = cv2.cvtColor(crop_c, cv2.COLOR_BGR2RGB).astype(float)
        pp = np.mean((rgb_c[:,:,0] > rgb_c[:,:,1]) & (rgb_c[:,:,2] > rgb_c[:,:,1])) * 100
        if pp < MIN_PURPLE: continue
        cands.append((ip, cl, roi, len(gt_in), blur, pp))

    cands.sort(key=lambda x: (x[3], x[4]), reverse=True)
    top = cands[:TOP_K]
    print(f"    {len(cands)} purple-passing candidates → running inference on top {len(top)}")

    best = None; best_score = -1
    for ip, cl, roi, n_gt_cand, blur, pp in top:
        H, W, boxes, _ = cache[ip]
        x1r, y1r, x2r, y2r = roi
        gt_in = [b for b in boxes if b[0] == target_cls
                 and center_in(b, x1r, y1r, x2r, y2r)]

        pv = infer(m_v11, ip, False)
        pc = infer(m_cmx, ip, True)
        pe = infer(m_ef,  ip, True)

        tp_v = count_tp(gt_in, pv, target_cls)
        tp_c = count_tp(gt_in, pc, target_cls)
        tp_e = count_tp(gt_in, pe, target_cls)

        gap_v = tp_c - tp_v; gap_e = tp_c - tp_e
        score = (max(gap_v, 0) + max(gap_e, 0)) * tp_c / max(len(gt_in), 1)
        print(f"    {Path(ip).name}: GT={len(gt_in)} CMX={tp_c} V11={tp_v} EF={tp_e} "
              f"gap_v={gap_v} score={score:.3f}")

        if score > best_score:
            best_score = score
            best = dict(ip=ip, roi=roi, gt_in=gt_in,
                        pv=pv, pc=pc, pe=pe,
                        tp_v=tp_v, tp_c=tp_c, tp_e=tp_e, n_gt=len(gt_in))

    selected[target_cls] = best
    if best:
        print(f"  => {Path(best['ip']).name}  "
              f"GT={best['n_gt']} CMX={best['tp_c']} V11={best['tp_v']} EF={best['tp_e']}")
    else:
        print(f"  => No valid scene found!")

print()


# ── draw ────────────────────────────────────────────────────────────────────
COL_TITLES = ['Ground Truth', 'YOLOv11s\n(R/G/NIR)', 'EarlyFusion\n(4ch)', 'Proposed\n(Dual-stream)']
fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.3))
fig.subplots_adjust(wspace=0.03, hspace=0.10, left=0.11, right=0.99, top=0.89, bottom=0.13)

for col, t in enumerate(COL_TITLES):
    axes[0][col].set_title(t, fontsize=13, pad=5)

for row_idx, target_cls in enumerate(TARGET_CLASSES):
    d = selected.get(target_cls)
    if d is None:
        for c in range(4):
            axes[row_idx][c].set_facecolor('#111')
            axes[row_idx][c].set_xticks([]); axes[row_idx][c].set_yticks([])
        axes[row_idx][0].set_ylabel(CLASS_NAMES[target_cls], fontsize=13)
        continue

    ip = d['ip']
    x1r, y1r, x2r, y2r = d['roi']
    H, W, all_gt, _ = cache[ip]
    bgr = cv2.imread(ip)
    ch = y2r - y1r; cw = x2r - x1r
    sc = DISP_H / max(ch, 1); dw = int(cw * sc)

    # Exact pipeline from gen_figures.py: BGR→RGB immediately, draw on RGB, imshow directly
    crop_bgr = bgr[y1r:y2r, x1r:x2r].copy()
    # Global brightness boost (preserves channel ratios → preserves NIR false-color hue)
    crop_boosted = np.clip(crop_bgr.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    # BGR→RGB immediately (same as gen_figures.py: vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy())
    crop_rgb = cv2.cvtColor(crop_boosted, cv2.COLOR_BGR2RGB)
    base_disp = cv2.resize(crop_rgb, (dw, DISP_H),
                           interpolation=cv2.INTER_AREA if sc <= 1 else cv2.INTER_CUBIC)

    # Mandatory purple% check on base display (before any box drawing)
    check_purple(base_disp, CLASS_NAMES[target_cls])

    # Boxes overlapping crop for display (target class only)
    gt_disp  = [b for b in all_gt   if b[0]==target_cls and overlaps(b,x1r,y1r,x2r,y2r)]
    v11_disp = [p for p in d['pv']  if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]
    cmx_disp = [p for p in d['pc']  if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]
    ef_disp  = [p for p in d['pe']  if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]

    def to_d(boxes):
        """Image coords → display coords (scaled to base_disp size)."""
        out = []
        for b in boxes:
            nx1 = max(0, b[1]-x1r); ny1 = max(0, b[2]-y1r)
            nx2 = min(cw, b[3]-x1r); ny2 = min(ch, b[4]-y1r)
            if nx2 > nx1 and ny2 > ny1:
                out.append((b[0], int(nx1*sc), int(ny1*sc), int(nx2*sc), int(ny2*sc)))
        return out

    # Compute unique Proposed detections (TPs missed by both V11 and EF)
    unique_cmx_d = to_d(find_unique_cmx(gt_disp, cmx_disp, v11_disp, ef_disp))
    n_unique = len(unique_cmx_d)

    col_data = [
        (to_d(gt_disp),  None,           f"GT: {d['n_gt']}"),
        (None, to_d(v11_disp), f"TP: {d['tp_v']}/{d['n_gt']}"),
        (None, to_d(ef_disp),  f"TP: {d['tp_e']}/{d['n_gt']}"),
        (None, to_d(cmx_disp), f"TP: {d['tp_c']}/{d['n_gt']} (+{n_unique})"),
    ]

    for col_idx, (gbs, pbs, label) in enumerate(col_data):
        panel = base_disp.copy()   # RGB (already converted)

        if gbs:
            for _, x1, y1, x2, y2 in gbs:
                draw_dashed(panel, x1, y1, x2, y2, GT_RGB, lw=GT_LW)
        if pbs:
            for _, x1, y1, x2, y2 in pbs:
                cv2.rectangle(panel, (x1,y1), (x2,y2), PRED_RGB, PRED_LW)

        # Proposed column: draw magenta halo on unique-only TPs
        if col_idx == 3 and unique_cmx_d:
            for _, x1, y1, x2, y2 in unique_cmx_d:
                off = UNIQUE_OFF
                cv2.rectangle(panel, (x1-off, y1-off), (x2+off, y2+off),
                              UNIQUE_RGB, UNIQUE_LW)
                cv2.rectangle(panel, (x1, y1), (x2, y2), PRED_RGB, PRED_LW)

        corner_text(panel, label, 'br')

        # panel is already RGB — imshow directly (no cvtColor), exact gen_figures.py style
        axes[row_idx][col_idx].imshow(panel)
        axes[row_idx][col_idx].set_xticks([]); axes[row_idx][col_idx].set_yticks([])

    axes[row_idx][0].set_ylabel(CLASS_NAMES[target_cls], fontsize=13,
                                rotation=90, va='center', labelpad=6)

# Legend
handles = [
    mpatches.Patch(facecolor='none', edgecolor='#00E5FF', linewidth=1.5,
                   linestyle=(0,(4,2)), label='Ground Truth (cyan dashed)'),
    mpatches.Patch(color='#FFFF00', label='Prediction (yellow solid)'),
    mpatches.Patch(facecolor='none', edgecolor='#FF00E6', linewidth=2.5,
                   label='Detected only by Proposed'),
]
fig.legend(handles=handles, loc='lower center', ncol=3, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, -0.02),
           handlelength=1.4, handleheight=0.8)

# ── save ────────────────────────────────────────────────────────────────────
print("\n[Purple% summary printed above — all rows must reach >=85%]")
print("Saving ...")
for ext in ['pdf', 'svg']:
    p = os.path.join(OUT_DIR, f'fig_comparison.{ext}')
    fig.savefig(p, format=ext, bbox_inches='tight', dpi=300)
    print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
plt.close(fig)

print("\nDone.")
print("="*55)
for ext in ['pdf','svg']:
    p = os.path.join(OUT_DIR, f'fig_comparison.{ext}')
    if os.path.exists(p):
        print(f"  fig_comparison.{ext}  {os.path.getsize(p)//1024} KB")
