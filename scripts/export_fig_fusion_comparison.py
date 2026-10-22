#!/usr/bin/env python3
"""
export_fig_fusion_comparison.py — paper_figures/fig_fusion_comparison.(pdf|svg)
2 rows × 5 cols: [Sweetening, Net formation] × [GT | CM-FRM | AFF | ICAFusion | GAFF]
Scene selection: picks ROIs with highest TP variance across the 4 fusion operators.
Style/functions identical to fix_comparison_v8.py (Fig 6).
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_DIR = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics'
OUT_DIR  = os.path.join(REPO_DIR, 'paper_figures')
RAW_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
NDVI_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi'

MODEL_CMX  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
MODEL_AFF  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp17_AFF_Clean/weights/best.pt')
MODEL_ICA  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp18_ICAFusion/weights/best.pt')
MODEL_GAFF = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp19_GAFF/weights/best.pt')

TEST_IMG_DIR = os.path.join(RAW_DIR, 'images', 'test')
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

print("Checkpoints:")
for name, p in [('CM-FRM', MODEL_CMX), ('AFF', MODEL_AFF),
                ('ICAFusion', MODEL_ICA), ('GAFF', MODEL_GAFF)]:
    print(f"  {name}: {p}")
print()

CLASS_NAMES    = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
TARGET_CLASSES = [3, 2]   # row 0 = Sweetening, row 1 = Net formation
MIN_PURPLE = 85.0
PRED_RGB   = (255, 255, 0)
GT_RGB     = (255, 255, 255)
PRED_LW    = 5
GT_LW      = 2
IOU_T      = 0.50
DISP_H     = 520
TOP_K      = 15
MIN_GT     = 4
MIN_BLUR   = 50

# ── identical rcParams to fix_comparison_v8.py ──────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['CMU Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.weight':      'normal',
    'axes.titleweight': 'normal',
    'font.size':        13,
    'axes.titlesize':   14,
})


# ── helper functions (verbatim from fix_comparison_v8.py) ───────────────────
def check_purple(rgb_img, label=""):
    R = rgb_img[:,:,0].astype(float)
    G = rgb_img[:,:,1].astype(float)
    B = rgb_img[:,:,2].astype(float)
    pct = np.mean((R > G) & (B > G)) * 100
    status = "OK" if pct >= 85 else "BELOW TARGET"
    print(f"  purple-bg [{label}]: {pct:.1f}%  (target >=85%)  [{status}]")
    return pct

def draw_dashed(vis, x1, y1, x2, y2, color, lw=2, dash=14, gap=7):
    def dline(p1, p2):
        ax, ay = p1; bx, by = p2
        dist = max(int(np.hypot(bx-ax, by-ay)), 1)
        dx = (bx-ax)/dist; dy = (by-ay)/dist
        d = 0; on = True
        while d < dist:
            de = min(d + (dash if on else gap), dist)
            if on:
                cv2.line(vis, (int(ax+d*dx), int(ay+d*dy)),
                              (int(ax+de*dx), int(ay+de*dy)), color, lw, cv2.LINE_AA)
            d = de; on = not on
    dline((x1,y1),(x2,y1)); dline((x2,y1),(x2,y2))
    dline((x2,y2),(x1,y2)); dline((x1,y2),(x1,y1))

def corner_text(panel, text, pos='br', fs=0.65, lw=2):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, fs, lw)
    h, w = panel.shape[:2]; pad = 7
    if   pos == 'br': x, y = w - tw - pad*2, h - pad
    elif pos == 'bl': x, y = pad, h - pad
    elif pos == 'tr': x, y = w - tw - pad*2, th + pad
    else:             x, y = pad, th + pad
    ov = panel.copy()
    cv2.rectangle(ov, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (0,0,0), -1)
    cv2.addWeighted(ov, 0.65, panel, 0.35, 0, panel)
    cv2.putText(panel, text, (x, y), font, fs, (255,255,255), lw, cv2.LINE_AA)

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

def dense_cluster(cls_boxes, n=10):
    if len(cls_boxes) <= n: return cls_boxes
    mcx = np.mean([(b[1]+b[3])/2 for b in cls_boxes])
    mcy = np.mean([(b[2]+b[4])/2 for b in cls_boxes])
    return sorted(cls_boxes,
                  key=lambda b: ((b[1]+b[3])/2-mcx)**2 + ((b[2]+b[4])/2-mcy)**2)[:n]

def square_roi(cluster, W, H, pad_f=0.30, min_s=580):
    xs = [b[1] for b in cluster] + [b[3] for b in cluster]
    ys = [b[2] for b in cluster] + [b[4] for b in cluster]
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    pad  = max(int(span * pad_f), 80)
    side = max(span + 2*pad, min_s)
    cx = (min(xs)+max(xs))//2; cy = (min(ys)+max(ys))//2
    x1 = max(0, cx-side//2); x2 = min(W, x1+side)
    y1 = max(0, cy-side//2); y2 = min(H, y1+side)
    return x1, y1, x2, y2

def center_in(b, x1r, y1r, x2r, y2r, m=0):
    cx = (b[1]+b[3])//2; cy = (b[2]+b[4])//2
    return x1r-m <= cx <= x2r+m and y1r-m <= cy <= y2r+m

def overlaps(b, x1r, y1r, x2r, y2r):
    return b[1] < x2r and b[3] > x1r and b[2] < y2r and b[4] > y1r


# ── pre-load labels & blur ───────────────────────────────────────────────────
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


# ── load models ──────────────────────────────────────────────────────────────
print("Loading models ...")
from ultralytics import YOLO
m_cmx  = YOLO(MODEL_CMX)
m_aff  = YOLO(MODEL_AFF)
m_ica  = YOLO(MODEL_ICA)
m_gaff = YOLO(MODEL_GAFF)
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

def infer(model, ip, conf=0.25, iou=0.6):
    bgr, img4 = load_4ch(ip)
    if bgr is None: return []
    res = model.predict(source=img4, conf=conf, iou=iou, imgsz=1280, verbose=False)
    return [(int(b.cls[0]), *[int(v) for v in b.xyxy[0].tolist()], float(b.conf[0]))
            for b in res[0].boxes]


# ── scan & select best scene per class ──────────────────────────────────────
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

        pc   = infer(m_cmx,  ip)
        pa   = infer(m_aff,  ip)
        pi   = infer(m_ica,  ip)
        pg   = infer(m_gaff, ip)

        tp_c = count_tp(gt_in, pc,   target_cls)
        tp_a = count_tp(gt_in, pa,   target_cls)
        tp_i = count_tp(gt_in, pi,   target_cls)
        tp_g = count_tp(gt_in, pg,   target_cls)

        tps = [tp_c, tp_a, tp_i, tp_g]
        # Score: variance across operators × best TP (prefer hard scenes with spread)
        variance = float(np.var(tps))
        score = variance * max(tps) / max(len(gt_in), 1)

        print(f"    {Path(ip).name}: GT={len(gt_in)} "
              f"CMX={tp_c} AFF={tp_a} ICA={tp_i} GAFF={tp_g} "
              f"var={variance:.2f} score={score:.3f}")

        if score > best_score:
            best_score = score
            best = dict(ip=ip, roi=roi, gt_in=gt_in,
                        pc=pc, pa=pa, pi=pi, pg=pg,
                        tp_c=tp_c, tp_a=tp_a, tp_i=tp_i, tp_g=tp_g,
                        n_gt=len(gt_in))

    selected[target_cls] = best
    if best:
        print(f"  => {Path(best['ip']).name}  GT={best['n_gt']} "
              f"CMX={best['tp_c']} AFF={best['tp_a']} "
              f"ICA={best['tp_i']} GAFF={best['tp_g']}")
    else:
        print(f"  => No valid scene found!")

print()


# ── draw: 2 rows × 5 cols ───────────────────────────────────────────────────
COL_TITLES = [
    'Ground Truth',
    'Proposed\n(CM-FRM)',
    'AFF',
    'ICAFusion',
    'GAFF',
]

fig, axes = plt.subplots(2, 5, figsize=(9.5, 4.3))
fig.subplots_adjust(wspace=0.03, hspace=0.18, left=0.09, right=0.99, top=0.89, bottom=0.13)

for col, t in enumerate(COL_TITLES):
    axes[0][col].set_title(t, fontsize=13, pad=5)

print("\n[Purple% checks]")
tp_report = {}

for row_idx, target_cls in enumerate(TARGET_CLASSES):
    d = selected.get(target_cls)
    if d is None:
        for c in range(5):
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

    crop_bgr = bgr[y1r:y2r, x1r:x2r].copy()
    crop_boosted = np.clip(crop_bgr.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    crop_rgb = cv2.cvtColor(crop_boosted, cv2.COLOR_BGR2RGB)
    base_disp = cv2.resize(crop_rgb, (dw, DISP_H),
                           interpolation=cv2.INTER_AREA if sc <= 1 else cv2.INTER_CUBIC)

    check_purple(base_disp, CLASS_NAMES[target_cls])

    gt_disp   = [b for b in all_gt if b[0]==target_cls and overlaps(b,x1r,y1r,x2r,y2r)]
    cmx_disp  = [p for p in d['pc'] if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]
    aff_disp  = [p for p in d['pa'] if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]
    ica_disp  = [p for p in d['pi'] if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]
    gaff_disp = [p for p in d['pg'] if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]

    def to_d(boxes):
        out = []
        for b in boxes:
            nx1 = max(0, b[1]-x1r); ny1 = max(0, b[2]-y1r)
            nx2 = min(cw, b[3]-x1r); ny2 = min(ch, b[4]-y1r)
            if nx2 > nx1 and ny2 > ny1:
                out.append((b[0], int(nx1*sc), int(ny1*sc), int(nx2*sc), int(ny2*sc)))
        return out

    col_data = [
        (to_d(gt_disp),   None,              f"GT: {d['n_gt']}"),
        (None, to_d(cmx_disp),  f"TP: {d['tp_c']}/{d['n_gt']}"),
        (None, to_d(aff_disp),  f"TP: {d['tp_a']}/{d['n_gt']}"),
        (None, to_d(ica_disp),  f"TP: {d['tp_i']}/{d['n_gt']}"),
        (None, to_d(gaff_disp), f"TP: {d['tp_g']}/{d['n_gt']}"),
    ]

    scene_key = CLASS_NAMES[target_cls]
    tp_report[scene_key] = dict(
        image=Path(ip).name,
        n_gt=d['n_gt'],
        CMX=d['tp_c'], AFF=d['tp_a'], ICA=d['tp_i'], GAFF=d['tp_g']
    )

    for col_idx, (gbs, pbs, label) in enumerate(col_data):
        panel = base_disp.copy()

        if gbs:
            for _, x1, y1, x2, y2 in gbs:
                draw_dashed(panel, x1, y1, x2, y2, GT_RGB, lw=GT_LW)
        if pbs:
            for _, x1, y1, x2, y2 in pbs:
                cv2.rectangle(panel, (x1,y1), (x2,y2), PRED_RGB, PRED_LW)

        corner_text(panel, label, 'br')
        axes[row_idx][col_idx].imshow(panel)
        axes[row_idx][col_idx].set_xticks([]); axes[row_idx][col_idx].set_yticks([])

    axes[row_idx][0].set_ylabel(CLASS_NAMES[target_cls], fontsize=13,
                                rotation=90, va='center', labelpad=6)

# Legend (same as Fig 6)
handles = [
    mpatches.Patch(facecolor='none', edgecolor='white', linewidth=1.5,
                   linestyle=(0,(4,2)), label='Ground Truth (white dashed)'),
    mpatches.Patch(color='#FFFF00', label='Prediction (yellow solid)'),
]
fig.legend(handles=handles, loc='lower center', ncol=2, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, -0.02),
           handlelength=1.4, handleheight=0.8)


# ── save ────────────────────────────────────────────────────────────────────
print("\nSaving ...")
for ext in ['pdf', 'svg']:
    p = os.path.join(OUT_DIR, f'fig_fusion_comparison.{ext}')
    fig.savefig(p, format=ext, bbox_inches='tight', dpi=300)
    print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
plt.close(fig)

print("\n=== TP REPORT ===")
for scene, info in tp_report.items():
    print(f"\n  Scene: {scene}  ({info['image']})")
    print(f"    GT={info['n_gt']}  CM-FRM={info['CMX']}  AFF={info['AFF']}  "
          f"ICAFusion={info['ICA']}  GAFF={info['GAFF']}")

print("\nDone.")
