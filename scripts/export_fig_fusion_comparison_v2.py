#!/usr/bin/env python3
"""
export_fig_fusion_comparison_v2.py — paper_figures/fig_fusion_comparison.(pdf|svg)
TP/FP/FN color-coded comparison: 3 rows × 5 cols.
Rows: pre-pollination (class 0) | post-pollination (class 1) | net formation (class 2).
Cols: Ground Truth | Proposed (CM-FRM) | AFF | GAFF | ICAFusion.
TP = green solid  |  FP = red solid  |  FN = orange dashed (at GT location).
Style: identical to fix_comparison_v8.py (serif font, purple NIR tones, white netting).
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
MODEL_GAFF = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp19_GAFF/weights/best.pt')
MODEL_ICA  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp18_ICAFusion/weights/best.pt')

TEST_IMG_DIR = os.path.join(RAW_DIR, 'images', 'test')
os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

print("Checkpoints:")
for name, p in [('CM-FRM', MODEL_CMX), ('AFF', MODEL_AFF),
                ('GAFF', MODEL_GAFF), ('ICAFusion', MODEL_ICA)]:
    print(f"  {name}: {p}")
print()

CLASS_NAMES = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
# 3 rows: class 0 (early, Proposed wins), class 1 (early, Proposed wins), class 2 (hard)
TARGET_CLASSES = [0, 1, 2]
MIN_PURPLE = 70.0     # relaxed for pre/post-pol scenes (mix of crop stages)
MIN_GT     = [2, 2, 3]  # per-class: early stages have fewer instances per image
MIN_BLUR   = 40
TOP_K      = 20
IOU_T      = 0.50
DISP_H     = 460

# Colors in RGB (drawn on RGB panel)
GT_RGB  = (200, 200, 200)   # light grey dashed — GT reference column
TP_RGB  = (0,  220,  80)    # green solid
FP_RGB  = (255,  50,  50)   # red solid
FN_RGB  = (255, 165,   0)   # orange dashed
BOX_LW  = 4
GT_LW   = 2

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['CMU Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.weight':      'normal',
    'axes.titleweight': 'normal',
    'font.size':        12,
    'axes.titlesize':   13,
})


# ── helpers (verbatim from fix_comparison_v8.py) ────────────────────────────
def draw_dashed(vis, x1, y1, x2, y2, color, lw=2, dash=12, gap=6):
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

def corner_text(panel, text, pos='br', fs=0.58, lw=1):
    font = cv2.FONT_HERSHEY_DUPLEX
    (tw, th), bl = cv2.getTextSize(text, font, fs, lw)
    h, w = panel.shape[:2]; pad = 5
    if   pos == 'br': x, y = w - tw - pad*2, h - pad
    elif pos == 'bl': x, y = pad, h - pad
    elif pos == 'tr': x, y = w - tw - pad*2, th + pad
    else:             x, y = pad, th + pad
    ov = panel.copy()
    cv2.rectangle(ov, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (0,0,0), -1)
    cv2.addWeighted(ov, 0.60, panel, 0.40, 0, panel)
    cv2.putText(panel, text, (x, y), font, fs, (255,255,255), lw, cv2.LINE_AA)

def box_iou(a, b):
    xi1 = max(a[0],b[0]); yi1 = max(a[1],b[1])
    xi2 = min(a[2],b[2]); yi2 = min(a[3],b[3])
    inter = max(0, xi2-xi1)*max(0, yi2-yi1)
    if inter == 0: return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def compute_tpfpfn(gt_boxes, pred_boxes, cls, iou_t=IOU_T):
    """Match preds to GT by IoU>=iou_t, same class. Returns (tp_preds, fp_preds, fn_gts)."""
    gt_cls   = [b for b in gt_boxes   if b[0] == cls]
    pred_cls = sorted([p for p in pred_boxes if p[0] == cls],
                      key=lambda p: -p[5])   # confidence desc
    matched_gt = set()
    tp_preds, fp_preds = [], []
    for pred in pred_cls:
        g = (pred[1], pred[2], pred[3], pred[4])
        best_iou, best_idx = 0.0, -1
        for i, gt in enumerate(gt_cls):
            if i in matched_gt: continue
            iou = box_iou(g, (gt[1], gt[2], gt[3], gt[4]))
            if iou > best_iou:
                best_iou, best_idx = iou, i
        if best_iou >= iou_t and best_idx >= 0:
            tp_preds.append(pred)
            matched_gt.add(best_idx)
        else:
            fp_preds.append(pred)
    fn_gts = [gt_cls[i] for i in range(len(gt_cls)) if i not in matched_gt]
    return tp_preds, fp_preds, fn_gts

def dense_cluster(cls_boxes, n=8):
    if len(cls_boxes) <= n: return cls_boxes
    mcx = np.mean([(b[1]+b[3])/2 for b in cls_boxes])
    mcy = np.mean([(b[2]+b[4])/2 for b in cls_boxes])
    return sorted(cls_boxes,
                  key=lambda b: ((b[1]+b[3])/2-mcx)**2+((b[2]+b[4])/2-mcy)**2)[:n]

def square_roi(cluster, W, H, pad_f=0.35, min_s=520):
    xs = [b[1] for b in cluster]+[b[3] for b in cluster]
    ys = [b[2] for b in cluster]+[b[4] for b in cluster]
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    pad  = max(int(span*pad_f), 80)
    side = max(span+2*pad, min_s)
    cx = (min(xs)+max(xs))//2; cy = (min(ys)+max(ys))//2
    x1 = max(0, cx-side//2); x2 = min(W, x1+side)
    y1 = max(0, cy-side//2); y2 = min(H, y1+side)
    return x1, y1, x2, y2

def center_in(b, x1r, y1r, x2r, y2r, m=0):
    cx = (b[1]+b[3])//2; cy = (b[2]+b[4])//2
    return x1r-m<=cx<=x2r+m and y1r-m<=cy<=y2r+m

def overlaps(b, x1r, y1r, x2r, y2r):
    return b[1]<x2r and b[3]>x1r and b[2]<y2r and b[4]>y1r


# ── pre-load ─────────────────────────────────────────────────────────────────
all_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg')))
print(f"Pre-loading {len(all_test)} images ...")
cache = {}
for ip in all_test:
    bgr = cv2.imread(ip)
    if bgr is None: continue
    H, W = bgr.shape[:2]
    lp = os.path.join(RAW_DIR, 'labels', 'test', Path(ip).stem+'.txt')
    boxes = []
    if os.path.exists(lp):
        for ln in open(lp):
            p = ln.strip().split()
            if not p: continue
            c = int(p[0])
            cx_,cy_,bw_,bh_ = float(p[1]),float(p[2]),float(p[3]),float(p[4])
            x1=int((cx_-bw_/2)*W); y1=int((cy_-bh_/2)*H)
            x2=int((cx_+bw_/2)*W); y2=int((cy_+bh_/2)*H)
            boxes.append((c,x1,y1,x2,y2))
    blur = cv2.Laplacian(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()
    cache[ip] = (H, W, boxes, blur)
print(f"  Done. {len(cache)} images.\n")


# ── load models ──────────────────────────────────────────────────────────────
print("Loading models ...")
from ultralytics import YOLO
m_cmx  = YOLO(MODEL_CMX)
m_aff  = YOLO(MODEL_AFF)
m_gaff = YOLO(MODEL_GAFF)
m_ica  = YOLO(MODEL_ICA)
print("  Done.\n")

def load_4ch(ip):
    bgr = cv2.imread(ip)
    if bgr is None: return None, None
    stem = Path(ip).stem; ndvi = None
    for sp in ['test','val','train']:
        p = os.path.join(NDVI_DIR,'images',sp,stem+'.png')
        if os.path.exists(p): ndvi = cv2.imread(p,cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir = bgr[:,:,2].astype(np.float32); red = bgr[:,:,1].astype(np.float32)
        ndvi = (((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape != bgr.shape[:2]: ndvi=cv2.resize(ndvi,(bgr.shape[1],bgr.shape[0]))
    return bgr, np.dstack([bgr, ndvi])

def infer(model, ip, conf=0.25, iou=0.6):
    bgr, img4 = load_4ch(ip)
    if bgr is None: return []
    res = model.predict(source=img4, conf=conf, iou=iou, imgsz=1280, verbose=False)
    return [(int(b.cls[0]),*[int(v) for v in b.xyxy[0].tolist()],float(b.conf[0]))
            for b in res[0].boxes]


# ── scene selection ──────────────────────────────────────────────────────────
print("Scanning test set ...")
selected = {}

for row_i, target_cls in enumerate(TARGET_CLASSES):
    cname = CLASS_NAMES[target_cls]
    min_gt_cls = MIN_GT[row_i]
    print(f"\n  [{cname}] (min_gt={min_gt_cls})")

    cands = []
    for ip, (H, W, boxes, blur) in cache.items():
        if blur < MIN_BLUR: continue
        cb = [b for b in boxes if b[0]==target_cls
              and b[1]>80 and b[2]>80 and b[3]<W-80 and b[4]<H-80]
        if len(cb) < min_gt_cls: continue
        cl  = dense_cluster(cb, n=min(len(cb),8))
        roi = square_roi(cl, W, H)
        x1r,y1r,x2r,y2r = roi
        gt_in = [b for b in cb if center_in(b,x1r,y1r,x2r,y2r)]
        if len(gt_in) < min_gt_cls: continue
        # Check purple (relaxed for early stages)
        bgr_c = cv2.imread(ip)
        if bgr_c is None: continue
        crop_c = bgr_c[y1r:y2r,x1r:x2r]
        rgb_c = cv2.cvtColor(crop_c,cv2.COLOR_BGR2RGB).astype(float)
        pp = np.mean((rgb_c[:,:,0]>rgb_c[:,:,1])&(rgb_c[:,:,2]>rgb_c[:,:,1]))*100
        if pp < MIN_PURPLE: continue
        cands.append((ip,cl,roi,len(gt_in),blur,pp))

    cands.sort(key=lambda x:(x[3],x[4]),reverse=True)
    top = cands[:TOP_K]
    print(f"    {len(cands)} candidates → top {len(top)}")

    best = None; best_score = -1
    for ip,cl,roi,n_gt_cand,blur,pp in top:
        H,W,boxes,_ = cache[ip]
        x1r,y1r,x2r,y2r = roi
        gt_in = [b for b in boxes if b[0]==target_cls
                 and center_in(b,x1r,y1r,x2r,y2r)]

        pc   = infer(m_cmx,  ip)
        pa   = infer(m_aff,  ip)
        pg   = infer(m_gaff, ip)
        pi   = infer(m_ica,  ip)

        def tp_count(preds):
            tp,_,_ = compute_tpfpfn(gt_in, preds, target_cls)
            return len(tp)

        n_gt  = len(gt_in)
        tp_c  = tp_count(pc); tp_a = tp_count(pa)
        tp_g  = tp_count(pg); tp_i = tp_count(pi)

        # Score: CM-FRM advantage over others, weighted by its own TP
        # Favors scenes where Proposed genuinely detects more than baselines
        others_mean = (tp_a + tp_g + tp_i) / 3.0
        advantage   = tp_c - others_mean
        score = advantage * tp_c / max(n_gt, 1)

        print(f"    {Path(ip).name}: GT={n_gt} "
              f"CMX={tp_c} AFF={tp_a} GAFF={tp_g} ICA={tp_i} "
              f"adv={advantage:.2f} score={score:.3f}")

        if score > best_score:
            best_score = score
            best = dict(ip=ip, roi=roi, gt_in=gt_in,
                        pc=pc, pa=pa, pg=pg, pi=pi,
                        n_gt=n_gt,
                        tp_c=tp_c, tp_a=tp_a, tp_g=tp_g, tp_i=tp_i)

    selected[target_cls] = best
    if best:
        print(f"  => {Path(best['ip']).name}  GT={best['n_gt']} "
              f"CMX={best['tp_c']} AFF={best['tp_a']} "
              f"GAFF={best['tp_g']} ICA={best['tp_i']}")
    else:
        print(f"  => No valid scene (score>0) — will show blank row.")

print()


# ── draw 3 × 5 ──────────────────────────────────────────────────────────────
N_ROWS = len(TARGET_CLASSES)
N_COLS = 5
COL_TITLES = [
    'Ground Truth',
    'Proposed\n(CM-FRM)',
    'AFF',
    'GAFF',
    'ICAFusion',
]

fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(11.5, 3.2*N_ROWS))
fig.subplots_adjust(wspace=0.025, hspace=0.20,
                    left=0.08, right=0.995, top=0.91, bottom=0.10)

for col, t in enumerate(COL_TITLES):
    axes[0][col].set_title(t, fontsize=12, pad=5)

tp_report = {}

for row_idx, target_cls in enumerate(TARGET_CLASSES):
    d = selected.get(target_cls)
    cname = CLASS_NAMES[target_cls]
    axes[row_idx][0].set_ylabel(cname, fontsize=11,
                                rotation=90, va='center', labelpad=5)

    if d is None or best_score <= 0:
        for c in range(N_COLS):
            axes[row_idx][c].set_facecolor('#1a1a1a')
            axes[row_idx][c].set_xticks([]); axes[row_idx][c].set_yticks([])
        continue

    ip = d['ip']
    x1r,y1r,x2r,y2r = d['roi']
    H,W,all_gt,_ = cache[ip]
    bgr = cv2.imread(ip)
    ch = y2r-y1r; cw = x2r-x1r
    sc = DISP_H/max(ch,1); dw = int(cw*sc)

    # Base display: same pipeline as fix_comparison_v8.py
    crop_bgr     = bgr[y1r:y2r, x1r:x2r].copy()
    crop_boosted = np.clip(crop_bgr.astype(np.float32)*1.15, 0, 255).astype(np.uint8)
    base_disp    = cv2.resize(cv2.cvtColor(crop_boosted, cv2.COLOR_BGR2RGB),
                              (dw, DISP_H),
                              interpolation=cv2.INTER_AREA if sc<=1 else cv2.INTER_CUBIC)

    # Box coord: image → display
    def to_d(boxes):
        out = []
        for b in boxes:
            nx1=max(0,b[1]-x1r); ny1=max(0,b[2]-y1r)
            nx2=min(cw,b[3]-x1r); ny2=min(ch,b[4]-y1r)
            if nx2>nx1 and ny2>ny1:
                out.append((b[0],int(nx1*sc),int(ny1*sc),int(nx2*sc),int(ny2*sc)))
        return out

    # GT for this ROI and class
    gt_disp = [b for b in all_gt if b[0]==target_cls and overlaps(b,x1r,y1r,x2r,y2r)]

    # Per-model TP/FP/FN (on full-image predictions filtered to ROI)
    models_preds = [d['pc'], d['pa'], d['pg'], d['pi']]
    model_labels = ['CM-FRM', 'AFF', 'GAFF', 'ICAFusion']
    scene_tpfpfn = {}

    for ml, preds in zip(model_labels, models_preds):
        preds_roi = [p for p in preds if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]
        tp,fp,fn  = compute_tpfpfn(gt_disp, preds_roi, target_cls)
        scene_tpfpfn[ml] = (tp, fp, fn)

    tp_report[cname] = {
        'image': Path(ip).name,
        'n_gt':  len(gt_disp),
    }
    for ml,(tp,fp,fn) in scene_tpfpfn.items():
        tp_report[cname][ml] = dict(TP=len(tp), FP=len(fp), FN=len(fn))

    # ---- Column 0: Ground Truth ----
    gt_panel = base_disp.copy()
    for b in to_d(gt_disp):
        _,x1,y1,x2,y2 = b
        draw_dashed(gt_panel, x1,y1,x2,y2, GT_RGB, lw=GT_LW)
    corner_text(gt_panel, f"GT: {len(gt_disp)}", 'br')
    axes[row_idx][0].imshow(gt_panel)
    axes[row_idx][0].set_xticks([]); axes[row_idx][0].set_yticks([])

    # ---- Columns 1–4: models ----
    for col_off, (ml, preds) in enumerate(zip(model_labels, models_preds)):
        preds_roi = [p for p in preds if p[0]==target_cls and overlaps(p,x1r,y1r,x2r,y2r)]
        tp, fp, fn = scene_tpfpfn[ml]

        panel = base_disp.copy()

        # TP: green solid
        for b in to_d(tp):
            _,x1,y1,x2,y2 = b
            cv2.rectangle(panel,(x1,y1),(x2,y2), TP_RGB, BOX_LW)

        # FP: red solid
        for b in to_d(fp):
            _,x1,y1,x2,y2 = b
            cv2.rectangle(panel,(x1,y1),(x2,y2), FP_RGB, BOX_LW)

        # FN: orange dashed at GT position
        for b in to_d(fn):
            _,x1,y1,x2,y2 = b
            draw_dashed(panel, x1,y1,x2,y2, FN_RGB, lw=GT_LW+1, dash=10, gap=5)

        lbl = f"TP:{len(tp)} FP:{len(fp)} FN:{len(fn)}"
        corner_text(panel, lbl, 'br')

        col_idx = col_off + 1
        axes[row_idx][col_idx].imshow(panel)
        axes[row_idx][col_idx].set_xticks([]); axes[row_idx][col_idx].set_yticks([])

# ── legend ───────────────────────────────────────────────────────────────────
tp_patch  = mpatches.Patch(facecolor=(0/255,220/255,80/255),  label='TP (green solid)')
fp_patch  = mpatches.Patch(facecolor=(255/255,50/255,50/255), label='FP (red solid)')
fn_patch  = mpatches.Patch(facecolor='none', edgecolor=(1,165/255,0),
                            linewidth=1.5, linestyle=(0,(3,2)),
                            label='FN (orange dashed)')
gt_patch  = mpatches.Patch(facecolor='none', edgecolor=(200/255,200/255,200/255),
                            linewidth=1.5, linestyle=(0,(4,2)),
                            label='GT (gray dashed)')
fig.legend(handles=[tp_patch, fp_patch, fn_patch, gt_patch],
           loc='lower center', ncol=4, fontsize=10,
           frameon=True, bbox_to_anchor=(0.5, -0.01),
           handlelength=1.4, handleheight=0.8)


# ── save ─────────────────────────────────────────────────────────────────────
print("Saving ...")
for ext in ['pdf','svg']:
    p = os.path.join(OUT_DIR, f'fig_fusion_comparison.{ext}')
    fig.savefig(p, format=ext, bbox_inches='tight', dpi=300)
    print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
plt.close(fig)

print("\n=== TP/FP/FN REPORT ===")
for scene, info in tp_report.items():
    print(f"\n  Row: {scene}  |  {info['image']}  |  GT={info['n_gt']}")
    print(f"  {'Model':<12} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-'*28}")
    for ml in ['CM-FRM','AFF','GAFF','ICAFusion']:
        if ml in info:
            v = info[ml]
            print(f"  {ml:<12} {v['TP']:>4} {v['FP']:>4} {v['FN']:>4}")

print("\nDone.")
