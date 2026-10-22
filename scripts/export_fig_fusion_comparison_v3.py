#!/usr/bin/env python3
"""
export_fig_fusion_comparison_v3.py — paper_figures/fig_fusion_comparison.(pdf|svg)
3 rows × 5 cols. Scenes FIXED after manual candidate scan:
  Row 0: Pre-pollination  — 2019_0101_001317_395.jpg  (GT=3, CMX=3/0/0, AFF=2/0/1)
  Row 1: Post-pollination — 2023_0610_144613_258.jpg  (GT=4, CMX=2/0/2, AFF=2/1/2, GAFF=1/1/3)
  Row 2: Net formation    — 2023_0620_161613_153.jpg  (unchanged, GT=10, CMX=9/0/1)
TP=green solid | FP=red solid | FN=orange dashed | GT col=grey dashed.
Style identical to fix_comparison_v8.py (serif, purple NIR, netting background).
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

# Fixed scenes: (image_file, target_class)
FIXED_SCENES = [
    ('2019_0101_001317_395.jpg', 0),  # Pre-pollination
    ('2023_0610_144613_258.jpg', 1),  # Post-pollination
    ('2023_0620_161613_153.jpg', 2),  # Net formation (unchanged)
]

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

print("Checkpoints:")
for name, p in [('CM-FRM', MODEL_CMX), ('AFF', MODEL_AFF), ('GAFF', MODEL_GAFF)]:
    print(f"  {name}: {p}")
print()

CLASS_NAMES = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
IOU_T   = 0.50
DISP_H  = 480
BOX_LW  = 4
GT_LW   = 3
GT_RGB  = (0, 229, 255)     # CYAN #00E5FF — high contrast on purple/blue NIR background
TP_RGB  = (0,   220,  80)
FP_RGB  = (255,  50,  50)
FN_RGB  = (255, 165,   0)

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['CMU Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.weight':      'normal',
    'axes.titleweight': 'normal',
    'font.size':        12,
    'axes.titlesize':   13,
})


# ── helpers ──────────────────────────────────────────────────────────────────
def draw_dashed(vis, x1, y1, x2, y2, color, lw=2, dash=12, gap=6):
    def dline(p1, p2):
        ax, ay = p1; bx, by = p2
        dist = max(int(np.hypot(bx-ax, by-ay)), 1)
        dx = (bx-ax)/dist; dy = (by-ay)/dist
        d = 0; on = True
        while d < dist:
            de = min(d+(dash if on else gap), dist)
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
    if   pos == 'br': x, y = w-tw-pad*2, h-pad
    elif pos == 'bl': x, y = pad, h-pad
    elif pos == 'tr': x, y = w-tw-pad*2, th+pad
    else:             x, y = pad, th+pad
    ov = panel.copy()
    cv2.rectangle(ov, (x-pad, y-th-pad), (x+tw+pad, y+bl+pad), (0,0,0), -1)
    cv2.addWeighted(ov, 0.60, panel, 0.40, 0, panel)
    cv2.putText(panel, text, (x,y), font, fs, (255,255,255), lw, cv2.LINE_AA)

def box_iou(a, b):
    xi1=max(a[0],b[0]); yi1=max(a[1],b[1])
    xi2=min(a[2],b[2]); yi2=min(a[3],b[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter==0: return 0.0
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)

def compute_tpfpfn(gt_boxes, pred_boxes, cls):
    gt_cls   = [b for b in gt_boxes   if b[0]==cls]
    pred_cls = sorted([p for p in pred_boxes if p[0]==cls], key=lambda p:-p[5])
    matched = set()
    tp, fp = [], []
    for pred in pred_cls:
        g = (pred[1],pred[2],pred[3],pred[4])
        best_iou, best_idx = 0.0, -1
        for i,gt in enumerate(gt_cls):
            if i in matched: continue
            iou = box_iou(g,(gt[1],gt[2],gt[3],gt[4]))
            if iou>best_iou: best_iou,best_idx = iou,i
        if best_iou>=IOU_T and best_idx>=0:
            tp.append(pred); matched.add(best_idx)
        else:
            fp.append(pred)
    fn = [gt_cls[i] for i in range(len(gt_cls)) if i not in matched]
    return tp, fp, fn

def square_roi(boxes, W, H, pad_f=0.38, min_s=520):
    xs = [b[1] for b in boxes]+[b[3] for b in boxes]
    ys = [b[2] for b in boxes]+[b[4] for b in boxes]
    span = max(max(xs)-min(xs), max(ys)-min(ys))
    pad  = max(int(span*pad_f), 90)
    side = max(span+2*pad, min_s)
    cx = (min(xs)+max(xs))//2; cy = (min(ys)+max(ys))//2
    x1 = max(0, cx-side//2); x2 = min(W, x1+side)
    y1 = max(0, cy-side//2); y2 = min(H, y1+side)
    return x1, y1, x2, y2

def overlaps(b, x1r, y1r, x2r, y2r):
    return b[1]<x2r and b[3]>x1r and b[2]<y2r and b[4]>y1r

def load_4ch(ip):
    bgr = cv2.imread(ip)
    if bgr is None: return None,None
    stem = Path(ip).stem; ndvi=None
    for sp in ['test','val','train']:
        p = os.path.join(NDVI_DIR,'images',sp,stem+'.png')
        if os.path.exists(p): ndvi=cv2.imread(p,cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir=bgr[:,:,2].astype(np.float32); red=bgr[:,:,1].astype(np.float32)
        ndvi=(((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape!=bgr.shape[:2]: ndvi=cv2.resize(ndvi,(bgr.shape[1],bgr.shape[0]))
    return bgr, np.dstack([bgr,ndvi])

def infer(model, ip):
    bgr, img4 = load_4ch(ip)
    if bgr is None: return []
    res = model.predict(source=img4, conf=0.25, iou=0.6, imgsz=1280, verbose=False)
    return [(int(b.cls[0]),*[int(v) for v in b.xyxy[0].tolist()],float(b.conf[0]))
            for b in res[0].boxes]

def read_gt(ip, W, H):
    lp = os.path.join(RAW_DIR,'labels','test',Path(ip).stem+'.txt')
    boxes=[]
    if os.path.exists(lp):
        for ln in open(lp):
            p=ln.strip().split()
            if not p: continue
            c=int(p[0])
            cx_,cy_,bw_,bh_=float(p[1]),float(p[2]),float(p[3]),float(p[4])
            x1=int((cx_-bw_/2)*W); y1=int((cy_-bh_/2)*H)
            x2=int((cx_+bw_/2)*W); y2=int((cy_+bh_/2)*H)
            boxes.append((c,x1,y1,x2,y2))
    return boxes


# ── load models ──────────────────────────────────────────────────────────────
print("Loading models ...")
from ultralytics import YOLO
m_cmx  = YOLO(MODEL_CMX)
m_aff  = YOLO(MODEL_AFF)
m_gaff = YOLO(MODEL_GAFF)
models      = [m_cmx, m_aff, m_gaff]
model_keys  = ['CM-FRM', 'AFF', 'GAFF']
print("  Done.\n")


# ── run inference for each fixed scene ───────────────────────────────────────
print("Running inference on fixed scenes ...")
scenes = []
for fname, cls_id in FIXED_SCENES:
    # Search in test split
    ip = None
    for split in ['test','val','train']:
        p = os.path.join(RAW_DIR,'images',split,fname)
        if os.path.exists(p): ip=p; break
    if ip is None:
        print(f"  WARNING: {fname} not found!"); scenes.append(None); continue

    bgr = cv2.imread(ip)
    H, W = bgr.shape[:2]
    all_gt = read_gt(ip, W, H)
    gt_cls = [b for b in all_gt if b[0]==cls_id]

    # ROI from GT boxes
    roi = square_roi(gt_cls, W, H) if gt_cls else (0,0,W,H)

    # Inference
    preds_list = [infer(m, ip) for m in models]

    scenes.append(dict(ip=ip, cls=cls_id, roi=roi,
                       all_gt=all_gt, gt_cls=gt_cls,
                       preds=preds_list))
    print(f"  {fname}  cls={CLASS_NAMES[cls_id]}  GT_cls={len(gt_cls)}")

print()


# ── draw 3 × 5 ───────────────────────────────────────────────────────────────
N_ROWS = len(FIXED_SCENES)
COL_TITLES = ['Ground Truth', 'Proposed\n(CM-FRM)', 'AFF', 'GAFF']

fig, axes = plt.subplots(N_ROWS, 4, figsize=(9.4, 2.6*N_ROWS))
fig.subplots_adjust(wspace=0.025, hspace=0.06,
                    left=0.10, right=0.995, top=0.89, bottom=0.04)

for col, t in enumerate(COL_TITLES):
    axes[0][col].set_title(t, fontsize=12, pad=5)

tp_report = {}

for row_idx, scene in enumerate(scenes):
    _, cls_id = FIXED_SCENES[row_idx]
    cname = CLASS_NAMES[cls_id]
    axes[row_idx][0].set_ylabel(cname, fontsize=11,
                                rotation=90, va='center', labelpad=5)

    if scene is None:
        for c in range(4):
            axes[row_idx][c].set_facecolor('#1a1a1a')
            axes[row_idx][c].set_xticks([]); axes[row_idx][c].set_yticks([])
        continue

    ip = scene['ip']
    x1r,y1r,x2r,y2r = scene['roi']
    gt_cls = scene['gt_cls']
    all_gt = scene['all_gt']
    bgr = cv2.imread(ip)
    H,W = bgr.shape[:2]
    ch = y2r-y1r; cw = x2r-x1r
    sc = DISP_H/max(ch,1); dw = int(cw*sc)

    crop_bgr     = bgr[y1r:y2r, x1r:x2r].copy()
    crop_boosted = np.clip(crop_bgr.astype(np.float32)*1.15, 0,255).astype(np.uint8)
    base_disp    = cv2.resize(cv2.cvtColor(crop_boosted,cv2.COLOR_BGR2RGB),
                              (dw, DISP_H),
                              interpolation=cv2.INTER_AREA if sc<=1 else cv2.INTER_CUBIC)

    def to_d(boxes):
        out=[]
        for b in boxes:
            nx1=max(0,b[1]-x1r); ny1=max(0,b[2]-y1r)
            nx2=min(cw,b[3]-x1r); ny2=min(ch,b[4]-y1r)
            if nx2>nx1 and ny2>ny1:
                out.append((b[0],int(nx1*sc),int(ny1*sc),int(nx2*sc),int(ny2*sc)))
        return out

    gt_roi = [b for b in gt_cls if overlaps(b,x1r,y1r,x2r,y2r)]

    # TP/FP/FN per model (predictions filtered to ROI before matching)
    tpfpfn_list = []
    for preds in scene['preds']:
        preds_roi = [p for p in preds if p[0]==cls_id and overlaps(p,x1r,y1r,x2r,y2r)]
        tp,fp,fn  = compute_tpfpfn(gt_roi, preds_roi, cls_id)
        tpfpfn_list.append((tp,fp,fn))

    tp_report[cname] = {'image': Path(ip).name, 'n_gt': len(gt_roi)}
    for mk,(tp,fp,fn) in zip(model_keys, tpfpfn_list):
        tp_report[cname][mk] = dict(TP=len(tp),FP=len(fp),FN=len(fn))

    # ── Column 0: Ground Truth ──
    gt_panel = base_disp.copy()
    for b in to_d(gt_roi):
        _,x1,y1,x2,y2 = b
        draw_dashed(gt_panel,x1,y1,x2,y2, GT_RGB, lw=GT_LW)
    corner_text(gt_panel, f"GT: {len(gt_roi)}", 'br')
    axes[row_idx][0].imshow(gt_panel)
    axes[row_idx][0].set_xticks([]); axes[row_idx][0].set_yticks([])

    # ── Columns 1–4: models ──
    for col_off, (mk,(tp,fp,fn)) in enumerate(zip(model_keys, tpfpfn_list)):
        panel = base_disp.copy()
        for b in to_d(tp):
            _,x1,y1,x2,y2 = b
            cv2.rectangle(panel,(x1,y1),(x2,y2), TP_RGB, BOX_LW)
        for b in to_d(fp):
            _,x1,y1,x2,y2 = b
            cv2.rectangle(panel,(x1,y1),(x2,y2), FP_RGB, BOX_LW)
        for b in to_d(fn):
            _,x1,y1,x2,y2 = b
            draw_dashed(panel,x1,y1,x2,y2, FN_RGB, lw=GT_LW+1, dash=10, gap=5)
        corner_text(panel, f"TP:{len(tp)} FP:{len(fp)} FN:{len(fn)}", 'br')
        axes[row_idx][col_off+1].imshow(panel)
        axes[row_idx][col_off+1].set_xticks([]); axes[row_idx][col_off+1].set_yticks([])

# ── legend ───────────────────────────────────────────────────────────────────
tp_p  = mpatches.Patch(facecolor=(0/255,220/255,80/255),  label='TP (green solid)')
fp_p  = mpatches.Patch(facecolor=(255/255,50/255,50/255), label='FP (red solid)')
fn_p  = mpatches.Patch(facecolor='none', edgecolor=(1,165/255,0),
                        linewidth=1.5, linestyle=(0,(3,2)), label='FN (orange dashed)')
gt_p  = mpatches.Patch(facecolor='none', edgecolor='#00E5FF',
                        linewidth=1.5, linestyle=(0,(4,2)), label='GT (cyan dashed)')
fig.legend(handles=[tp_p, fp_p, fn_p, gt_p],
           loc='lower center', ncol=4, fontsize=10,
           frameon=True, bbox_to_anchor=(0.55, 0.005),
           handlelength=1.4, handleheight=0.8)


# ── save ─────────────────────────────────────────────────────────────────────
print("Saving ...")
for ext in ['pdf','svg']:
    p = os.path.join(OUT_DIR, f'fig_fusion_comparison.{ext}')
    fig.savefig(p, format=ext, bbox_inches='tight', dpi=300)
    print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
plt.close(fig)

print("\n=== TP/FP/FN REPORT ===")
for scene_name, info in tp_report.items():
    print(f"\n  Row: {scene_name}  |  {info['image']}  |  GT={info['n_gt']}")
    print(f"  {'Model':<12} {'TP':>4} {'FP':>4} {'FN':>4}")
    print(f"  {'-'*28}")
    for mk in model_keys:
        if mk in info:
            v = info[mk]
            print(f"  {mk:<12} {v['TP']:>4} {v['FP']:>4} {v['FN']:>4}")

print("\nDone.")
