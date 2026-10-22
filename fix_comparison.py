#!/usr/bin/env python3
"""
fix_comparison.py
─────────────────
Regenerates:
  paper_figures/fig_comparison.(pdf|svg)   — zoomed-in crops, "Proposed" label, no suptitle
  paper_figures/fig_confusion_matrix.(pdf|svg)  — removes ax title
  paper_figures/fig_failcase.(pdf|svg)          — removes suptitle, "Proposed" label
  paper_figures/confusion_matrix_values.txt     — rebuilt from same CMX inference

Strategy for comparison:
  Run all 3 models on 80 candidate images (post-pol heavy).
  Find GT boxes where Proposed matches but YOLOv11s doesn't.
  Pick 3 diverse scenes; crop tightly to 2–4 fruits.
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

random.seed(42)
np.random.seed(42)

# ─── Paths ─────────────────────────────────────────────────────────────────
REPO_DIR  = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics'
OUT_DIR   = os.path.join(REPO_DIR, 'paper_figures')
RAW_DIR   = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
NDVI_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi'
DATA_YAML = os.path.join(RAW_DIR, 'data.yaml')

MODEL_CMX = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
MODEL_V11 = '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolo11s_1280_balanced_b4/weights/best.pt'
MODEL_EF  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp1_NoPretrain_1280px/weights/best.pt')

TEST_IMG_DIR = os.path.join(RAW_DIR, 'images', 'test')

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

# ─── Style ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
CLASS_SHORT  = ['Pre-pol.', 'Post-pol.', 'Net form.', 'Sweetening']
NC = 4

# High-contrast, avoids red-on-purple: magenta / cyan / gold / orange
BOX_RGB_HEX = ['#FF00FF', '#00FFFF', '#FFD700', '#FF8C00']
BOX_BGR     = [(255,  0, 255),   # magenta
               (255, 255,   0),  # cyan (BGR)
               (  0, 215, 255),  # gold (BGR)
               (  0, 140, 255)]  # orange (BGR)

GT_BGR  = (255, 255, 255)   # white GT boxes

plt.rcParams.update({
    'font.family':    'DejaVu Sans',
    'font.size':      9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

# ─── Load models ───────────────────────────────────────────────────────────
from ultralytics import YOLO
print("Loading models ...")
m_cmx = YOLO(MODEL_CMX)
m_v11 = YOLO(MODEL_V11)
m_ef  = YOLO(MODEL_EF)
print("  Done.\n")

# ─── Helpers ───────────────────────────────────────────────────────────────
def load_4ch(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None:
        return None, None
    fname = Path(img_path).stem + '.png'
    ndvi  = None
    for sp in ['test', 'val', 'train']:
        p = os.path.join(NDVI_DIR, 'images', sp, fname)
        if os.path.exists(p):
            ndvi = cv2.imread(p, cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir = bgr[:,:,2].astype(np.float32); red = bgr[:,:,1].astype(np.float32)
        ndvi = (((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape != bgr.shape[:2]:
        ndvi = cv2.resize(ndvi,(bgr.shape[1],bgr.shape[0]))
    return bgr, np.dstack([bgr,ndvi])

def load_gt(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None: return None,[]
    H,W = bgr.shape[:2]
    stem = Path(img_path).stem
    boxes = []
    for sp in ['test','val','train']:
        lp = os.path.join(RAW_DIR,'labels',sp,stem+'.txt')
        if os.path.exists(lp):
            for line in open(lp):
                p = line.strip().split()
                if not p: continue
                cls=int(p[0]); cx,cy,bw,bh=float(p[1]),float(p[2]),float(p[3]),float(p[4])
                x1=int((cx-bw/2)*W); y1=int((cy-bh/2)*H)
                x2=int((cx+bw/2)*W); y2=int((cy+bh/2)*H)
                boxes.append((cls,x1,y1,x2,y2))
            break
    return bgr, boxes

def run_infer(model, img_path, is_4ch, conf=0.25, iou=0.6):
    bgr, img4 = load_4ch(img_path)
    if bgr is None: return None,[]
    src = img4 if is_4ch else img_path
    res = model.predict(source=src, conf=conf, iou=iou, imgsz=1280, verbose=False)
    boxes = []
    for box in res[0].boxes:
        cls=int(box.cls[0]); cf=float(box.conf[0])
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append((cls,x1,y1,x2,y2,cf))
    return bgr, boxes

def box_iou(b1,b2):
    xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter==0: return 0.0
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter)

def draw_boxes_cv(bgr_in, pred_boxes, gt_boxes=None, lw=8, fs=0.70):
    """Draw GT (white) then pred (class color). Returns RGB crop."""
    vis = bgr_in.copy()
    if gt_boxes:
        for (cls,x1,y1,x2,y2) in gt_boxes:
            cv2.rectangle(vis,(x1,y1),(x2,y2),GT_BGR,lw)
            lbl=CLASS_SHORT[cls]
            (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,fs,2)
            cv2.rectangle(vis,(x1,y1-th-10),(x1+tw+6,y1),GT_BGR,-1)
            cv2.putText(vis,lbl,(x1+3,y1-4),cv2.FONT_HERSHEY_DUPLEX,fs,(0,0,0),2,cv2.LINE_AA)
    for (cls,x1,y1,x2,y2,cf) in pred_boxes:
        color=BOX_BGR[cls]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,lw)
        lbl=f'{CLASS_SHORT[cls]} {cf:.2f}'
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,fs,2)
        cv2.rectangle(vis,(x1,y1-th-10),(x1+tw+6,y1),color,-1)
        tc=(0,0,0) if sum(color)>400 else (255,255,255)
        cv2.putText(vis,lbl,(x1+3,y1-4),cv2.FONT_HERSHEY_DUPLEX,fs,tc,2,cv2.LINE_AA)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

def gt_only_cv(bgr_in, gt_boxes, lw=8, fs=0.70):
    vis = bgr_in.copy()
    for (cls,x1,y1,x2,y2) in gt_boxes:
        color = BOX_BGR[cls]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,lw)
        lbl=CLASS_SHORT[cls]
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,fs,2)
        cv2.rectangle(vis,(x1,y1-th-10),(x1+tw+6,y1),color,-1)
        tc=(0,0,0) if sum(color)>400 else (255,255,255)
        cv2.putText(vis,lbl,(x1+3,y1-4),cv2.FONT_HERSHEY_DUPLEX,fs,tc,2,cv2.LINE_AA)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

def clip_boxes_to_roi(boxes, x1r,y1r,x2r,y2r, margin=60):
    """Keep only boxes whose center falls inside ROI+margin."""
    kept = []
    for b in boxes:
        bx1,by1,bx2,by2 = b[1],b[2],b[3],b[4]
        cx=(bx1+bx2)//2; cy=(by1+by2)//2
        if x1r-margin<=cx<=x2r+margin and y1r-margin<=cy<=y2r+margin:
            kept.append(b)
    return kept

def tight_crop_rgb(img_rgb, roi, target_h=520):
    """Crop img_rgb to roi=(x1,y1,x2,y2); resize to target_h keeping AR."""
    x1,y1,x2,y2 = roi
    H,W = img_rgb.shape[:2]
    x1=max(0,x1); y1=max(0,y1); x2=min(W,x2); y2=min(H,y2)
    crop = img_rgb[y1:y2, x1:x2]
    if crop.size==0: return img_rgb
    h,w = crop.shape[:2]
    nw = int(w*target_h/h)
    return cv2.resize(crop,(nw,target_h),interpolation=cv2.INTER_AREA)

def pad_to_width(img, target_w, fill=0):
    h,w = img.shape[:2]
    if w >= target_w: return img[:, :target_w]
    pad = np.full((h, target_w-w, 3), fill, dtype=np.uint8)
    return np.hstack([img, pad])

def save_fig(fig, name):
    for ext in ['pdf','svg']:
        p = os.path.join(OUT_DIR,f'{name}.{ext}')
        fig.savefig(p,format=ext,bbox_inches='tight',dpi=300)
        print(f"  Saved: {p}")
    plt.close(fig)

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Run all 3 models on the 3 known comparison scenes (pre-identified)
#          + CMX on ALL test images for CM and failcase
# ══════════════════════════════════════════════════════════════════════════════
IOU_T = 0.5

# Scenes identified in the previous run (score=30/30/25, diverse classes)
COMP_STEMS = [
    '2023_0610_145243_453',   # post-pol dominant, score=30, 2 wins
    '2023_0620_162254_353',   # net formation, score=30, 3 wins
    '2023_0610_145033_388',   # post-pol+pre-pol, score=25, 2 wins
]
comp_imgs = [os.path.join(TEST_IMG_DIR, s+'.jpg') for s in COMP_STEMS]

all_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR,'*.jpg')))

print(f"Running 3 models on {len(comp_imgs)} comparison scenes ...")
cand_data = {}
for ip in comp_imgs:
    print(f"  {os.path.basename(ip)}")
    bgr,cmx = run_infer(m_cmx,ip,is_4ch=True,  conf=0.25,iou=0.6)
    _,  v11 = run_infer(m_v11,ip,is_4ch=False, conf=0.25,iou=0.6)
    _,  ef  = run_infer(m_ef, ip,is_4ch=True,  conf=0.25,iou=0.6)
    _,  gt  = load_gt(ip)
    cand_data[ip] = dict(bgr=bgr, cmx=cmx, v11=v11, ef=ef, gt=gt)
print(f"  Done.\n")

# Run CMX on ALL test images (for confusion matrix + failcase)
print(f"Running CMX on ALL {len(all_test)} test images (for CM + failcase) ...")
all_cmx = {}
for i,ip in enumerate(all_test):
    if (i+1)%50==0: print(f"  {i+1}/{len(all_test)}")
    if ip in cand_data:
        all_cmx[ip] = (cand_data[ip]['bgr'], cand_data[ip]['cmx'], cand_data[ip]['gt'])
    else:
        bgr,cmx = run_infer(m_cmx,ip,is_4ch=True,conf=0.25,iou=0.6)
        _,gt    = load_gt(ip)
        all_cmx[ip] = (bgr,cmx,gt)
print(f"  Done.\n")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Determine win boxes for each selected scene
# ══════════════════════════════════════════════════════════════════════════════

win_scenes = []
for ip in comp_imgs:
    d = cand_data[ip]
    gt=d['gt']; cmx=d['cmx']; v11=d['v11']; ef=d['ef']; bgr=d['bgr']
    if bgr is None: continue
    wins = []
    for gbox in gt:
        gc,gx1,gy1,gx2,gy2 = gbox
        cmx_iou = max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in cmx), default=0)
        v11_iou = max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in v11), default=0)
        if cmx_iou>=IOU_T and v11_iou<IOU_T:
            wins.append(gbox)
    if wins:
        # score = wins that also EF misses (double-win) + wins where box is reasonably sized
        H,W = bgr.shape[:2]
        double_wins = 0
        for gbox in wins:
            gc,gx1,gy1,gx2,gy2 = gbox
            ef_iou = max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in ef), default=0)
            if ef_iou < IOU_T: double_wins += 1
        win_scenes.append((ip, wins, gt, cmx, v11, ef))

selected = win_scenes   # already the 3 curated scenes in order
print(f"Comparison scenes with win boxes:")
for ip,wins,gt,*_ in selected:
    print(f"  {os.path.basename(ip)}: {len(wins)} CMX-wins, {len(gt)} GT boxes total")
print()

# ── Build crop ROI for each scene ─────────────────────────────────────────
def build_roi(win_boxes, gt_boxes, W, H, max_fruits=4, pad_ratio=0.55):
    """
    Center ROI on win boxes, expand to include nearby GT (up to max_fruits total),
    then add padding.  Returns (x1,y1,x2,y2).
    """
    # Sort nearby GT boxes by distance to centroid of win_boxes
    wcx = np.mean([(b[1]+b[3])//2 for b in win_boxes])
    wcy = np.mean([(b[2]+b[4])//2 for b in win_boxes])
    gt_dist = sorted(gt_boxes, key=lambda b: ((b[1]+b[3])//2-wcx)**2+((b[2]+b[4])//2-wcy)**2)
    roi_boxes = gt_dist[:max_fruits]
    xs = [b[1] for b in roi_boxes]+[b[3] for b in roi_boxes]
    ys = [b[2] for b in roi_boxes]+[b[4] for b in roi_boxes]
    x1,x2,y1,y2 = min(xs),max(xs),min(ys),max(ys)
    bw,bh = max(x2-x1,1),max(y2-y1,1)
    pad = max(int(max(bw,bh)*pad_ratio), 120)
    return (max(0,x1-pad), max(0,y1-pad), min(W,x2+pad), min(H,y2+pad))

# ══════════════════════════════════════════════════════════════════════════════
# FIG COMPARISON — 3 rows × 4 cols (GT | V11 | EF | Proposed)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/3] fig_comparison ...")

COL_TITLES = ['Ground Truth', 'YOLOv11s\n(R/G/NIR)', 'EarlyFusion\n(4ch)', 'Proposed\n(4ch, Dual-stream)']
DISP_H     = 480   # height of each panel in display pixels

nrows = len(selected)
fig, axes = plt.subplots(nrows, 4, figsize=(14.0, 4.2*nrows))
fig.subplots_adjust(wspace=0.03, hspace=0.08, left=0.07, right=0.99,
                    top=0.93, bottom=0.03)

for col,title in enumerate(COL_TITLES):
    axes[0][col].set_title(title, fontsize=9.5, fontweight='bold', pad=5)

row_labels = []
for row,(ip,wins,gt,cmx,v11,ef) in enumerate(selected):
    bgr = cv2.imread(ip)
    H,W = bgr.shape[:2]

    # Build crop ROI
    roi = build_roi(wins, gt, W, H, max_fruits=4, pad_ratio=0.55)
    x1r,y1r,x2r,y2r = roi

    # Clip pred boxes to ROI area
    gt_in   = [b for b in gt  if x1r-80<=(b[1]+b[3])//2<=x2r+80 and y1r-80<=(b[2]+b[4])//2<=y2r+80]
    cmx_in  = clip_boxes_to_roi(cmx,  x1r,y1r,x2r,y2r)
    v11_in  = clip_boxes_to_roi(v11,  x1r,y1r,x2r,y2r)
    ef_in   = clip_boxes_to_roi(ef,   x1r,y1r,x2r,y2r)

    # Annotated full images → crop → resize
    imgs = [
        gt_only_cv(bgr,  gt_in,  lw=8, fs=0.70),
        draw_boxes_cv(bgr, v11_in, lw=8, fs=0.70),
        draw_boxes_cv(bgr, ef_in,  lw=8, fs=0.70),
        draw_boxes_cv(bgr, cmx_in, lw=8, fs=0.70),
    ]

    # Crop ROI then resize each panel to fixed height
    crops_resized = []
    for im in imgs:
        crop = im[y1r:y2r, x1r:x2r]
        if crop.size == 0: crop = im
        h, w = crop.shape[:2]
        nw = int(w * DISP_H / max(h, 1))
        crops_resized.append(cv2.resize(crop, (nw, DISP_H), interpolation=cv2.INTER_AREA))
    # Pad all to same width (widest in row)
    max_w = max(c.shape[1] for c in crops_resized)
    crops_resized = [pad_to_width(c, max_w) for c in crops_resized]

    for col,crop in enumerate(crops_resized):
        axes[row][col].imshow(crop)
        axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
        for sp in axes[row][col].spines.values():
            sp.set_visible(True); sp.set_linewidth(1.0); sp.set_edgecolor('#aaa')

    dom = max(set(b[0] for b in gt_in), key=lambda c: sum(1 for b in gt_in if b[0]==c)) if gt_in else 0
    lbl = ['Pre-pollination','Post-pollination','Net formation','Sweetening'][dom]
    axes[row][0].set_ylabel(lbl, fontsize=9.5, fontweight='bold',
                             rotation=90, va='center', labelpad=8)

# Legend
handles = [mpatches.Patch(color='white', label='GT box (colored)', ec='gray')]
for i in range(4):
    handles.append(mpatches.Patch(color=BOX_RGB_HEX[i], label=CLASS_NAMES[i]))
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=8.5,
           frameon=True, bbox_to_anchor=(0.55, -0.01))

save_fig(fig,'fig_comparison')
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# Rebuild CM + failure data from all_cmx
# ══════════════════════════════════════════════════════════════════════════════
print("Rebuilding confusion matrix ...")
cm = np.zeros((NC+1,NC+1),dtype=np.int64)

fn_cases=[]; fp_cases=[]; conf_cases=[]; glare_cases=[]

for ip,(bgr,pred,gt) in all_cmx.items():
    gt_matched=[False]*len(gt); pred_matched=[False]*len(pred)
    for gi,(gc,gx1,gy1,gx2,gy2) in enumerate(gt):
        best_iou,best_pi=0.0,-1
        for pi,(pc,px1,py1,px2,py2,_) in enumerate(pred):
            if pred_matched[pi]: continue
            iv=box_iou((gx1,gy1,gx2,gy2),(px1,py1,px2,py2))
            if iv>best_iou: best_iou,best_pi=iv,pi
        if best_pi>=0 and best_iou>=IOU_T:
            pred_matched[best_pi]=True; gt_matched[gi]=True
            pc=pred[best_pi][0]; cm[gc][pc]+=1
            if pc!=gc: conf_cases.append((ip,gt[gi],pred[best_pi],pred,gt))
        else:
            cm[gc][NC]+=1; fn_cases.append((ip,gt[gi],pred,gt))
    for pi,pbox in enumerate(pred):
        if not pred_matched[pi]:
            cm[NC][pbox[0]]+=1; fp_cases.append((ip,pbox,pred,gt))
    if len(gt)>=3:
        recall=sum(gt_matched)/len(gt)
        if recall<0.5:
            avg_a=np.mean([(b[3]-b[1])*(b[4]-b[2]) for b in gt]) if gt else 0
            glare_cases.append((recall,avg_a,ip,pred,gt))

glare_cases.sort(key=lambda x:x[0])
print(f"  FN={len(fn_cases)} FP={len(fp_cases)} Conf={len(conf_cases)} Glare={len(glare_cases)}\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIG CONFUSION MATRIX  (no ax title)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/3] fig_confusion_matrix ...")

labels_5 = CLASS_NAMES + ['Background']
cm_f = cm.astype(np.float64)
row_sum = cm_f.sum(axis=1,keepdims=True).clip(min=1)
cm_norm = cm_f/row_sum

with open(os.path.join(OUT_DIR,'confusion_matrix_values.txt'),'w') as f:
    f.write("Proposed (GMD-YOLO) Confusion Matrix  (test-383, conf=0.25, iou_match=0.5)\n")
    f.write("Rows = Ground Truth  |  Cols = Predicted  |  'Background' = missed/FP\n\n")
    hdr='  '.join(f'{l[:6]:>8}' for l in labels_5)
    f.write(f"{'GT/Pred':>18}  {hdr}\n")
    for i,rl in enumerate(labels_5):
        f.write(f"  {rl:>16}  "+'  '.join(f'{int(cm[i,j]):>8d}' for j in range(NC+1))+'\n')
    f.write("\n--- Normalized (row-sum = 1.0, i.e. row = recall breakdown) ---\n")
    f.write(f"{'GT/Pred':>18}  {hdr}\n")
    for i,rl in enumerate(labels_5):
        f.write(f"  {rl:>16}  "+'  '.join(f'{cm_norm[i,j]:>8.3f}' for j in range(NC+1))+'\n')

fig,ax = plt.subplots(figsize=(6.5,5.8))
im = ax.imshow(cm_norm,cmap='Blues',vmin=0,vmax=1,aspect='auto')

for i in range(NC+1):
    for j in range(NC+1):
        v=cm_norm[i,j]; raw=int(cm[i,j])
        tc='white' if v>0.55 else 'black'
        ax.text(j,i-0.12,f'{v:.2f}',ha='center',va='center',
                fontsize=10.5,color=tc,fontweight='bold',fontfamily='DejaVu Sans')
        ax.text(j,i+0.22,f'({raw})',ha='center',va='center',
                fontsize=7.5,color=tc,fontfamily='DejaVu Sans')

tick_lbls = ['\n'.join(l.split()) for l in labels_5]
ax.set_xticks(range(NC+1)); ax.set_yticks(range(NC+1))
ax.set_xticklabels(tick_lbls,fontsize=8.5,fontfamily='DejaVu Sans')
ax.set_yticklabels(tick_lbls,fontsize=8.5,fontfamily='DejaVu Sans')
ax.set_xlabel('Predicted label',fontsize=9.5,labelpad=8)
ax.set_ylabel('Ground truth label',fontsize=9.5,labelpad=8)
# NO ax.set_title()  ← suptitle removed per spec
cb=fig.colorbar(im,ax=ax,fraction=0.046,pad=0.04)
cb.ax.tick_params(labelsize=8); cb.set_label('Recall fraction',fontsize=8.5)
fig.tight_layout()
save_fig(fig,'fig_confusion_matrix')
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG FAILCASE  (remove suptitle, "Proposed" label)
# ══════════════════════════════════════════════════════════════════════════════
print("[3/3] fig_failcase ...")

FAIL_H=420

def make_failcase_panel(ip,gt_boxes,pred_boxes,highlight_gt=None,highlight_pred=None):
    bgr=cv2.imread(ip)
    vis=bgr.copy()
    for (cls,x1,y1,x2,y2) in gt_boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),(180,180,180),3)
    for (cls,x1,y1,x2,y2,cf) in pred_boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),BOX_BGR[cls],5)
    if highlight_gt is not None:
        cls,x1,y1,x2,y2=highlight_gt[:5]
        cv2.rectangle(vis,(x1-4,y1-4),(x2+4,y2+4),(0,0,255),7)
        lbl=f'GT: {CLASS_SHORT[cls]}'
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.70,2)
        cv2.rectangle(vis,(x1,y2+2),(x1+tw+6,y2+th+14),(0,0,200),-1)
        cv2.putText(vis,lbl,(x1+3,y2+th+6),cv2.FONT_HERSHEY_DUPLEX,0.70,(255,255,255),2,cv2.LINE_AA)
    if highlight_pred is not None:
        cls,x1,y1,x2,y2,cf=highlight_pred
        cv2.rectangle(vis,(x1-4,y1-4),(x2+4,y2+4),(0,128,255),7)
        lbl=f'PRED: {CLASS_SHORT[cls]} {cf:.2f}'
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.70,2)
        cv2.rectangle(vis,(x1,y1-th-14),(x1+tw+6,y1),(0,128,200),-1)
        cv2.putText(vis,lbl,(x1+3,y1-4),cv2.FONT_HERSHEY_DUPLEX,0.70,(255,255,255),2,cv2.LINE_AA)
    vis_rgb=cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)
    if highlight_gt: roi_b=[highlight_gt]
    elif highlight_pred: roi_b=[(highlight_pred[0],highlight_pred[1],highlight_pred[2],highlight_pred[3],highlight_pred[4])]
    else: roi_b=gt_boxes[:3] if gt_boxes else []
    if roi_b:
        xs=[b[1] for b in roi_b]+[b[3] for b in roi_b]
        ys=[b[2] for b in roi_b]+[b[4] for b in roi_b]
        x1c,x2c,y1c,y2c=min(xs),max(xs),min(ys),max(ys)
        bw,bh=x2c-x1c,y2c-y1c
        pad=max(int(max(bw,bh)*0.7),150)
        H,W=vis_rgb.shape[:2]
        c1=max(0,x1c-pad);c2=min(W,x2c+pad);r1=max(0,y1c-pad);r2=min(H,y2c+pad)
        crop=vis_rgb[r1:r2,c1:c2]
    else: crop=vis_rgb
    h,w=crop.shape[:2]; nw=int(w*FAIL_H/h)
    return cv2.resize(crop,(nw,FAIL_H),interpolation=cv2.INTER_AREA)

def pick_best(cases,key_fn,n=1,min_area=800,max_area=500000):
    filtered=[c for c in cases if len(c)>=2 and min_area<=(c[1][3]-c[1][1])*(c[1][4]-c[1][2])<=max_area]
    seen=set(); result=[]
    for c in sorted(filtered,key=key_fn):
        if c[0] not in seen: seen.add(c[0]); result.append(c)
        if len(result)>=n: break
    return result

fn_pick   = pick_best(fn_cases,  key_fn=lambda c:-(c[1][3]-c[1][1])*(c[1][4]-c[1][2]), n=1)
fp_pick   = pick_best(fp_cases,  key_fn=lambda c:-c[1][5],                               n=1)
nsc_pick  = pick_best([c for c in conf_cases if {c[1][0],c[2][0]}=={2,3}],
                      key_fn=lambda c:-(c[1][3]-c[1][1])*(c[1][4]-c[1][2]),n=1)
if not nsc_pick:
    nsc_pick=pick_best(conf_cases,key_fn=lambda c:-(c[1][3]-c[1][1])*(c[1][4]-c[1][2]),n=1)
glare_pick=[];
for (recall,avg_a,ip,pred,gt) in glare_cases[:15]:
    if avg_a>=600: glare_pick.append((ip,pred,gt)); break

fail_panels=[]; fail_titles=[]
if fn_pick:
    ip,mg,pred,gt=fn_pick[0]
    fail_panels.append(make_failcase_panel(ip,gt,pred,highlight_gt=mg))
    fail_titles.append(f'(1) Missed detection\nGT: {CLASS_SHORT[mg[0]]}')
if fp_pick:
    ip,fp,pred,gt=fp_pick[0]
    fail_panels.append(make_failcase_panel(ip,gt,pred,highlight_pred=fp))
    fail_titles.append(f'(2) False positive\nPred: {CLASS_SHORT[fp[0]]} {fp[5]:.2f}')
if nsc_pick:
    ip,gb,pb,pred,gt=nsc_pick[0]
    fail_panels.append(make_failcase_panel(ip,gt,pred,highlight_gt=gb,highlight_pred=pb))
    fail_titles.append(f'(3) Class confusion\n{CLASS_SHORT[gb[0]]}→{CLASS_SHORT[pb[0]]}')
if glare_pick:
    ip,pred,gt=glare_pick[0]
    fail_panels.append(make_failcase_panel(ip,gt,pred))
    fail_titles.append('(4) Low-recall scene\n(occlusion/lighting)')

n_panels=len(fail_panels)
n_cols=min(n_panels,4); n_rows=(n_panels+n_cols-1)//n_cols
fig,axes=plt.subplots(n_rows,n_cols,figsize=(5.2*n_cols,5.2*n_rows))
if n_rows==1 and n_cols==1: axes=np.array([[axes]])
elif n_rows==1:              axes=axes[np.newaxis,:]
elif n_cols==1:              axes=axes[:,np.newaxis]

legend_handles=[
    mpatches.Patch(color='white',label='GT box (gray outline)',ec='gray'),
    mpatches.Patch(color='red',  label='Missed GT (red outline)'),
    mpatches.Patch(color='#FF8000',label='Highlighted pred (orange)'),
]
for i in range(4):
    legend_handles.append(mpatches.Patch(color=BOX_RGB_HEX[i],label=f'Pred: {CLASS_SHORT[i]}'))

for idx,(crop,title) in enumerate(zip(fail_panels,fail_titles)):
    r,c=idx//n_cols,idx%n_cols
    axes[r][c].imshow(crop)
    axes[r][c].set_title(title,fontsize=9,fontweight='bold',pad=5,fontfamily='DejaVu Sans')
    axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
for idx in range(n_panels,n_rows*n_cols):
    axes[idx//n_cols][idx%n_cols].axis('off')

# NO fig.suptitle() ← removed per spec
fig.legend(handles=legend_handles,loc='lower center',ncol=4,fontsize=8.5,
           frameon=True,bbox_to_anchor=(0.5,-0.04))
fig.tight_layout(pad=0.5)
save_fig(fig,'fig_failcase')
print("  Done.\n")

# ─── Summary ───────────────────────────────────────────────────────────────
print("="*60)
print(f"Regenerated files in: {OUT_DIR}/")
targets=['fig_comparison','fig_confusion_matrix','fig_failcase','confusion_matrix_values']
for t in targets:
    for ext in ['pdf','svg','txt']:
        p=os.path.join(OUT_DIR,f'{t}.{ext}' if not t.endswith('values') else f'{t}.txt')
        if os.path.exists(p):
            sz=os.path.getsize(p)//1024
            print(f"  {os.path.basename(p):50s} {sz:5d} KB")
