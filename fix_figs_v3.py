#!/usr/bin/env python3
"""
fix_figs_v3.py
──────────────
Regenerates:
  fig_dataset_samples.(pdf|svg)  — 2×4 layout (wide/short), blur-filtered, no suptitle
  fig_comparison.(pdf|svg)       — 3 diverse scenes (post-pol/net/pre-pol), correct labels
"""
import os, sys, glob, random
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

random.seed(42); np.random.seed(42)

# ─── Paths ─────────────────────────────────────────────────────────────────
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

# ─── Style ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
CLASS_SHORT  = ['Pre-pol.', 'Post-pol.', 'Net form.', 'Sweetening']
NC = 4

BOX_RGB_HEX = ['#FF00FF', '#00FFFF', '#FFD700', '#FF8C00']
BOX_BGR     = [(255, 0, 255), (255, 255, 0), (0, 215, 255), (0, 140, 255)]
GT_BGR      = (255, 255, 255)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
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
    if bgr is None: return None, None
    fname = Path(img_path).stem + '.png'
    ndvi  = None
    for sp in ['test','val','train']:
        p = os.path.join(NDVI_DIR,'images',sp,fname)
        if os.path.exists(p): ndvi=cv2.imread(p,cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir=bgr[:,:,2].astype(np.float32); red=bgr[:,:,1].astype(np.float32)
        ndvi=(((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape!=bgr.shape[:2]: ndvi=cv2.resize(ndvi,(bgr.shape[1],bgr.shape[0]))
    return bgr, np.dstack([bgr,ndvi])

def load_gt(img_path):
    bgr=cv2.imread(img_path)
    if bgr is None: return None,[]
    H,W=bgr.shape[:2]; stem=Path(img_path).stem; boxes=[]
    for sp in ['test','val','train']:
        lp=os.path.join(RAW_DIR,'labels',sp,stem+'.txt')
        if os.path.exists(lp):
            for line in open(lp):
                p=line.strip().split()
                if not p: continue
                cls=int(p[0]); cx,cy,bw,bh=float(p[1]),float(p[2]),float(p[3]),float(p[4])
                x1=int((cx-bw/2)*W); y1=int((cy-bh/2)*H)
                x2=int((cx+bw/2)*W); y2=int((cy+bh/2)*H)
                boxes.append((cls,x1,y1,x2,y2))
            break
    return bgr, boxes

def run_infer(model, img_path, is_4ch, conf=0.25, iou=0.6):
    bgr,img4=load_4ch(img_path)
    if bgr is None: return None,[]
    src=img4 if is_4ch else img_path
    res=model.predict(source=src,conf=conf,iou=iou,imgsz=1280,verbose=False)
    boxes=[]
    for box in res[0].boxes:
        cls=int(box.cls[0]); cf=float(box.conf[0])
        x1,y1,x2,y2=[int(v) for v in box.xyxy[0].tolist()]
        boxes.append((cls,x1,y1,x2,y2,cf))
    return bgr, boxes

def box_iou(b1,b2):
    xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter==0: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)

def sharpness(bgr, x1,y1,x2,y2, pad=30):
    H,W=bgr.shape[:2]
    c1=max(0,x1-pad); c2=min(W,x2+pad)
    r1=max(0,y1-pad); r2=min(H,y2+pad)
    roi=bgr[r1:r2,c1:c2]
    if roi.size==0: return 0
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray,cv2.CV_64F).var()

def draw_pred_cv(bgr_in, pred_boxes, lw=8, fs=0.68):
    vis=bgr_in.copy()
    for (cls,x1,y1,x2,y2,cf) in pred_boxes:
        color=BOX_BGR[cls]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,lw)
        lbl=f'{CLASS_SHORT[cls]} {cf:.2f}'
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,fs,2)
        cv2.rectangle(vis,(x1,y1-th-10),(x1+tw+6,y1),color,-1)
        tc=(0,0,0) if sum(color)>400 else (255,255,255)
        cv2.putText(vis,lbl,(x1+3,y1-4),cv2.FONT_HERSHEY_DUPLEX,fs,tc,2,cv2.LINE_AA)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

def draw_gt_cv(bgr_in, gt_boxes, lw=8, fs=0.68):
    vis=bgr_in.copy()
    for (cls,x1,y1,x2,y2) in gt_boxes:
        color=BOX_BGR[cls]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,lw)
        lbl=CLASS_SHORT[cls]
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,fs,2)
        cv2.rectangle(vis,(x1,y1-th-10),(x1+tw+6,y1),color,-1)
        tc=(0,0,0) if sum(color)>400 else (255,255,255)
        cv2.putText(vis,lbl,(x1+3,y1-4),cv2.FONT_HERSHEY_DUPLEX,fs,tc,2,cv2.LINE_AA)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

def clip_to_roi(boxes, x1r,y1r,x2r,y2r, margin=80):
    return [b for b in boxes if x1r-margin<=(b[1]+b[3])//2<=x2r+margin
            and y1r-margin<=(b[2]+b[4])//2<=y2r+margin]

def build_roi(win_boxes, gt_boxes, W, H, max_fruits=4, pad_ratio=0.55):
    wcx=np.mean([(b[1]+b[3])//2 for b in win_boxes])
    wcy=np.mean([(b[2]+b[4])//2 for b in win_boxes])
    gt_sort=sorted(gt_boxes,key=lambda b:((b[1]+b[3])//2-wcx)**2+((b[2]+b[4])//2-wcy)**2)
    roi_b=gt_sort[:max_fruits]
    xs=[b[1] for b in roi_b]+[b[3] for b in roi_b]
    ys=[b[2] for b in roi_b]+[b[4] for b in roi_b]
    x1,x2,y1,y2=min(xs),max(xs),min(ys),max(ys)
    pad=max(int(max(x2-x1,y2-y1)*pad_ratio),120)
    return max(0,x1-pad),max(0,y1-pad),min(W,x2+pad),min(H,y2+pad)

def crop_resize(img_rgb, x1r,y1r,x2r,y2r, target_h):
    crop=img_rgb[y1r:y2r,x1r:x2r]
    if crop.size==0: crop=img_rgb
    h,w=crop.shape[:2]; nw=int(w*target_h/max(h,1))
    return cv2.resize(crop,(nw,target_h),interpolation=cv2.INTER_AREA)

def pad_w(img, target_w):
    h,w=img.shape[:2]
    if w>=target_w: return img[:,:target_w]
    pad=np.full((h,target_w-w,3),30,dtype=np.uint8)
    return np.hstack([img,pad])

def save_fig(fig, name):
    for ext in ['pdf','svg']:
        p=os.path.join(OUT_DIR,f'{name}.{ext}')
        fig.savefig(p,format=ext,bbox_inches='tight',dpi=300)
        print(f"  Saved: {p}")
    plt.close(fig)

all_test=sorted(glob.glob(os.path.join(TEST_IMG_DIR,'*.jpg')))

# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — DATASET SAMPLES  2 rows × 4 cols  (wide/short, no suptitle)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/2] fig_dataset_samples ...")

BLUR_THRESH = 80    # Laplacian variance threshold (lower = accept more images)
MIN_AREA    = 2000  # px² minimum box area
MAX_AREA    = 250000
MARGIN_PX   = 80    # min distance from image edge

def pick_sharp_crops(cls_target, n=2):
    candidates = []
    for ip in all_test:
        bgr=cv2.imread(ip)
        if bgr is None: continue
        H,W=bgr.shape[:2]
        _,gt=load_gt(ip)
        for (cls,x1,y1,x2,y2) in gt:
            if cls!=cls_target: continue
            area=(x2-x1)*(y2-y1)
            if not (MIN_AREA<=area<=MAX_AREA): continue
            if x1<MARGIN_PX or y1<MARGIN_PX or x2>W-MARGIN_PX or y2>H-MARGIN_PX: continue
            sharp=sharpness(bgr,x1,y1,x2,y2,pad=60)
            if sharp<BLUR_THRESH: continue
            # Prefer boxes with "ideal" area ~8 000–20 000 px²
            size_score=abs(np.log(max(area,1))-np.log(12000))
            candidates.append((sharp, size_score, area, ip, cls, x1,y1,x2,y2))

    # Sort: sharpest first, then ideal size
    candidates.sort(key=lambda c:(-c[0], c[1]))

    seen=set(); crops=[]
    for (sharp,ss,area,ip,cls,x1,y1,x2,y2) in candidates:
        if ip in seen: continue
        bgr=cv2.imread(ip); H,W=bgr.shape[:2]
        bw,bh=x2-x1,y2-y1
        pad=max(int(max(bw,bh)*0.65),100)
        c1=max(0,x1-pad); c2=min(W,x2+pad)
        r1=max(0,y1-pad); r2=min(H,y2+pad)
        crop=bgr[r1:r2,c1:c2].copy()
        nx1,ny1=x1-c1,y1-r1; nx2,ny2=x2-c1,y2-r1
        color=BOX_BGR[cls]
        cv2.rectangle(crop,(nx1,ny1),(nx2,ny2),color,7)
        lbl=CLASS_SHORT[cls]
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.65,2)
        cv2.rectangle(crop,(nx1,ny1-th-10),(nx1+tw+6,ny1),color,-1)
        tc=(0,0,0) if sum(color)>400 else (255,255,255)
        cv2.putText(crop,lbl,(nx1+3,ny1-4),cv2.FONT_HERSHEY_DUPLEX,0.65,tc,2,cv2.LINE_AA)
        crops.append(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
        seen.add(ip)
        if len(crops)>=n: break

    # Pad to n if not enough
    while len(crops)<n:
        crops.append(np.zeros((400,400,3),dtype=np.uint8))
    return crops

# Collect 2 crops per class
all_crops=[]   # shape: [4 classes][2 samples]
for cls in range(4):
    print(f"  Class {cls}: picking crops ...")
    all_crops.append(pick_sharp_crops(cls, n=2))

# Normalise all crops to same square size for consistent grid
CELL_PX = 500   # target cell size (square)
def square_crop(img_rgb, size=CELL_PX):
    h,w=img_rgb.shape[:2]
    # Centre crop to square
    side=min(h,w)
    r1=(h-side)//2; r2=r1+side
    c1=(w-side)//2; c2=c1+side
    sq=img_rgb[r1:r2,c1:c2]
    return cv2.resize(sq,(size,size),interpolation=cv2.INTER_AREA)

fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.0))
fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.02, right=0.99,
                    top=0.91, bottom=0.01)

# Column headers = class names
for cls in range(4):
    axes[0][cls].set_title(CLASS_NAMES[cls], fontsize=10, fontweight='bold',
                            color=BOX_RGB_HEX[cls], pad=5)

for row in range(2):
    for cls in range(4):
        ax = axes[row][cls]
        img = square_crop(all_crops[cls][row])
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(BOX_RGB_HEX[cls]); sp.set_linewidth(2.5)

# Row labels on left margin
for row, lbl in enumerate(['Sample 1', 'Sample 2']):
    axes[row][0].set_ylabel(lbl, fontsize=9, rotation=90, va='center', labelpad=6)

# NO suptitle
save_fig(fig, 'fig_dataset_samples')
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — COMPARISON  3 diverse scenes: Post-pol / Net form. / Pre-pol
# ══════════════════════════════════════════════════════════════════════════════
print("[2/2] fig_comparison ...")

# 3 scenes with different dominant class & confirmed CMX wins
COMP_STEMS = [
    '2023_0610_145243_453',   # Post-pol dominant (9 post + 4 pre), CMX wins=2
    '2023_0620_162254_353',   # Net formation (13), CMX wins=3
    '2019_0101_001915_574',   # Pre-pol only (9), CMX wins=2
]
comp_imgs=[os.path.join(TEST_IMG_DIR,s+'.jpg') for s in COMP_STEMS]

IOU_T=0.5

print("  Running 3 models on 3 scenes ...")
scene_data=[]
for ip in comp_imgs:
    print(f"    {os.path.basename(ip)}")
    bgr,cmx=run_infer(m_cmx,ip,is_4ch=True,  conf=0.25,iou=0.6)
    _,  v11=run_infer(m_v11,ip,is_4ch=False, conf=0.25,iou=0.6)
    _,  ef =run_infer(m_ef, ip,is_4ch=True,  conf=0.25,iou=0.6)
    _,  gt =load_gt(ip)
    # Find win boxes (CMX detects, V11 misses)
    wins=[]
    for gbox in gt:
        gc,gx1,gy1,gx2,gy2=gbox
        ci=max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in cmx),default=0)
        vi=max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in v11),default=0)
        if ci>=IOU_T and vi<IOU_T: wins.append(gbox)
    # Fallback if no win found (pre-pol scene might have wins at low conf)
    if not wins:
        wins=gt[:2] if gt else []
    scene_data.append(dict(ip=ip,bgr=bgr,cmx=cmx,v11=v11,ef=ef,gt=gt,wins=wins))
    print(f"      GT={len(gt)}, CMX={len(cmx)}, V11={len(v11)}, EF={len(ef)}, wins={len(wins)}")
print()

COL_TITLES=['Ground Truth', 'YOLOv11s\n(R/G/NIR)', 'EarlyFusion\n(4ch)', 'Proposed\n(Dual-stream)']
DISP_H=480

nrows=len(scene_data)
fig,axes=plt.subplots(nrows,4,figsize=(14.5,4.4*nrows))
fig.subplots_adjust(wspace=0.03,hspace=0.08,left=0.09,right=0.99,top=0.94,bottom=0.03)

for col,t in enumerate(COL_TITLES):
    axes[0][col].set_title(t,fontsize=9.5,fontweight='bold',pad=5)

for row,d in enumerate(scene_data):
    bgr=d['bgr']; gt=d['gt']; cmx=d['cmx']; v11=d['v11']; ef=d['ef']; wins=d['wins']
    H,W=bgr.shape[:2]

    roi=build_roi(wins,gt,W,H,max_fruits=4,pad_ratio=0.55)
    x1r,y1r,x2r,y2r=roi

    gt_in =[b for b in gt  if x1r-80<=(b[1]+b[3])//2<=x2r+80 and y1r-80<=(b[2]+b[4])//2<=y2r+80]
    cmx_in=clip_to_roi(cmx,x1r,y1r,x2r,y2r)
    v11_in=clip_to_roi(v11,x1r,y1r,x2r,y2r)
    ef_in =clip_to_roi(ef, x1r,y1r,x2r,y2r)

    imgs_rgb=[
        draw_gt_cv(bgr,  gt_in,  lw=8,fs=0.68),
        draw_pred_cv(bgr,v11_in, lw=8,fs=0.68),
        draw_pred_cv(bgr,ef_in,  lw=8,fs=0.68),
        draw_pred_cv(bgr,cmx_in, lw=8,fs=0.68),
    ]

    panels=[]
    for im in imgs_rgb:
        crop=im[y1r:y2r,x1r:x2r]
        if crop.size==0: crop=im
        h,w=crop.shape[:2]; nw=int(w*DISP_H/max(h,1))
        panels.append(cv2.resize(crop,(nw,DISP_H),interpolation=cv2.INTER_AREA))

    max_w=max(p.shape[1] for p in panels)
    panels=[pad_w(p,max_w) for p in panels]

    for col,panel in enumerate(panels):
        axes[row][col].imshow(panel)
        axes[row][col].set_xticks([]); axes[row][col].set_yticks([])

    # Row label = dominant class of WIN boxes in this crop
    win_in_roi=[b for b in wins if x1r-80<=(b[1]+b[3])//2<=x2r+80
                and y1r-80<=(b[2]+b[4])//2<=y2r+80]
    if win_in_roi:
        dom_win=Counter(b[0] for b in win_in_roi).most_common(1)[0][0]
    elif gt_in:
        dom_win=Counter(b[0] for b in gt_in).most_common(1)[0][0]
    else:
        dom_win=Counter(b[0] for b in d['gt']).most_common(1)[0][0]

    axes[row][0].set_ylabel(CLASS_NAMES[dom_win],fontsize=9.5,fontweight='bold',
                             rotation=90,va='center',labelpad=8)

# Legend
handles=[mpatches.Patch(color='white',label='GT box',ec='gray')]
for i in range(4):
    handles.append(mpatches.Patch(color=BOX_RGB_HEX[i],label=CLASS_NAMES[i]))
fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=8.5,
           frameon=True,bbox_to_anchor=(0.55,-0.01))

# NO suptitle
save_fig(fig,'fig_comparison')
print("  Done.\n")

print("="*55)
for name in ['fig_dataset_samples','fig_comparison']:
    for ext in ['pdf','svg']:
        p=os.path.join(OUT_DIR,f'{name}.{ext}')
        kb=os.path.getsize(p)//1024 if os.path.exists(p) else 0
        print(f"  {name}.{ext:3s}  {kb:6d} KB")
