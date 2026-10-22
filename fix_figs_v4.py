#!/usr/bin/env python3
"""
fix_figs_v4.py
─────────────────────────────────────────────────────────────
Regenerates:
  fig_dataset_samples.(pdf|svg)  — 2×2, cluster crops, enhanced, no suptitle
  fig_comparison.(pdf|svg)       — 2×4, enhanced, dashed GT, thick pred, circle highlights

Key changes vs v3:
  • Percentile stretch 2–98% per channel (images look less purple/dark)
  • GT: thin DASHED WHITE box; Pred: thick solid HIGH-CONTRAST colored box
  • Labels: semi-transparent black background, white text
  • dataset_samples: one dense cluster per stage (3–6 fruits), 2×2 layout
  • comparison: 2 rows × 4 cols, attention circle marks the key win area
"""
import os, sys, glob
from pathlib import Path
from collections import Counter

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

# High-contrast prediction box colors (BGR) on purple false-color background
PRED_BGR = [
    (  0, 255, 255),   # Pre-pol  → YELLOW  (very visible on purple)
    (255, 255,   0),   # Post-pol → CYAN    (visible on purple)
    (  0, 100, 255),   # Net form.→ ORANGE  (warm complement to purple)
    (  0, 255,   0),   # Sweetening → LIME  (bg is false-color, not natural green)
]
PRED_RGB_HEX = ['#FFFF00', '#00FFFF', '#FF6400', '#00FF00']

GT_BGR    = (255, 255, 255)   # GT: dashed WHITE, thin
CIRC_BGR  = (0,   255, 255)   # Attention circle: YELLOW (BGR)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.titlesize':10,'axes.labelsize':9,
                     'xtick.labelsize':8,'ytick.labelsize':8})


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def enhance(bgr, p_lo=2, p_hi=98):
    """Per-channel percentile stretch → more vivid false-color display."""
    out = np.empty_like(bgr)
    for c in range(3):
        ch = bgr[:,:,c].astype(np.float32)
        lo = np.percentile(ch, p_lo); hi = np.percentile(ch, p_hi)
        if hi > lo:
            out[:,:,c] = np.clip((ch-lo)/(hi-lo)*255, 0, 255).astype(np.uint8)
        else:
            out[:,:,c] = ch.astype(np.uint8)
    return out

def enhance_light(bgr, p_lo=5, p_hi=95, gamma=0.85):
    """Lighter stretch (5-95%) + gamma — less noise amplification than 2-98%."""
    out = np.empty((bgr.shape[0], bgr.shape[1], 3), dtype=np.float32)
    for c in range(3):
        ch = bgr[:,:,c].astype(np.float32)
        lo = np.percentile(ch, p_lo); hi = np.percentile(ch, p_hi)
        if hi > lo:
            out[:,:,c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0)
        else:
            out[:,:,c] = ch / 255.0
    out = np.power(np.clip(out, 0.0, 1.0), gamma) * 255.0
    return out.astype(np.uint8)

def draw_dashed_rect(vis, x1, y1, x2, y2, color, lw=3, dash=18, gap=10):
    """Draw a dashed rectangle (four sides)."""
    def dline(p1, p2):
        ax,ay=p1; bx,by=p2
        dist=max(int(np.hypot(bx-ax,by-ay)),1)
        dx=(bx-ax)/dist; dy=(by-ay)/dist
        d=0; on=True
        while d<dist:
            de=min(d+(dash if on else gap),dist)
            if on:
                cv2.line(vis,(int(ax+d*dx),int(ay+d*dy)),
                         (int(ax+de*dx),int(ay+de*dy)),color,lw,cv2.LINE_AA)
            d=de; on=not on
    dline((x1,y1),(x2,y1)); dline((x2,y1),(x2,y2))
    dline((x2,y2),(x1,y2)); dline((x1,y2),(x1,y1))

def put_label(vis, text, x, y, color_bg, scale=0.65, lw=2):
    """Text with semi-transparent dark background."""
    font=cv2.FONT_HERSHEY_DUPLEX
    (tw,th),bl=cv2.getTextSize(text,font,scale,lw)
    overlay=vis.copy()
    cv2.rectangle(overlay,(x-2,y-th-6),(x+tw+4,y+bl+2),(0,0,0),-1)
    cv2.addWeighted(overlay,0.65,vis,0.35,0,vis)
    cv2.putText(vis,text,(x,y),font,scale,(255,255,255),lw,cv2.LINE_AA)

def draw_pred_boxes(vis, pred_boxes, lw=8, scale=0.65):
    for (cls,x1,y1,x2,y2,cf) in pred_boxes:
        color=PRED_BGR[cls]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,lw)
        lbl=f'{CLASS_SHORT[cls]} {cf:.2f}'
        put_label(vis,lbl,x1,y1-3,color,scale=scale,lw=2)

def draw_gt_dashed(vis, gt_boxes, lw=3, scale=0.60):
    for (cls,x1,y1,x2,y2) in gt_boxes:
        draw_dashed_rect(vis,x1,y1,x2,y2,GT_BGR,lw=lw)
        put_label(vis,CLASS_SHORT[cls],x1,y1-3,GT_BGR,scale=scale,lw=2)

def clip_to_roi(boxes, x1r,y1r,x2r,y2r, margin=80):
    return [b for b in boxes
            if x1r-margin<=(b[1]+b[3])//2<=x2r+margin
            and y1r-margin<=(b[2]+b[4])//2<=y2r+margin]

def draw_attn_circle(vis, win_box_orig, roi, disp_h, disp_w):
    """Overlay yellow attention circle at the win box location."""
    x1r,y1r,x2r,y2r=roi
    _,bx1,by1,bx2,by2=win_box_orig[:5]
    cw=max(x2r-x1r,1); ch=max(y2r-y1r,1)
    sx=disp_w/cw; sy=disp_h/ch
    cx=int(((bx1+bx2)/2-x1r)*sx)
    cy=int(((by1+by2)/2-y1r)*sy)
    r=int(max((bx2-bx1)*sx,(by2-by1)*sy)/2*1.6)
    r=max(r,30)
    cv2.circle(vis,(cx,cy),r,CIRC_BGR,5,cv2.LINE_AA)
    cv2.circle(vis,(cx,cy),5,CIRC_BGR,-1)

def save_fig(fig, name):
    for ext in ['pdf','svg']:
        p=os.path.join(OUT_DIR,f'{name}.{ext}')
        fig.savefig(p,format=ext,bbox_inches='tight',dpi=300)
        print(f"  Saved: {p}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# PRE-LOAD: read all test images ONCE → cache (H,W, gt_boxes, blur_score)
# ══════════════════════════════════════════════════════════════════════════════
all_test=sorted(glob.glob(os.path.join(TEST_IMG_DIR,'*.jpg')))
print(f"Pre-loading {len(all_test)} test images (single pass) ...")

image_cache = {}   # ip → (H, W, gt_boxes, blur_score)
for i,ip in enumerate(all_test):
    bgr=cv2.imread(ip)
    if bgr is None: continue
    H,W=bgr.shape[:2]
    stem=Path(ip).stem
    lp=os.path.join(RAW_DIR,'labels','test',stem+'.txt')
    boxes=[]
    if os.path.exists(lp):
        for line in open(lp):
            p=line.strip().split()
            if not p: continue
            cls=int(p[0]); cx,cy,bw,bh=float(p[1]),float(p[2]),float(p[3]),float(p[4])
            x1=int((cx-bw/2)*W); y1=int((cy-bh/2)*H)
            x2=int((cx+bw/2)*W); y2=int((cy+bh/2)*H)
            boxes.append((cls,x1,y1,x2,y2))
    gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
    blur=cv2.Laplacian(gray,cv2.CV_64F).var()
    image_cache[ip]=(H,W,boxes,blur)
print(f"  Done. {len(image_cache)} images cached.\n")


def find_dense_cluster(cls_boxes, n=4):
    """Return the n-box cluster of cls_boxes closest to their centroid."""
    if len(cls_boxes)<=n: return cls_boxes
    mcx=np.mean([(b[1]+b[3])/2 for b in cls_boxes])
    mcy=np.mean([(b[2]+b[4])/2 for b in cls_boxes])
    return sorted(cls_boxes,key=lambda b:((b[1]+b[3])/2-mcx)**2+((b[2]+b[4])/2-mcy)**2)[:n]

def pick_best_cluster(cls_target, n_cluster=5, min_n=3, blur_min=100,
                      margin=100, top_k=25, min_avg_box_px=60):
    """Find image+cluster: best LOCAL sharpness × GT annotation coverage density.

    min_avg_box_px: skip clusters where avg(max(w,h)) < this — avoids tiny/distant melons.
    Coverage = sum of same-class GT box areas within tight crop / crop_area.
    """
    candidates = []
    for ip, (H, W, boxes, global_blur) in image_cache.items():
        if global_blur < blur_min:
            continue
        cls_boxes = [b for b in boxes if b[0] == cls_target
                     and b[1] > margin and b[2] > margin
                     and b[3] < W - margin and b[4] < H - margin]
        if len(cls_boxes) < min_n:
            continue
        cluster = find_dense_cluster(cls_boxes, n=n_cluster)
        # Reject clusters of tiny/distant melons
        avg_dim = np.mean([max(b[3]-b[1], b[4]-b[2]) for b in cluster])
        if avg_dim < min_avg_box_px:
            continue
        xs = [b[1] for b in cluster] + [b[3] for b in cluster]
        ys = [b[2] for b in cluster] + [b[4] for b in cluster]
        per_box_area = (max(xs)-min(xs)) * (max(ys)-min(ys)) / max(len(cluster), 1)
        candidates.append((ip, cluster, per_box_area, min(xs), min(ys), max(xs), max(ys)))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[2])  # densest first

    best = None; best_score = -1.0
    for ip, cluster, per_box_area, x1c, y1c, x2c, y2c in candidates[:top_k]:
        H, W, all_boxes, _ = image_cache[ip]

        # Tight crop bounds (mirrors display crop: pad=25% of span, min 400px)
        span = max(x2c - x1c, y2c - y1c)
        tight_pad = max(int(span * 0.25), 40)
        side_px = max(span + 2*tight_pad, 400)
        cx_m = (x1c+x2c)//2; cy_m = (y1c+y2c)//2
        cx1 = max(0, cx_m - side_px//2); cx2 = min(W, cx1 + side_px)
        cy1 = max(0, cy_m - side_px//2); cy2 = min(H, cy1 + side_px)
        crop_area = max((cx2-cx1) * (cy2-cy1), 1)

        # GT coverage: same-class boxes whose center falls in crop
        boxes_in = [b for b in all_boxes if b[0] == cls_target
                    and cx1 <= (b[1]+b[3])//2 <= cx2
                    and cy1 <= (b[2]+b[4])//2 <= cy2]
        box_sum = sum((b[3]-b[1]) * (b[4]-b[2]) for b in boxes_in)
        coverage = box_sum / crop_area

        # Local sharpness on cluster bounding box
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        crop_r = bgr[max(0, y1c):min(H, y2c), max(0, x1c):min(W, x2c)]
        if crop_r.size == 0:
            continue
        gray = cv2.cvtColor(crop_r, cv2.COLOR_BGR2GRAY)
        local_lap = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Combined score: sharpness × coverage bonus / density penalty
        score = local_lap * (1.0 + coverage * 5.0) / max(per_box_area ** 0.5, 1.0)
        if score > best_score:
            best_score = score; best = (ip, cluster)

    return best


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — DATASET SAMPLES  2×2  (one dense cluster per stage)
# ══════════════════════════════════════════════════════════════════════════════
print("[1/2] fig_dataset_samples ...")

CELL_H=520   # display height per cell

best_clusters={}
for cls in range(4):
    # Progressive fallback: (blur_min, n_cluster, min_n, min_avg_box_px)
    result = None
    for bmin, nc, mn, mbp in [(150,5,3,80),(100,5,3,60),(80,5,3,40),(60,5,3,30),(30,3,2,0)]:
        result = pick_best_cluster(cls, n_cluster=nc, min_n=mn,
                                   blur_min=bmin, min_avg_box_px=mbp)
        if result is not None:
            break
    best_clusters[cls] = result
    if result:
        ip, cluster = result
        _,_,_,gblur = image_cache[ip]
        avg_dim = np.mean([max(b[3]-b[1], b[4]-b[2]) for b in cluster])
        print(f"  cls{cls} ({CLASS_NAMES[cls]}): {os.path.basename(ip)}, "
              f"{len(cluster)} boxes, blur={gblur:.0f}, avg_box={avg_dim:.0f}px")
    else:
        print(f"  cls{cls}: NOT FOUND")

# 2×2 grid: top row cls0,cls1 / bottom row cls2,cls3
fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.5))
fig.subplots_adjust(wspace=0.04, hspace=0.08, left=0.01, right=0.99,
                    top=0.93, bottom=0.01)

ORDER = [(0,0,0),(0,1,1),(1,0,2),(1,1,3)]  # (grid_row, grid_col, cls)
for gr, gc, cls in ORDER:
    ax=axes[gr][gc]
    result=best_clusters.get(cls)
    if result is None:
        ax.set_facecolor('#111'); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(CLASS_NAMES[cls],fontsize=10,fontweight='bold',
                     color=PRED_RGB_HEX[cls],pad=5)
        continue
    ip,cluster=result
    H,W,all_gt,_=image_cache[ip]
    bgr=cv2.imread(ip)

    # Tight square crop centered on cluster
    xs=[b[1] for b in cluster]+[b[3] for b in cluster]
    ys=[b[2] for b in cluster]+[b[4] for b in cluster]
    span_w=max(xs)-min(xs); span_h=max(ys)-min(ys)
    pad=max(int(max(span_w,span_h)*0.25), 40)
    # Soft floor: ensure crop >= 400px to limit upscale ratio to <1.3×
    side=max(max(span_w,span_h) + 2*pad, 400)
    cx_mid=(min(xs)+max(xs))//2; cy_mid=(min(ys)+max(ys))//2
    c1=max(0,cx_mid-side//2); c2=min(W,c1+side)
    r1=max(0,cy_mid-side//2); r2=min(H,r1+side)
    side=c2-c1  # actual after image-boundary clamp

    crop=bgr[r1:r2,c1:c2].copy()
    crop_en=enhance_light(crop)

    # Draw ALL same-class GT boxes that overlap the crop (any overlap, not just center-in)
    # This ensures every labeled melon visible in the panel gets a box.
    box_lw = max(5, int(side * 0.013))
    for (bc,bx1,by1,bx2,by2) in all_gt:
        if bc != cls: continue
        # Keep if box overlaps crop rectangle at all
        if bx2 <= c1 or bx1 >= c2 or by2 <= r1 or by1 >= r2: continue
        nx1=max(0,bx1-c1); ny1=max(0,by1-r1)
        nx2=min(side,bx2-c1); ny2=min(r2-r1,by2-r1)
        if nx2>nx1 and ny2>ny1:
            cv2.rectangle(crop_en,(nx1,ny1),(nx2,ny2),PRED_BGR[cls],box_lw)

    crop_rgb=cv2.cvtColor(crop_en,cv2.COLOR_BGR2RGB)
    # INTER_AREA when downscaling (side>CELL_H), INTER_CUBIC for mild upscale
    interp=cv2.INTER_AREA if side >= CELL_H else cv2.INTER_CUBIC
    disp=cv2.resize(crop_rgb,(CELL_H,CELL_H),interpolation=interp)

    ax.imshow(disp)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(CLASS_NAMES[cls],fontsize=10,fontweight='bold',
                 color=PRED_RGB_HEX[cls],pad=5)
    for sp in ax.spines.values():
        sp.set_edgecolor(PRED_RGB_HEX[cls]); sp.set_linewidth(2.5)

# NO suptitle
save_fig(fig,'fig_dataset_samples')
print("  Done.\n")


# ══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS (needed only for comparison)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading models ...")
from ultralytics import YOLO
m_cmx=YOLO(MODEL_CMX); m_v11=YOLO(MODEL_V11); m_ef=YOLO(MODEL_EF)
print("  Done.\n")

def load_4ch(ip):
    bgr=cv2.imread(ip)
    if bgr is None: return None,None
    fname=Path(ip).stem+'.png'; ndvi=None
    for sp in ['test','val','train']:
        p=os.path.join(NDVI_DIR,'images',sp,fname)
        if os.path.exists(p): ndvi=cv2.imread(p,cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir=bgr[:,:,2].astype(np.float32); red=bgr[:,:,1].astype(np.float32)
        ndvi=(((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape!=bgr.shape[:2]: ndvi=cv2.resize(ndvi,(bgr.shape[1],bgr.shape[0]))
    return bgr,np.dstack([bgr,ndvi])

def run_infer(model,ip,is_4ch,conf=0.25,iou=0.6):
    bgr,img4=load_4ch(ip)
    if bgr is None: return None,[]
    res=model.predict(source=img4 if is_4ch else ip,conf=conf,iou=iou,imgsz=1280,verbose=False)
    return bgr,[(int(b.cls[0]),*[int(v) for v in b.xyxy[0].tolist()],float(b.conf[0]))
                for b in res[0].boxes]

def box_iou(b1,b2):
    xi1=max(b1[0],b2[0]);yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]);yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter==0: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — COMPARISON  2 rows × 4 cols  (Post-pol + Net formation)
# ══════════════════════════════════════════════════════════════════════════════
print("[2/2] fig_comparison ...")

# 2 scenes, different dominant class, confirmed CMX wins
COMP_STEMS=[
    '2023_0610_145243_453',   # Post-pollination (9 post + 4 pre), wins=2
    '2023_0620_162254_353',   # Net formation (13), wins=3
]
comp_imgs=[os.path.join(TEST_IMG_DIR,s+'.jpg') for s in COMP_STEMS]
IOU_T=0.5

print("  Running 3 models on 2 scenes ...")
scene_data=[]
for ip in comp_imgs:
    H,W,gt_all,blur=image_cache[ip]
    bgr,cmx=run_infer(m_cmx,ip,is_4ch=True)
    _, v11=run_infer(m_v11,ip,is_4ch=False)
    _,  ef=run_infer(m_ef, ip,is_4ch=True)
    # Find win boxes
    wins=[]
    for gbox in gt_all:
        gc,gx1,gy1,gx2,gy2=gbox
        ci=max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in cmx),default=0)
        vi=max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in v11),default=0)
        if ci>=IOU_T and vi<IOU_T: wins.append(gbox)
    if not wins: wins=gt_all[:2]  # fallback
    scene_data.append(dict(ip=ip,bgr=bgr,gt=gt_all,cmx=cmx,v11=v11,ef=ef,wins=wins))
    dom=Counter(b[0] for b in gt_all).most_common(1)[0][0]
    print(f"    {os.path.basename(ip)}: dominant={CLASS_NAMES[dom]}, wins={len(wins)}, "
          f"GT={len(gt_all)}, CMX={len(cmx)}, V11={len(v11)}, EF={len(ef)}")
print()

COL_TITLES=['Ground Truth','YOLOv11s\n(R/G/NIR)','EarlyFusion\n(4ch)','Proposed\n(Dual-stream)']
DISP_H=500

fig,axes=plt.subplots(2,4,figsize=(15.0,6.0))
fig.subplots_adjust(wspace=0.03,hspace=0.10,left=0.08,right=0.99,top=0.91,bottom=0.04)

for col,t in enumerate(COL_TITLES):
    axes[0][col].set_title(t,fontsize=10,fontweight='bold',pad=5)

for row,d in enumerate(scene_data):
    ip=d['ip']; bgr=d['bgr']; gt=d['gt']
    cmx=d['cmx']; v11=d['v11']; ef=d['ef']; wins=d['wins']
    H,W=bgr.shape[:2]

    # Build ROI centered on wins, covering ~3–5 GT boxes
    wcx=np.mean([(b[1]+b[3])/2 for b in wins]); wcy=np.mean([(b[2]+b[4])/2 for b in wins])
    gt_near=sorted(gt,key=lambda b:((b[1]+b[3])/2-wcx)**2+((b[2]+b[4])/2-wcy)**2)[:5]
    xs=[b[1] for b in gt_near]+[b[3] for b in gt_near]
    ys=[b[2] for b in gt_near]+[b[4] for b in gt_near]
    bw=max(xs)-min(xs); bh=max(ys)-min(ys)
    pad=max(int(max(bw,bh)*0.55),130)
    x1r=max(0,min(xs)-pad); y1r=max(0,min(ys)-pad)
    x2r=min(W,max(xs)+pad); y2r=min(H,max(ys)+pad)
    roi=(x1r,y1r,x2r,y2r)

    # Filter boxes to ROI
    def in_roi(b): return x1r-80<=(b[1]+b[3])//2<=x2r+80 and y1r-80<=(b[2]+b[4])//2<=y2r+80
    gt_in  =[b for b in gt  if in_roi(b)]
    cmx_in =[b for b in cmx if in_roi(b)]
    v11_in =[b for b in v11 if in_roi(b)]
    ef_in  =[b for b in ef  if in_roi(b)]
    wins_in=[b for b in wins if in_roi(b)]

    # Build enhanced base
    crop_base=enhance(bgr[y1r:y2r,x1r:x2r].copy())

    # Primary win box (first win inside ROI, fallback to first win overall)
    primary_win=wins_in[0] if wins_in else (wins[0] if wins else gt_in[0] if gt_in else None)

    panels=[]
    for col_idx,(lbl_boxes,pred_boxes) in enumerate([
        (gt_in,   []),        # GT column
        ([],      v11_in),    # YOLOv11s
        ([],      ef_in),     # EarlyFusion
        ([],      cmx_in),    # Proposed
    ]):
        vis=crop_base.copy()
        # Draw GT (dashed white)
        draw_gt_dashed(vis,lbl_boxes,lw=3,scale=0.60)
        # Draw predictions (thick solid)
        draw_pred_boxes(vis,pred_boxes,lw=9,scale=0.65)
        # Attention circle on win location (all 4 cols)
        vis_rgb=cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)
        # Resize
        ch,cw=vis_rgb.shape[:2]; nw=int(cw*DISP_H/max(ch,1))
        panel=cv2.resize(vis_rgb,(nw,DISP_H),interpolation=cv2.INTER_AREA)
        # Draw circle after resize (in display space)
        if primary_win is not None:
            draw_attn_circle(panel,primary_win,roi,DISP_H,nw)
        panels.append(panel)

    max_w=max(p.shape[1] for p in panels)
    def pad_w(img,tw):
        h,w=img.shape[:2]
        if w>=tw: return img[:,:tw]
        return np.hstack([img,np.full((h,tw-w,3),20,np.uint8)])
    panels=[pad_w(p,max_w) for p in panels]

    for col_idx,panel in enumerate(panels):
        axes[row][col_idx].imshow(panel)
        axes[row][col_idx].set_xticks([]); axes[row][col_idx].set_yticks([])

    # Row label: dominant class of wins inside ROI
    dom_cls=(Counter(b[0] for b in wins_in).most_common(1)[0][0] if wins_in
             else Counter(b[0] for b in gt_in).most_common(1)[0][0] if gt_in
             else Counter(b[0] for b in wins).most_common(1)[0][0])
    axes[row][0].set_ylabel(CLASS_NAMES[dom_cls],fontsize=10,fontweight='bold',
                             rotation=90,va='center',labelpad=8)

# Legend
handles=[mpatches.Patch(color='white',label='GT (dashed white)',ec='gray')]
for i in range(4):
    handles.append(mpatches.Patch(color=PRED_RGB_HEX[i],label=CLASS_NAMES[i]))
handles.append(mpatches.Patch(color='#FFFF00',label='⊙ Win area (circle)'))
fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=8.5,
           frameon=True,bbox_to_anchor=(0.55,-0.01))

save_fig(fig,'fig_comparison')
print("  Done.\n")

print("="*55)
for name in ['fig_dataset_samples','fig_comparison']:
    for ext in ['pdf','svg']:
        p=os.path.join(OUT_DIR,f'{name}.{ext}')
        kb=os.path.getsize(p)//1024 if os.path.exists(p) else -1
        print(f"  {name}.{ext:3s}  {kb:6d} KB")
