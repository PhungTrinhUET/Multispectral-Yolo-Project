#!/usr/bin/env python3
"""
fix_comparison_v5.py  — Regenerate fig_comparison ONLY
FIX A: tight ROI (pad = 1.5× box_size per side, single win box as anchor)
FIX B: coords transformed to crop space before drawing
FIX C: colors = RED / YELLOW / ORANGE / MAGENTA  (no cyan/blue/green)
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

REPO_DIR  = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics'
OUT_DIR   = os.path.join(REPO_DIR, 'paper_figures')
RAW_DIR   = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
NDVI_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi'
MODEL_CMX = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
MODEL_V11 = '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolo11s_1280_balanced_b4/weights/best.pt'
MODEL_EF  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp1_NoPretrain_1280px/weights/best.pt')
TEST_DIR  = os.path.join(RAW_DIR, 'images', 'test')

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

# ── Class / color constants ───────────────────────────────────────────────────
CLASS_NAMES = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
CLASS_SHORT = ['Pre-pol.', 'Post-pol.', 'Net form.', 'Sweetening']

# Drawn on BGR vis → after BGR→RGB conversion:
# (0,255,255)BGR = YELLOW,  (0,0,255)BGR = RED
# (0,128,255)BGR = ORANGE,  (255,0,255)BGR = MAGENTA
PRED_BGR = [(0,255,255), (0,0,255), (0,128,255), (255,0,255)]
PRED_HEX = ['#FFFF00',   '#FF0000',  '#FF8000',   '#FF00FF']
GT_BGR   = (255,255,255)   # dashed white (stays white after BGR→RGB)
PRED_LW  = 10              # thick prediction border
GT_LW    = 2               # thin dashed GT border
DISP_H   = 520             # panel height in pixels

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.titlesize':10,'axes.labelsize':9})


# ── Drawing on BGR crop (coordinates already in crop space) ──────────────────
def enhance(bgr, lo=2, hi=98):
    out=np.empty_like(bgr)
    for c in range(3):
        ch=bgr[:,:,c].astype(np.float32)
        a,b=np.percentile(ch,lo),np.percentile(ch,hi)
        out[:,:,c]=np.clip((ch-a)/max(b-a,1)*255,0,255).astype(np.uint8)
    return out

def dashed_line(vis, ax,ay,bx,by, color, lw, dash=14, gap=8):
    dist=max(int(np.hypot(bx-ax,by-ay)),1)
    dx,dy=(bx-ax)/dist,(by-ay)/dist
    d=0; on=True
    while d<dist:
        de=min(d+(dash if on else gap),dist)
        if on:
            cv2.line(vis,(int(ax+d*dx),int(ay+d*dy)),
                        (int(ax+de*dx),int(ay+de*dy)),color,lw,cv2.LINE_AA)
        d=de; on=not on

def dashed_rect(vis, x1,y1,x2,y2, color, lw=GT_LW):
    dashed_line(vis,x1,y1,x2,y1,color,lw)
    dashed_line(vis,x2,y1,x2,y2,color,lw)
    dashed_line(vis,x2,y2,x1,y2,color,lw)
    dashed_line(vis,x1,y2,x1,y1,color,lw)

def put_label(vis, text, x, y, scale=0.65, lw=2):
    font=cv2.FONT_HERSHEY_DUPLEX
    (tw,th),bl=cv2.getTextSize(text,font,scale,lw)
    ov=vis.copy()
    cv2.rectangle(ov,(x-2,y-th-5),(x+tw+4,y+bl+2),(0,0,0),-1)
    cv2.addWeighted(ov,0.65,vis,0.35,0,vis)
    cv2.putText(vis,text,(x,y),font,scale,(255,255,255),lw,cv2.LINE_AA)

def to_crop(boxes_orig, ox, oy):
    """Shift box coords from original-image space to crop space."""
    result=[]
    for b in boxes_orig:
        if len(b)==5:
            cls,x1,y1,x2,y2=b
            result.append((cls,x1-ox,y1-oy,x2-ox,y2-oy))
        else:
            cls,x1,y1,x2,y2,cf=b
            result.append((cls,x1-ox,y1-oy,x2-ox,y2-oy,cf))
    return result

def draw_pred_crop(vis, pred_boxes_crop):
    for b in pred_boxes_crop:
        cls,x1,y1,x2,y2,cf=b
        cv2.rectangle(vis,(x1,y1),(x2,y2),PRED_BGR[cls],PRED_LW)
        put_label(vis,f'{CLASS_SHORT[cls]} {cf:.2f}',x1,max(y1-3,18),scale=0.65,lw=2)

def draw_gt_crop(vis, gt_boxes_crop):
    for b in gt_boxes_crop:
        cls,x1,y1,x2,y2=b
        dashed_rect(vis,x1,y1,x2,y2,GT_BGR,lw=GT_LW)
        put_label(vis,CLASS_SHORT[cls],x1,max(y1-3,18),scale=0.60,lw=1)

def add_circle(panel_rgb, win_orig, x1r,y1r, crop_w,crop_h):
    """Draw yellow attention circle on panel_rgb (RGB, already resized)."""
    _,bx1,by1,bx2,by2=win_orig[:5]
    ph,pw=panel_rgb.shape[:2]
    sx=pw/max(crop_w,1); sy=ph/max(crop_h,1)
    cx=int(((bx1+bx2)/2-x1r)*sx)
    cy=int(((by1+by2)/2-y1r)*sy)
    r=int(max((bx2-bx1)*sx,(by2-by1)*sy)/2*1.7)
    r=max(min(r,pw//3),20)
    cv2.circle(panel_rgb,(cx,cy),r,(255,255,0),4,cv2.LINE_AA)
    cv2.circle(panel_rgb,(cx,cy),5,(255,255,0),-1)

def make_panel(vis_bgr_crop, disp_h):
    h,w=vis_bgr_crop.shape[:2]
    nw=max(int(w*disp_h/max(h,1)),1)
    return cv2.cvtColor(cv2.resize(vis_bgr_crop,(nw,disp_h),cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)


# ── Inference helpers ─────────────────────────────────────────────────────────
from ultralytics import YOLO

def load_4ch(ip):
    bgr=cv2.imread(ip)
    if bgr is None: return None,None
    stem=Path(ip).stem+'.png'; ndvi=None
    for sp in ['test','val','train']:
        p=os.path.join(NDVI_DIR,'images',sp,stem)
        if os.path.exists(p): ndvi=cv2.imread(p,cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir=bgr[:,:,2].astype(np.float32); red=bgr[:,:,1].astype(np.float32)
        ndvi=(((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape!=bgr.shape[:2]: ndvi=cv2.resize(ndvi,(bgr.shape[1],bgr.shape[0]))
    return bgr,np.dstack([bgr,ndvi])

def run_infer(model, ip, is_4ch, conf=0.25, iou=0.6):
    bgr,img4=load_4ch(ip)
    if bgr is None: return None,[]
    res=model.predict(source=img4 if is_4ch else ip,conf=conf,iou=iou,imgsz=1280,verbose=False)
    return bgr,[(int(b.cls[0]),int(b.xyxy[0][0]),int(b.xyxy[0][1]),
                               int(b.xyxy[0][2]),int(b.xyxy[0][3]),float(b.conf[0]))
                for b in res[0].boxes]

def load_gt(ip):
    bgr=cv2.imread(ip); H,W=bgr.shape[:2]
    lp=os.path.join(RAW_DIR,'labels','test',Path(ip).stem+'.txt')
    boxes=[]
    if os.path.exists(lp):
        for line in open(lp):
            p=line.strip().split()
            if not p: continue
            c=int(p[0]); cx,cy,bw,bh=float(p[1]),float(p[2]),float(p[3]),float(p[4])
            boxes.append((c,int((cx-bw/2)*W),int((cy-bh/2)*H),
                            int((cx+bw/2)*W),int((cy+bh/2)*H)))
    return bgr,boxes

def iou(b1,b2):
    xi1,yi1=max(b1[0],b2[0]),max(b1[1],b2[1])
    xi2,yi2=min(b1[2],b2[2]),min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if not inter: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)

def in_roi(box, x1r,y1r,x2r,y2r, m=60):
    cx=(box[1]+box[3])/2; cy=(box[2]+box[4])/2
    return x1r-m<=cx<=x2r+m and y1r-m<=cy<=y2r+m


# ══════════════════════════════════════════════════════════════════════════════
print("Loading models ...")
m_cmx=YOLO(MODEL_CMX); m_v11=YOLO(MODEL_V11); m_ef=YOLO(MODEL_EF)
print("Done.\n")

COMP_STEMS = ['2023_0610_145243_453', '2023_0620_162254_353']
IOU_T = 0.5

print("Running inference ...")
scenes=[]
for stem in COMP_STEMS:
    ip=os.path.join(TEST_DIR,stem+'.jpg')
    bgr,gt=load_gt(ip)
    H,W=bgr.shape[:2]
    _,cmx=run_infer(m_cmx,ip,True)
    _,v11=run_infer(m_v11,ip,False)
    _,ef =run_infer(m_ef, ip,True)
    wins=[]
    for gb in gt:
        _,gx1,gy1,gx2,gy2=gb
        ci=max((iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in cmx),default=0)
        vi=max((iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in v11),default=0)
        if ci>=IOU_T and vi<IOU_T: wins.append(gb)
    if not wins: wins=gt[:1]
    print(f"  {stem}.jpg: GT={len(gt)} CMX={len(cmx)} V11={len(v11)} wins={len(wins)}")
    # Print win box coords for debug
    for w in wins:
        cls,x1,y1,x2,y2=w
        print(f"    win[{cls}]: ({x1},{y1})→({x2},{y2})  size={x2-x1}×{y2-y1}px")
    scenes.append(dict(ip=ip,bgr=bgr,gt=gt,cmx=cmx,v11=v11,ef=ef,wins=wins,H=H,W=W))
print()

# ── Build ROI around FIRST (best) win box only — keeps ROI tight ─────────────
def build_roi(win_box, H, W, pad_factor=1.8):
    _,x1,y1,x2,y2=win_box[:5]
    bw,bh=x2-x1,y2-y1
    pad=max(int(max(bw,bh)*pad_factor),150)
    cx,cy=(x1+x2)//2,(y1+y2)//2
    # Square crop centered on box
    half=max(bw,bh)//2+pad
    x1r=max(0,cx-half); x2r=min(W,cx+half)
    y1r=max(0,cy-half); y2r=min(H,cy+half)
    print(f"    ROI: ({x1r},{y1r})→({x2r},{y2r})  [{x2r-x1r}×{y2r-y1r}px]  "
          f"box={bw}×{bh}px  pad={pad}px")
    return x1r,y1r,x2r,y2r

# ── Generate figure ───────────────────────────────────────────────────────────
COL_TITLES=['Ground Truth','YOLOv11s\n(RGB only)','EarlyFusion\n(4-ch)','Proposed\n(Dual-stream)']

fig,axes=plt.subplots(2,4,figsize=(16,7.8))
fig.subplots_adjust(wspace=0.025,hspace=0.10,left=0.09,right=0.995,top=0.91,bottom=0.06)

for col,t in enumerate(COL_TITLES):
    axes[0][col].set_title(t,fontsize=10,fontweight='bold',pad=5)

for row,d in enumerate(scenes):
    bgr=d['bgr']; gt=d['gt']; cmx=d['cmx']; v11=d['v11']; ef=d['ef']
    wins=d['wins']; H=d['H']; W=d['W']

    # ROI around first win box
    pw=wins[0]
    x1r,y1r,x2r,y2r=build_roi(pw,H,W,pad_factor=1.8)
    cw,ch=x2r-x1r,y2r-y1r

    # Enhanced crop (BGR), draw BOTH GT and preds in CROP coordinate space
    base_en=enhance(bgr[y1r:y2r,x1r:x2r].copy())

    # Filter boxes to ROI (using original coords for filtering)
    gt_in  =[b for b in gt  if in_roi(b,x1r,y1r,x2r,y2r)]
    cmx_in =[b for b in cmx if in_roi(b,x1r,y1r,x2r,y2r)]
    v11_in =[b for b in v11 if in_roi(b,x1r,y1r,x2r,y2r)]
    ef_in  =[b for b in ef  if in_roi(b,x1r,y1r,x2r,y2r)]

    print(f"  Row {row}: GT_in={len(gt_in)} CMX_in={len(cmx_in)} V11_in={len(v11_in)} "
          f"EF_in={len(ef_in)}  display_zoom={DISP_H/max(ch,1):.2f}×")

    # Shift coords from original → crop space
    gt_c  =to_crop(gt_in, x1r,y1r)
    cmx_c =to_crop(cmx_in,x1r,y1r)
    v11_c =to_crop(v11_in,x1r,y1r)
    ef_c  =to_crop(ef_in, x1r,y1r)

    # Build 4 panels
    col_specs=[(gt_c,[]), ([],v11_c), ([],ef_c), ([],cmx_c)]
    panels=[]
    for lbl,pred in col_specs:
        vis=base_en.copy()
        draw_gt_crop(vis,lbl)
        draw_pred_crop(vis,pred)
        panel=make_panel(vis,DISP_H)
        add_circle(panel,pw,x1r,y1r,cw,ch)
        panels.append(panel)

    # Uniform width
    mw=max(p.shape[1] for p in panels)
    def pad_w(p):
        h,w=p.shape[:2]
        if w>=mw: return p[:,:mw]
        return np.hstack([p,np.full((h,mw-w,3),20,np.uint8)])
    panels=[pad_w(p) for p in panels]

    for col_idx,panel in enumerate(panels):
        ax=axes[row][col_idx]
        ax.imshow(panel); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('white'); sp.set_linewidth(1.0)

    dom=Counter(b[0] for b in wins).most_common(1)[0][0]
    axes[row][0].set_ylabel(CLASS_NAMES[dom],fontsize=10,fontweight='bold',
                             rotation=90,va='center',labelpad=8)

# Legend
handles=[mpatches.Patch(facecolor='white',edgecolor='gray',label='GT (white dashed)')]
for i in range(4):
    handles.append(mpatches.Patch(color=PRED_HEX[i],label=CLASS_NAMES[i]))
handles.append(mpatches.Patch(color='#FFFF00',label='○ CMX-win region'))
fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=8.5,
           frameon=True,bbox_to_anchor=(0.55,0.0))

for ext in ['pdf','svg']:
    p=os.path.join(OUT_DIR,f'fig_comparison.{ext}')
    fig.savefig(p,format=ext,bbox_inches='tight',dpi=300)
    print(f"Saved: {p}  ({os.path.getsize(p)//1024} KB)")
plt.close(fig)
print("Done.")
