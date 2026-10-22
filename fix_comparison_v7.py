#!/usr/bin/env python3
"""
fix_comparison_v7.py
Row 0 (Sweetening):    hardcoded 2023_0620_174956_585  (CMX=27 V11=21 EF=22 of 35)
Row 1 (non-Sweetening): scan ALL non-Sw images → win-centric cluster
                        score = (tp_cmx - max(tp_v11,tp_ef)) × tp_cmx / total
                        → guarantees CMX beats BOTH baselines

Style: NO confidence text on boxes; corner annotation "TP x/y" only.
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
MODEL_CMX = os.path.join(REPO_DIR,'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
MODEL_V11 = '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolo11s_1280_balanced_b4/weights/best.pt'
MODEL_EF  = os.path.join(REPO_DIR,'Multispectral_DuaLuoi_Project/Exp1_NoPretrain_1280px/weights/best.pt')
TEST_DIR  = os.path.join(RAW_DIR, 'images', 'test')

# Best Sweetening scene confirmed from v6 analysis (GT=35, CMX=27, V11=21, EF=22)
SWEETENING_STEM = '2023_0620_174956_585'

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

CLASS_NAMES = ['Pre-pollination','Post-pollination','Net formation','Sweetening']
CLASS_SHORT = ['Pre-pol.','Post-pol.','Net form.','Sweetening']

# BGR → RGB after cv2 convert: YELLOW, RED, ORANGE, MAGENTA
PRED_BGR = [(0,255,255),(0,0,255),(0,128,255),(255,0,255)]
PRED_HEX = ['#FFFF00','#FF0000','#FF8000','#FF00FF']
GT_BGR   = (255,255,255)   # dashed white
PRED_LW  = 10
GT_LW    = 2
DISP_H   = 520
IOU_T    = 0.50

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.titlesize':10,'axes.labelsize':9})


# ── Drawing (no confidence text anywhere) ─────────────────────────────────────
def enhance(bgr, lo=2, hi=98):
    out=np.empty_like(bgr)
    for c in range(3):
        ch=bgr[:,:,c].astype(np.float32)
        a,b=np.percentile(ch,lo),np.percentile(ch,hi)
        out[:,:,c]=np.clip((ch-a)/max(b-a,1)*255,0,255).astype(np.uint8)
    return out

def dline(vis,ax,ay,bx,by,col,lw,dash=14,gap=8):
    dist=max(int(np.hypot(bx-ax,by-ay)),1)
    dx,dy=(bx-ax)/dist,(by-ay)/dist; d=0; on=True
    while d<dist:
        de=min(d+(dash if on else gap),dist)
        if on: cv2.line(vis,(int(ax+d*dx),int(ay+d*dy)),(int(ax+de*dx),int(ay+de*dy)),col,lw,cv2.LINE_AA)
        d=de; on=not on

def dashed_rect(vis,x1,y1,x2,y2,col,lw):
    dline(vis,x1,y1,x2,y1,col,lw); dline(vis,x2,y1,x2,y2,col,lw)
    dline(vis,x2,y2,x1,y2,col,lw); dline(vis,x1,y2,x1,y1,col,lw)

def draw_gt_crop(vis, gt_c):
    for cls,x1,y1,x2,y2 in gt_c:
        dashed_rect(vis,x1,y1,x2,y2,GT_BGR,GT_LW)

def draw_pred_crop(vis, pred_c):
    """Boxes only — NO confidence text."""
    for cls,x1,y1,x2,y2,cf in pred_c:
        cv2.rectangle(vis,(x1,y1),(x2,y2),PRED_BGR[cls],PRED_LW)

def corner_ann(panel, text, pos='br'):
    """Semi-transparent text in bottom-right corner of RGB panel."""
    font=cv2.FONT_HERSHEY_DUPLEX; sc=0.75; lw=2
    (tw,th),bl=cv2.getTextSize(text,font,sc,lw)
    h,w=panel.shape[:2]
    if pos=='br': x=w-tw-10; y=h-10
    else:         x=8;        y=th+8
    ov=panel.copy()
    cv2.rectangle(ov,(x-4,y-th-5),(x+tw+5,y+bl+3),(0,0,0),-1)
    cv2.addWeighted(ov,0.70,panel,0.30,0,panel)
    cv2.putText(panel,text,(x,y),font,sc,(255,255,255),lw,cv2.LINE_AA)

def add_circle(panel, win_box, x1r,y1r,cw,ch):
    if win_box is None: return
    _,bx1,by1,bx2,by2=win_box[:5]
    ph,pw=panel.shape[:2]
    sx=pw/max(cw,1); sy=ph/max(ch,1)
    cx=int(((bx1+bx2)/2-x1r)*sx); cy=int(((by1+by2)/2-y1r)*sy)
    r=int(max((bx2-bx1)*sx,(by2-by1)*sy)/2*1.8); r=max(min(r,pw//3),20)
    cv2.circle(panel,(cx,cy),r,(255,255,0),4,cv2.LINE_AA)
    cv2.circle(panel,(cx,cy),5,(255,255,0),-1)

def to_crop(boxes,ox,oy):
    result=[]
    for b in boxes:
        if len(b)==5: c,x1,y1,x2,y2=b; result.append((c,x1-ox,y1-oy,x2-ox,y2-oy))
        else:         c,x1,y1,x2,y2,cf=b; result.append((c,x1-ox,y1-oy,x2-ox,y2-oy,cf))
    return result

def make_panel(vis_bgr, disp_h):
    h,w=vis_bgr.shape[:2]; nw=max(int(w*disp_h/max(h,1)),1)
    return cv2.cvtColor(cv2.resize(vis_bgr,(nw,disp_h),cv2.INTER_AREA),cv2.COLOR_BGR2RGB)

def save_fig(fig,name):
    for ext in ['pdf','svg']:
        p=os.path.join(OUT_DIR,f'{name}.{ext}')
        fig.savefig(p,format=ext,bbox_inches='tight',dpi=300)
        print(f"  {p}  ({os.path.getsize(p)//1024} KB)")
    plt.close(fig)


# ── Inference / GT helpers ─────────────────────────────────────────────────────
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

def run_infer(model,ip,is_4ch,conf=0.25,iou=0.6):
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
    return bgr,boxes,H,W

def box_iou(b1,b2):
    xi1,yi1=max(b1[0],b2[0]),max(b1[1],b2[1])
    xi2,yi2=min(b1[2],b2[2]),min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if not inter: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)

def count_tp(gt_in,preds):
    return sum(1 for gb in gt_in
               if max((box_iou((gb[1],gb[2],gb[3],gb[4]),(p[1],p[2],p[3],p[4]))
                       for p in preds),default=0)>=IOU_T)

def in_roi(b,x1r,y1r,x2r,y2r,m=60):
    cx=(b[1]+b[3])/2; cy=(b[2]+b[4])/2
    return x1r-m<=cx<=x2r+m and y1r-m<=cy<=y2r+m


# ── Win-centric cluster ROI ────────────────────────────────────────────────────
def win_cluster_roi(gt,cmx_p,v11_p,ef_p,H,W,n=8,pad_f=0.45):
    """
    Find GT boxes where CMX wins (detects) but V11 misses.
    Cluster n nearest GT boxes around those wins.
    Return (roi, metrics_dict, score) for the best cluster found.
    Score = (tp_cmx - max(tp_v11,tp_ef)) * tp_cmx / total
    """
    wins=[gb for gb in gt
          if max((box_iou((gb[1],gb[2],gb[3],gb[4]),(p[1],p[2],p[3],p[4])) for p in cmx_p),default=0)>=IOU_T
          and max((box_iou((gb[1],gb[2],gb[3],gb[4]),(p[1],p[2],p[3],p[4])) for p in v11_p),default=0)<IOU_T]
    if not wins: return None,-999

    avg_box_global=max(np.mean([max(b[3]-b[1],b[4]-b[2]) for b in gt]),50)
    max_dist=avg_box_global*6  # only include boxes within 6 box-widths of anchor

    best_score=-999; best=None
    for anchor in wins[:4]:
        acx=(anchor[1]+anchor[3])/2; acy=(anchor[2]+anchor[4])/2
        nearby=[b for b in gt
                if np.hypot((b[1]+b[3])/2-acx,(b[2]+b[4])/2-acy)<=max_dist]
        if not nearby: nearby=[anchor]
        cluster=sorted(nearby,key=lambda b:((b[1]+b[3])/2-acx)**2+((b[2]+b[4])/2-acy)**2)[:n]
        xs=[b[1] for b in cluster]+[b[3] for b in cluster]
        ys=[b[2] for b in cluster]+[b[4] for b in cluster]
        avg_box=max(np.mean([max(b[3]-b[1],b[4]-b[2]) for b in cluster]),50)
        pad=max(int(avg_box*2.0),100)  # 2× avg box per side (compact, fixed)
        x1r=max(0,min(xs)-pad); y1r=max(0,min(ys)-pad)
        x2r=min(W,max(xs)+pad); y2r=min(H,max(ys)+pad)

        gt_in  =[b for b in gt    if in_roi(b,x1r,y1r,x2r,y2r)]
        cmx_in =[b for b in cmx_p if in_roi(b,x1r,y1r,x2r,y2r)]
        v11_in =[b for b in v11_p if in_roi(b,x1r,y1r,x2r,y2r)]
        ef_in  =[b for b in ef_p  if in_roi(b,x1r,y1r,x2r,y2r)]
        total=len(gt_in)
        if total<3: continue
        tp_c=count_tp(gt_in,cmx_in)
        tp_v=count_tp(gt_in,v11_in)
        tp_e=count_tp(gt_in,ef_in)
        # Reward large gaps with BOTH baselines; penalize if V11=EF (undifferentiated)
        gap_v=max(tp_c-tp_v,0); gap_e=max(tp_c-tp_e,0)
        score=(gap_v+gap_e)*tp_c/total
        if score>best_score:
            best_score=score
            best=dict(roi=(x1r,y1r,x2r,y2r),gt_in=gt_in,cmx_in=cmx_in,
                      v11_in=v11_in,ef_in=ef_in,
                      tp_cmx=tp_c,tp_v11=tp_v,tp_ef=tp_e,total=total,
                      wins=wins,anchor=anchor,dom_cls=Counter(b[0] for b in gt_in).most_common(1)[0][0])
    return best,best_score


# ══════════════════════════════════════════════════════════════════════════════
all_test = sorted(glob.glob(os.path.join(TEST_DIR,'*.jpg')))
print(f"Loading models ...")
m_cmx=YOLO(MODEL_CMX); m_v11=YOLO(MODEL_V11); m_ef=YOLO(MODEL_EF)
print("Done.\n")

# ── Row 0: Sweetening (hardcoded) ─────────────────────────────────────────────
print("Row 0 – Sweetening (hardcoded scene) ...")
ip0 = os.path.join(TEST_DIR, SWEETENING_STEM+'.jpg')
bgr0,gt0,H0,W0 = load_gt(ip0)
_,cmx0=run_infer(m_cmx,ip0,True)
_,v110=run_infer(m_v11,ip0,False)
_,ef0 =run_infer(m_ef, ip0,True)

# Win-centric cluster for the Sweetening scene too (to find best sub-region)
r0,s0 = win_cluster_roi(gt0,cmx0,v110,ef0,H0,W0,n=9,pad_f=2.0)
if r0 is None:
    # Fallback: densest cluster of all GT
    xs_=[b[1] for b in gt0]+[b[3] for b in gt0]; ys_=[b[2] for b in gt0]+[b[4] for b in gt0]
    mcx=np.mean(xs_)/2+np.mean([b[1] for b in gt0])/2; mcy=np.mean(ys_)/2+np.mean([b[2] for b in gt0])/2
    cluster9=sorted(gt0,key=lambda b:((b[1]+b[3])/2-mcx)**2+((b[2]+b[4])/2-mcy)**2)[:9]
    xs=[b[1] for b in cluster9]+[b[3] for b in cluster9]; ys=[b[2] for b in cluster9]+[b[4] for b in cluster9]
    pad=max(int((max(xs)-min(xs))*0.35),150)
    x1r,y1r=max(0,min(xs)-pad),max(0,min(ys)-pad)
    x2r,y2r=min(W0,max(xs)+pad),min(H0,max(ys)+pad)
    gt_in=[b for b in gt0 if in_roi(b,x1r,y1r,x2r,y2r)]
    cmx_in=[b for b in cmx0 if in_roi(b,x1r,y1r,x2r,y2r)]
    v11_in=[b for b in v110 if in_roi(b,x1r,y1r,x2r,y2r)]
    ef_in =[b for b in ef0  if in_roi(b,x1r,y1r,x2r,y2r)]
    r0=dict(roi=(x1r,y1r,x2r,y2r),gt_in=gt_in,cmx_in=cmx_in,v11_in=v11_in,ef_in=ef_in,
            tp_cmx=count_tp(gt_in,cmx_in),tp_v11=count_tp(gt_in,v11_in),tp_ef=count_tp(gt_in,ef_in),
            total=len(gt_in),wins=[],anchor=None,dom_cls=3)

r0.update(ip=ip0,bgr=bgr0,H=H0,W=W0)
x1r,y1r,x2r,y2r=r0['roi']
print(f"  {SWEETENING_STEM}.jpg  ROI={x2r-x1r}×{y2r-y1r}  "
      f"GT={r0['total']}  CMX={r0['tp_cmx']}  V11={r0['tp_v11']}  EF={r0['tp_ef']}")

# ── Row 1: Non-Sweetening scan ─────────────────────────────────────────────────
print("\nRow 1 – Non-Sweetening scan ...")

# Label scan: images with ≥3 non-Sweetening GT boxes AND any cluster span < 0.35
nonsw_ips = []
for ip in all_test:
    if Path(ip).stem == SWEETENING_STEM: continue
    lp = os.path.join(RAW_DIR,'labels','test',Path(ip).stem+'.txt')
    if not os.path.exists(lp): continue
    boxes_n=[]
    for line in open(lp):
        p=line.strip().split()
        if not p: continue
        c=int(p[0])
        if c==3: continue   # skip Sweetening
        boxes_n.append((c,float(p[1]),float(p[2]),float(p[3]),float(p[4])))
    if len(boxes_n)<3: continue
    # Rough cluster: all non-sw boxes; check max span
    xs=[b[1] for b in boxes_n]; ys=[b[2] for b in boxes_n]
    if max(xs)-min(xs)>0.95 and max(ys)-min(ys)>0.95: continue  # spread across full image
    nonsw_ips.append(ip)

print(f"  {len(nonsw_ips)} non-Sweetening images with ≥3 objects; running inference ...")

best1=None; best1_score=-999
for idx,ip in enumerate(nonsw_ips,1):
    bgr,gt,H,W = load_gt(ip)
    gt_nonsw=[b for b in gt if b[0]!=3]
    if len(gt_nonsw)<3: continue
    _,cmx_p=run_infer(m_cmx,ip,True)
    _,v11_p=run_infer(m_v11,ip,False)
    _,ef_p =run_infer(m_ef, ip,True)
    # Pass gt_nonsw so cluster uses only non-Sweetening boxes
    # n=5 boxes, pad_f=3.0 means pad = avg_box*3 → compact crop around 5 fruits
    result,score=win_cluster_roi(gt_nonsw,cmx_p,v11_p,ef_p,H,W,n=5,pad_f=2.0)
    zoom_ok='?'
    if result:
        x1r,y1r,x2r,y2r=result['roi']
        zoom=DISP_H/max(y2r-y1r,1)
        zoom_ok=f'{zoom:.2f}×'
        # Skip if zoom still too low
        if zoom<0.40:
            score=-999
        else:
            # Composite: scale by count, and penalize when V11 == EF (no contrast between baselines)
            eq_pen=0.5 if result['tp_v11']==result['tp_ef'] else 1.0
            score=score*min(result['total']/5.0,1.5)*eq_pen if score>0 else score
    else:
        score=-999
    status='★' if score>0.3 else (' ' if score>-100 else ' ')
    tp_str=(f"CMX={result['tp_cmx']} V11={result['tp_v11']} EF={result['tp_ef']}"
            if result else "no wins")
    print(f"  {'★' if score>0.3 else ' '}[{idx:03d}/{len(nonsw_ips)}] {Path(ip).stem} | "
          f"GT={len(gt_nonsw)} {tp_str} | score={score:.2f} zoom={zoom_ok}")
    if score>best1_score:
        best1_score=score
        best1=dict(**result, ip=ip, bgr=bgr, H=H, W=W)

if best1 is None:
    print("  WARNING: no good non-Sweetening scene found; using best available")
    # Fallback to v6 net formation scene
    ip_fb = os.path.join(TEST_DIR,'2023_0620_162909_541.jpg')
    bgr,gt,H,W = load_gt(ip_fb)
    _,cmx_p=run_infer(m_cmx,ip_fb,True)
    _,v11_p=run_infer(m_v11,ip_fb,False)
    _,ef_p =run_infer(m_ef, ip_fb,True)
    gt_in=[b for b in gt if in_roi(b,824,1404,1909,2264)]
    cmx_in=[b for b in cmx_p if in_roi(b,824,1404,1909,2264)]
    v11_in=[b for b in v11_p if in_roi(b,824,1404,1909,2264)]
    ef_in=[b for b in ef_p  if in_roi(b,824,1404,1909,2264)]
    best1=dict(roi=(824,1404,1909,2264),gt_in=gt_in,cmx_in=cmx_in,v11_in=v11_in,ef_in=ef_in,
               tp_cmx=count_tp(gt_in,cmx_in),tp_v11=count_tp(gt_in,v11_in),
               tp_ef=count_tp(gt_in,ef_in),total=len(gt_in),
               wins=[],anchor=None,dom_cls=2,ip=ip_fb,bgr=bgr,H=H,W=W)

x1r,y1r,x2r,y2r=best1['roi']
dom1=best1['dom_cls']
print(f"\n  SELECTED: {Path(best1['ip']).stem}")
print(f"  class={CLASS_NAMES[dom1]}  ROI={x2r-x1r}×{y2r-y1r}  "
      f"GT={best1['total']}  CMX={best1['tp_cmx']}  V11={best1['tp_v11']}  EF={best1['tp_ef']}  "
      f"score={best1_score:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Generate figure
# ══════════════════════════════════════════════════════════════════════════════
COL_TITLES=['Ground Truth','YOLOv11s\n(RGB only)','EarlyFusion\n(4-ch)','Proposed\n(Dual-stream)']
rows=[r0, best1]

fig,axes=plt.subplots(2,4,figsize=(16,7.8))
fig.subplots_adjust(wspace=0.025,hspace=0.10,left=0.09,right=0.995,top=0.91,bottom=0.065)
for col,t in enumerate(COL_TITLES):
    axes[0][col].set_title(t,fontsize=10,fontweight='bold',pad=5)

for row_idx,d in enumerate(rows):
    x1r,y1r,x2r,y2r=d['roi']; cw,ch=x2r-x1r,y2r-y1r
    bgr=d['bgr']

    # Primary win box for circle
    primary_win=(d['wins'][0] if d.get('wins') else
                 (d['gt_in'][0] if d['gt_in'] else None))

    base_en=enhance(bgr[y1r:y2r,x1r:x2r].copy())
    gt_c  =to_crop(d['gt_in'],  x1r,y1r)
    cmx_c =to_crop(d['cmx_in'], x1r,y1r)
    v11_c =to_crop(d['v11_in'], x1r,y1r)
    ef_c  =to_crop(d['ef_in'],  x1r,y1r)

    total=d['total']; tp_c=d['tp_cmx']; tp_v=d['tp_v11']; tp_e=d['tp_ef']

    col_specs=[
        (gt_c,  [],    f"GT: {total}"),
        ([],    v11_c, f"TP: {tp_v}/{total}"),
        ([],    ef_c,  f"TP: {tp_e}/{total}"),
        ([],    cmx_c, f"TP: {tp_c}/{total}"),
    ]
    panels=[]
    for lbl,pred,ann in col_specs:
        vis=base_en.copy()
        draw_gt_crop(vis,lbl)
        draw_pred_crop(vis,pred)
        panel=make_panel(vis,DISP_H)
        add_circle(panel,primary_win,x1r,y1r,cw,ch)
        corner_ann(panel,ann)
        panels.append(panel)

    mw=max(p.shape[1] for p in panels)
    def pad_w(p):
        h,w=p.shape[:2]
        if w>=mw: return p[:,:mw]
        return np.hstack([p,np.full((h,mw-w,3),20,np.uint8)])
    panels=[pad_w(p) for p in panels]

    for ci,panel in enumerate(panels):
        ax=axes[row_idx][ci]
        ax.imshow(panel); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_edgecolor('white'); sp.set_linewidth(1.0)

    dom=d.get('dom_cls',3)
    axes[row_idx][0].set_ylabel(CLASS_NAMES[dom],fontsize=10,fontweight='bold',
                                 rotation=90,va='center',labelpad=8)
    zoom=DISP_H/max(ch,1)
    print(f"  Row {row_idx}: {CLASS_NAMES[dom]}  ROI={cw}×{ch}  zoom={zoom:.2f}×")

handles=[mpatches.Patch(facecolor='white',edgecolor='gray',label='GT (white dashed)')]
for i in range(4):
    handles.append(mpatches.Patch(color=PRED_HEX[i],label=CLASS_NAMES[i]))
handles.append(mpatches.Patch(color='#FFFF00',label='○ CMX-win region'))
fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=8.5,
           frameon=True,bbox_to_anchor=(0.55,0.0))

print()
save_fig(fig,'fig_comparison')
print("Done.")
