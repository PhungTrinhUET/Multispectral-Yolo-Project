#!/usr/bin/env python3
"""
fix_comparison_v6.py
Phase 1 : Scan label files (no image reads) → top-30 dense clusters
Phase 2 : Run 3-model inference on top-30 → score by (TP_cmx − TP_v11)×TP_cmx
Phase 3 : Pick 2 best scenes with DIFFERENT dominant class
Phase 4 : Generate 2×4 comparison with TP annotations
Colors  : RED / YELLOW / ORANGE / MAGENTA  (no cyan/blue/green)
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

os.makedirs(OUT_DIR, exist_ok=True)
os.chdir(REPO_DIR); sys.path.insert(0, REPO_DIR)

CLASS_NAMES = ['Pre-pollination','Post-pollination','Net formation','Sweetening']
CLASS_SHORT = ['Pre-pol.','Post-pol.','Net form.','Sweetening']

# BGR colors drawn on BGR vis, then converted BGR→RGB for display:
# (0,255,255)BGR→YELLOW  (0,0,255)BGR→RED  (0,128,255)BGR→ORANGE  (255,0,255)BGR→MAGENTA
PRED_BGR = [(0,255,255),(0,0,255),(0,128,255),(255,0,255)]
PRED_HEX = ['#FFFF00','#FF0000','#FF8000','#FF00FF']
GT_BGR   = (255,255,255)
PRED_LW  = 10
GT_LW    = 2
DISP_H   = 520

# Cluster scan params
N_CLUSTER     = 7    # boxes per cluster
MIN_N         = 4    # minimum cluster size
SPAN_MAX_NORM = 0.30 # max normalized span per side (relaxed for non-Sweetening)
TOP_PER_CLASS = 10   # pick densest N candidates from EACH class
IOU_T         = 0.50 # IoU threshold for TP

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                     'axes.titlesize':10,'axes.labelsize':9})

# ── Drawing helpers (all operate on BGR vis in crop coordinate space) ─────────
def enhance(bgr, lo=2, hi=98):
    out=np.empty_like(bgr)
    for c in range(3):
        ch=bgr[:,:,c].astype(np.float32)
        a,b=np.percentile(ch,lo),np.percentile(ch,hi)
        out[:,:,c]=np.clip((ch-a)/max(b-a,1)*255,0,255).astype(np.uint8)
    return out

def dline(vis,ax,ay,bx,by,color,lw,dash=14,gap=8):
    dist=max(int(np.hypot(bx-ax,by-ay)),1)
    dx,dy=(bx-ax)/dist,(by-ay)/dist; d=0; on=True
    while d<dist:
        de=min(d+(dash if on else gap),dist)
        if on: cv2.line(vis,(int(ax+d*dx),int(ay+d*dy)),(int(ax+de*dx),int(ay+de*dy)),color,lw,cv2.LINE_AA)
        d=de; on=not on

def dashed_rect(vis,x1,y1,x2,y2,color,lw=GT_LW):
    dline(vis,x1,y1,x2,y1,color,lw); dline(vis,x2,y1,x2,y2,color,lw)
    dline(vis,x2,y2,x1,y2,color,lw); dline(vis,x1,y2,x1,y1,color,lw)

def put_label(vis,text,x,y,scale=0.65,lw=2):
    font=cv2.FONT_HERSHEY_DUPLEX
    (tw,th),bl=cv2.getTextSize(text,font,scale,lw)
    ov=vis.copy()
    cv2.rectangle(ov,(x-2,y-th-5),(x+tw+4,y+bl+2),(0,0,0),-1)
    cv2.addWeighted(ov,0.65,vis,0.35,0,vis)
    cv2.putText(vis,text,(x,y),font,scale,(255,255,255),lw,cv2.LINE_AA)

def annotate_count(panel_rgb, label):
    """Bottom-right corner text on already-RGB panel."""
    font=cv2.FONT_HERSHEY_DUPLEX; scale=0.70; lw=2
    (tw,th),bl=cv2.getTextSize(label,font,scale,lw)
    h,w=panel_rgb.shape[:2]
    x=w-tw-10; y=h-8
    ov=panel_rgb.copy()
    cv2.rectangle(ov,(x-4,y-th-5),(x+tw+4,y+bl+3),(0,0,0),-1)
    cv2.addWeighted(ov,0.70,panel_rgb,0.30,0,panel_rgb)
    cv2.putText(panel_rgb,label,(x,y),font,scale,(255,255,255),lw,cv2.LINE_AA)

def to_crop(boxes, ox, oy):
    result=[]
    for b in boxes:
        if len(b)==5: cls,x1,y1,x2,y2=b; result.append((cls,x1-ox,y1-oy,x2-ox,y2-oy))
        else:         cls,x1,y1,x2,y2,cf=b; result.append((cls,x1-ox,y1-oy,x2-ox,y2-oy,cf))
    return result

def draw_gt_crop(vis, gt_c):
    for b in gt_c:
        cls,x1,y1,x2,y2=b
        dashed_rect(vis,x1,y1,x2,y2,GT_BGR,lw=GT_LW)
        put_label(vis,CLASS_SHORT[cls],x1,max(y1-3,20),scale=0.58,lw=1)

def draw_pred_crop(vis, pred_c):
    for b in pred_c:
        cls,x1,y1,x2,y2,cf=b
        cv2.rectangle(vis,(x1,y1),(x2,y2),PRED_BGR[cls],PRED_LW)
        put_label(vis,f'{CLASS_SHORT[cls]} {cf:.2f}',x1,max(y1-3,20),scale=0.60,lw=2)

def add_circle(panel_rgb, win_box_orig, x1r, y1r, cw, ch):
    if win_box_orig is None: return
    _,bx1,by1,bx2,by2=win_box_orig[:5]
    ph,pw=panel_rgb.shape[:2]
    sx=pw/max(cw,1); sy=ph/max(ch,1)
    cx=int(((bx1+bx2)/2-x1r)*sx); cy=int(((by1+by2)/2-y1r)*sy)
    r=int(max((bx2-bx1)*sx,(by2-by1)*sy)/2*1.8)
    r=max(min(r,pw//3),20)
    cv2.circle(panel_rgb,(cx,cy),r,(255,255,0),4,cv2.LINE_AA)
    cv2.circle(panel_rgb,(cx,cy),5,(255,255,0),-1)

def make_panel(vis_bgr_crop, disp_h):
    h,w=vis_bgr_crop.shape[:2]; nw=max(int(w*disp_h/max(h,1)),1)
    return cv2.cvtColor(cv2.resize(vis_bgr_crop,(nw,disp_h),cv2.INTER_AREA),cv2.COLOR_BGR2RGB)

def save_fig(fig, name):
    for ext in ['pdf','svg']:
        p=os.path.join(OUT_DIR,f'{name}.{ext}')
        fig.savefig(p,format=ext,bbox_inches='tight',dpi=300)
        print(f"  Saved {p}  ({os.path.getsize(p)//1024} KB)")
    plt.close(fig)

# ── Inference helpers ──────────────────────────────────────────────────────────
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
    return bgr,boxes,H,W

def box_iou(b1,b2):
    xi1,yi1=max(b1[0],b2[0]),max(b1[1],b2[1])
    xi2,yi2=min(b1[2],b2[2]),min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if not inter: return 0.0
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter)

def count_tp(gt_in, preds, iou_t=IOU_T):
    tp=0
    for gb in gt_in:
        _,gx1,gy1,gx2,gy2=gb
        if max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in preds),default=0)>=iou_t:
            tp+=1
    return tp

def in_roi(box, x1r,y1r,x2r,y2r, m=50):
    cx=(box[1]+box[3])/2; cy=(box[2]+box[4])/2
    return x1r-m<=cx<=x2r+m and y1r-m<=cy<=y2r+m

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — Fast label scan  (no image reads)
# ══════════════════════════════════════════════════════════════════════════════
all_test = sorted(glob.glob(os.path.join(TEST_DIR,'*.jpg')))
print(f"Phase 1: scanning {len(all_test)} label files ...")

candidates = []   # (score_density, stem, cluster_norm, dom_cls)

for ip in all_test:
    stem = Path(ip).stem
    lp   = os.path.join(RAW_DIR,'labels','test',stem+'.txt')
    if not os.path.exists(lp): continue
    boxes_n = []
    for line in open(lp):
        p=line.strip().split()
        if not p: continue
        c=int(p[0]); cx,cy,bw,bh=float(p[1]),float(p[2]),float(p[3]),float(p[4])
        boxes_n.append((c,cx,cy,bw,bh))
    if len(boxes_n) < MIN_N: continue

    # Dense cluster: N boxes closest to centroid
    mcx=np.mean([b[1] for b in boxes_n]); mcy=np.mean([b[2] for b in boxes_n])
    cluster = sorted(boxes_n, key=lambda b:(b[1]-mcx)**2+(b[2]-mcy)**2)[:N_CLUSTER]

    # Cluster bounding span (normalized)
    xs=[b[1]-b[3]/2 for b in cluster]+[b[1]+b[3]/2 for b in cluster]
    ys=[b[2]-b[4]/2 for b in cluster]+[b[2]+b[4]/2 for b in cluster]
    span_x=max(xs)-min(xs); span_y=max(ys)-min(ys)
    if span_x>SPAN_MAX_NORM or span_y>SPAN_MAX_NORM: continue  # too spread out

    density = len(cluster) / max(span_x*span_y, 1e-6)
    dom_cls = Counter(b[0] for b in cluster).most_common(1)[0][0]
    candidates.append((density, stem, cluster, dom_cls))

# Per-class: take top-N densest from each of the 4 classes
by_class = {c:[] for c in range(4)}
for item in candidates:
    by_class[item[3]].append(item)
for c in by_class:
    by_class[c].sort(key=lambda x:-x[0])
top_candidates = []
for c in range(4):
    top_candidates.extend(by_class[c][:TOP_PER_CLASS])

total_cands = len(top_candidates)
per_cls_counts = {c: len(by_class[c]) for c in range(4)}
print(f"  → tight clusters per class: {per_cls_counts}  → running inference on {total_cands} candidates")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Inference on top-K candidates
# ══════════════════════════════════════════════════════════════════════════════
print("Loading models ...")
m_cmx=YOLO(MODEL_CMX); m_v11=YOLO(MODEL_V11); m_ef=YOLO(MODEL_EF)
print("Done.\n")

print("Phase 2: inference on candidates ...")
scored = []
for rank,(density,stem,cluster_n,dom_cls) in enumerate(top_candidates):
    ip = os.path.join(TEST_DIR, stem+'.jpg')
    bgr,gt,H,W = load_gt(ip)

    # Convert normalized cluster → pixel ROI
    xs_px=[]; ys_px=[]
    for c,cx,cy,bw,bh in cluster_n:
        xs_px.extend([int((cx-bw/2)*W),int((cx+bw/2)*W)])
        ys_px.extend([int((cy-bh/2)*H),int((cy+bh/2)*H)])
    bw_px=max(xs_px)-min(xs_px); bh_px=max(ys_px)-min(ys_px)
    avg_box_px=np.mean([max(int(b[3]*W),int(b[4]*H)) for b in cluster_n])
    pad = max(int(max(bw_px,bh_px)*0.40), int(avg_box_px*1.5), 120)
    x1r=max(0,min(xs_px)-pad); y1r=max(0,min(ys_px)-pad)
    x2r=min(W,max(xs_px)+pad); y2r=min(H,max(ys_px)+pad)
    roi=(x1r,y1r,x2r,y2r)

    # Inference
    _,cmx_p=run_infer(m_cmx,ip,True)
    _,v11_p=run_infer(m_v11,ip,False)
    _,ef_p =run_infer(m_ef, ip,True)

    # Filter to ROI
    gt_in  =[b for b in gt    if in_roi(b,*roi)]
    cmx_in =[b for b in cmx_p if in_roi(b,*roi)]
    v11_in =[b for b in v11_p if in_roi(b,*roi)]
    ef_in  =[b for b in ef_p  if in_roi(b,*roi)]

    total=len(gt_in)
    if total<3: continue

    tp_cmx=count_tp(gt_in,cmx_in)
    tp_v11=count_tp(gt_in,v11_in)
    tp_ef =count_tp(gt_in,ef_in)

    # Score: want (cmx>>v11) AND cmx high, normalized by GT count
    score = (tp_cmx - tp_v11) * tp_cmx / max(total, 1)

    print(f"  [{rank+1:02d}/{total_cands}] {stem} | cls={CLASS_NAMES[dom_cls][:4]} | "
          f"GT={total} CMX={tp_cmx} V11={tp_v11} EF={tp_ef} | score={score:.2f} "
          f"| ROI={x2r-x1r}×{y2r-y1r}")

    scored.append(dict(ip=ip,bgr=bgr,gt=gt,cmx=cmx_p,v11=v11_p,ef=ef_p,
                       gt_in=gt_in,cmx_in=cmx_in,v11_in=v11_in,ef_in=ef_in,
                       roi=roi,score=score,dom_cls=dom_cls,
                       tp_cmx=tp_cmx,tp_v11=tp_v11,tp_ef=tp_ef,total=total))

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3 — Pick 2 best with different dominant class
# ══════════════════════════════════════════════════════════════════════════════
scored.sort(key=lambda x: -x['score'])
selected=[]; used_cls=set()
for r in scored:
    if r['dom_cls'] not in used_cls:
        selected.append(r); used_cls.add(r['dom_cls'])
    if len(selected)==2: break
if len(selected)<2: selected=scored[:2]

print(f"\nSelected scenes:")
for i,r in enumerate(selected):
    print(f"  Row {i}: {Path(r['ip']).stem}")
    print(f"    class={CLASS_NAMES[r['dom_cls']]}  GT={r['total']}  "
          f"CMX={r['tp_cmx']}  V11={r['tp_v11']}  EF={r['tp_ef']}  score={r['score']:.2f}")
    print(f"    ROI={r['roi']}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4 — Generate figure
# ══════════════════════════════════════════════════════════════════════════════
COL_TITLES=['Ground Truth','YOLOv11s\n(RGB only)','EarlyFusion\n(4-ch)','Proposed\n(Dual-stream)']

fig,axes=plt.subplots(2,4,figsize=(16,7.8))
fig.subplots_adjust(wspace=0.025,hspace=0.10,left=0.09,right=0.995,top=0.91,bottom=0.065)

for col,t in enumerate(COL_TITLES):
    axes[0][col].set_title(t,fontsize=10,fontweight='bold',pad=5)

for row,d in enumerate(selected):
    x1r,y1r,x2r,y2r=d['roi']; cw,ch=x2r-x1r,y2r-y1r
    bgr=d['bgr']

    # CMX-wins (detected by CMX, missed by V11) — for attention circle
    wins_in=[]
    for gb in d['gt_in']:
        _,gx1,gy1,gx2,gy2=gb
        ci=max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in d['cmx_in']),default=0)
        vi=max((box_iou((gx1,gy1,gx2,gy2),(p[1],p[2],p[3],p[4])) for p in d['v11_in']),default=0)
        if ci>=IOU_T and vi<IOU_T: wins_in.append(gb)
    primary_win=wins_in[0] if wins_in else (d['gt_in'][0] if d['gt_in'] else None)

    base_en=enhance(bgr[y1r:y2r,x1r:x2r].copy())

    gt_c  =to_crop(d['gt_in'], x1r,y1r)
    cmx_c =to_crop(d['cmx_in'],x1r,y1r)
    v11_c =to_crop(d['v11_in'],x1r,y1r)
    ef_c  =to_crop(d['ef_in'], x1r,y1r)

    # TP counts for annotation
    total=d['total']; tp_cmx=d['tp_cmx']; tp_v11=d['tp_v11']; tp_ef=d['tp_ef']

    col_specs=[
        (gt_c,  [],    f"GT: {total} fruits"),
        ([],    v11_c, f"TP: {tp_v11}/{total}"),
        ([],    ef_c,  f"TP: {tp_ef}/{total}"),
        ([],    cmx_c, f"TP: {tp_cmx}/{total}"),
    ]
    panels=[]
    for lbl,pred,ann in col_specs:
        vis=base_en.copy()
        draw_gt_crop(vis,lbl)
        draw_pred_crop(vis,pred)
        panel=make_panel(vis,DISP_H)
        add_circle(panel,primary_win,x1r,y1r,cw,ch)
        annotate_count(panel,ann)
        panels.append(panel)

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

    axes[row][0].set_ylabel(CLASS_NAMES[d['dom_cls']],fontsize=10,fontweight='bold',
                             rotation=90,va='center',labelpad=8)

    zoom=DISP_H/max(ch,1)
    print(f"  Row {row}: ROI {cw}×{ch}px → display zoom {zoom:.2f}×  "
          f"wins_in={len(wins_in)}")

handles=[mpatches.Patch(facecolor='white',edgecolor='gray',label='GT (white dashed)')]
for i in range(4):
    handles.append(mpatches.Patch(color=PRED_HEX[i],label=CLASS_NAMES[i]))
handles.append(mpatches.Patch(color='#FFFF00',label='○ CMX-win region'))
fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=8.5,
           frameon=True,bbox_to_anchor=(0.55,0.0))

print()
save_fig(fig,'fig_comparison')
print("Done.")
