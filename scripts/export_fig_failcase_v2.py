"""fig_failcase — 2×2 layout, column-width sizing (~7 in), font >= 12pt.
Runs GMD-YOLO on all 383 test images, picks 4 failure-type panels.
Panels: FN / FP / class confusion / low-recall scene.
"""
import os, sys, glob
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

REPO_DIR = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics'
FIG_DIR  = os.path.join(REPO_DIR, 'paper_figures')
RAW_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
NDVI_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi'
MODEL_CMX = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
TEST_IMG_DIR = os.path.join(RAW_DIR, 'images', 'test')

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['CMU Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.weight':      'normal',
    'axes.titleweight': 'normal',
    'font.size':        12,
    'axes.titlesize':   12,
})

CLASS_NAMES  = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
CLASS_SHORT  = ['Pre-pol.', 'Post-pol.', 'Net form.', 'Sweetening']
BOX_BGR      = [(255, 0, 255), (255, 191, 0), (0, 215, 255), (64, 64, 255)]
BOX_RGB_HEX  = ['#FF00FF', '#00BFFF', '#FFD700', '#FF4040']
GT_BGR       = (180, 180, 180)
FAIL_H       = 420

def load_4ch(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None: return None, None
    fname = Path(img_path).stem + '.png'; ndvi = None
    for sp in ['test', 'val', 'train']:
        p = os.path.join(NDVI_DIR, 'images', sp, fname)
        if os.path.exists(p): ndvi = cv2.imread(p, cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir=bgr[:,:,2].astype(np.float32); red=bgr[:,:,1].astype(np.float32)
        ndvi=(((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape != bgr.shape[:2]:
        ndvi = cv2.resize(ndvi, (bgr.shape[1], bgr.shape[0]))
    return bgr, np.dstack([bgr, ndvi])

def load_gt(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None: return None, []
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

def run_infer(model, img_path, is_4ch=True, conf=0.25, iou=0.6):
    bgr, img4 = load_4ch(img_path)
    if bgr is None: return None, []
    src = img4 if is_4ch else img_path
    res = model.predict(source=src, conf=conf, iou=iou, imgsz=1280, verbose=False)
    boxes = []
    for box in res[0].boxes:
        cls=int(box.cls[0]); cf=float(box.conf[0])
        x1,y1,x2,y2=[int(v) for v in box.xyxy[0].tolist()]
        boxes.append((cls,x1,y1,x2,y2,cf))
    return bgr, boxes

def box_iou(b1, b2):
    xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter==0: return 0.0
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter)

def crop_to_roi(img_rgb, roi_boxes, pad_ratio=0.6, min_size=200):
    if not roi_boxes: return img_rgb
    xs=[b[1] for b in roi_boxes]+[b[3] for b in roi_boxes]
    ys=[b[2] for b in roi_boxes]+[b[4] for b in roi_boxes]
    x1c,x2c=min(xs),max(xs); y1c,y2c=min(ys),max(ys)
    bw,bh=x2c-x1c,y2c-y1c
    pad_x=max(int(bw*pad_ratio),min_size//4); pad_y=max(int(bh*pad_ratio),min_size//4)
    H,W=img_rgb.shape[:2]
    c1=max(0,x1c-pad_x); c2=min(W,x2c+pad_x)
    r1=max(0,y1c-pad_y); r2=min(H,y2c+pad_y)
    crop=img_rgb[r1:r2,c1:c2]
    return crop if crop.size>0 else img_rgb

def resize_h(img, target_h=420):
    H,W=img.shape[:2]; W2=int(W*target_h/H)
    return cv2.resize(img,(W2,target_h),interpolation=cv2.INTER_AREA)

def make_failcase_panel(ip, gt_boxes, pred_boxes, highlight_gt=None, highlight_pred=None):
    bgr = cv2.imread(ip); vis = bgr.copy()
    for (cls,x1,y1,x2,y2) in gt_boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),GT_BGR,3)
    for (cls,x1,y1,x2,y2,cf) in pred_boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),BOX_BGR[cls],5)
    if highlight_gt is not None:
        cls,x1,y1,x2,y2 = highlight_gt[:5]
        cv2.rectangle(vis,(x1-4,y1-4),(x2+4,y2+4),(0,0,255),6)
        lbl=f'GT: {CLASS_SHORT[cls]}'
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.65,2)
        cv2.rectangle(vis,(x1,y2+2),(x1+tw+6,y2+th+12),(0,0,200),-1)
        cv2.putText(vis,lbl,(x1+3,y2+th+5),cv2.FONT_HERSHEY_DUPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
    if highlight_pred is not None:
        cls,x1,y1,x2,y2,cf = highlight_pred
        cv2.rectangle(vis,(x1-4,y1-4),(x2+4,y2+4),(0,128,255),6)
        lbl=f'PRED: {CLASS_SHORT[cls]} {cf:.2f}'
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.65,2)
        cv2.rectangle(vis,(x1,y1-th-12),(x1+tw+6,y1),(0,128,200),-1)
        cv2.putText(vis,lbl,(x1+3,y1-3),cv2.FONT_HERSHEY_DUPLEX,0.65,(255,255,255),2,cv2.LINE_AA)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    if highlight_gt is not None:
        roi = [highlight_gt[:5]]
    elif highlight_pred is not None:
        roi = [(highlight_pred[0],highlight_pred[1],highlight_pred[2],
                highlight_pred[3],highlight_pred[4])]
    else:
        roi = gt_boxes[:3] if gt_boxes else []
    return crop_to_roi(vis_rgb, roi, pad_ratio=0.6, min_size=200)

def pick_best(cases, key_fn, n=1, min_area=800, max_area=500000):
    filtered=[]
    for c in cases:
        if len(c)>=3:
            box=c[1]; area=(box[3]-box[1])*(box[4]-box[2]) if len(box)>=5 else 0
            if min_area<=area<=max_area: filtered.append(c)
    seen=set(); result=[]
    for c in sorted(filtered,key=key_fn):
        ip=c[0]
        if ip not in seen: seen.add(ip); result.append(c)
        if len(result)>=n: break
    return result

print("Loading GMD-YOLO model...")
from ultralytics import YOLO
m_cmx = YOLO(MODEL_CMX)
print("  Loaded.")

all_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, '*.jpg')))
print(f"Running GMD-YOLO on all {len(all_test)} test images (conf=0.25)...")
all_results = {}
for i, ip in enumerate(all_test):
    if (i+1) % 50 == 0: print(f"  {i+1}/{len(all_test)}")
    bgr, pred = run_infer(m_cmx, ip, is_4ch=True)
    _, gt = load_gt(ip)
    all_results[ip] = (bgr, pred, gt)
print(f"  Done. {len(all_results)} images processed.")

print("Collecting failure cases...")
fn_cases=[]; fp_cases=[]; conf_cases=[]; glare_cases=[]
for ip,(bgr,pred,gt) in all_results.items():
    gt_matched=[False]*len(gt); pred_matched=[False]*len(pred)
    for gi,(gc,gx1,gy1,gx2,gy2) in enumerate(gt):
        best_iou,best_pi=0.0,-1
        for pi,(pc,px1,py1,px2,py2,_) in enumerate(pred):
            if pred_matched[pi]: continue
            v=box_iou((gx1,gy1,gx2,gy2),(px1,py1,px2,py2))
            if v>best_iou: best_iou=v; best_pi=pi
        if best_pi>=0 and best_iou>=0.5:
            pred_matched[best_pi]=True; gt_matched[gi]=True
            if pred[best_pi][0]!=gc: conf_cases.append((ip,gt[gi],pred[best_pi],pred,gt))
        else:
            fn_cases.append((ip,gt[gi],pred,gt))
    for pi,pbox in enumerate(pred):
        if not pred_matched[pi]: fp_cases.append((ip,pbox,pred,gt))
    if len(gt)>=3:
        recall=sum(gt_matched)/len(gt)
        if recall<0.5:
            avg_a=np.mean([(b[3]-b[1])*(b[4]-b[2]) for b in gt]) if gt else 0
            glare_cases.append((recall,avg_a,ip,pred,gt))
glare_cases.sort(key=lambda x:x[0])
print(f"  FN:{len(fn_cases)} FP:{len(fp_cases)} Conf:{len(conf_cases)} Glare:{len(glare_cases)}")

fn_pick = pick_best(fn_cases,   key_fn=lambda c:-(c[1][3]-c[1][1])*(c[1][4]-c[1][2]), n=1)
fp_pick = pick_best(fp_cases,   key_fn=lambda c:-c[1][5], n=1)
net_sweet_conf = [c for c in conf_cases if {c[1][0],c[2][0]}=={2,3}]
nsc_pick = pick_best(net_sweet_conf or conf_cases,
                     key_fn=lambda c:-(c[1][3]-c[1][1])*(c[1][4]-c[1][2]), n=1)
glare_pick = []
for (recall,avg_a,ip,pred,gt) in glare_cases[:10]:
    if avg_a<500: continue
    glare_pick.append((ip,pred,gt)); break

fail_panels=[]; fail_titles=[]
if fn_pick:
    ip,missed_gt,pred,gt=fn_pick[0]
    fail_panels.append(make_failcase_panel(ip,gt,pred,highlight_gt=missed_gt))
    fail_titles.append(f'(a) Missed detection\n(GT: {CLASS_SHORT[missed_gt[0]]})')
if fp_pick:
    ip,fp_box,pred,gt=fp_pick[0]
    fail_panels.append(make_failcase_panel(ip,gt,pred,highlight_pred=fp_box))
    fail_titles.append(f'(b) False positive\n(Pred: {CLASS_SHORT[fp_box[0]]} {fp_box[5]:.2f})')
if nsc_pick:
    ip,gt_box,pred_box,pred,gt=nsc_pick[0]
    fail_panels.append(make_failcase_panel(ip,gt,pred,highlight_gt=gt_box,highlight_pred=pred_box))
    gt_cls,pred_cls=gt_box[0],pred_box[0]
    fail_titles.append(f'(c) Class confusion\n{CLASS_SHORT[gt_cls]} → {CLASS_SHORT[pred_cls]}')
if glare_pick:
    ip,pred,gt=glare_pick[0]
    fail_panels.append(make_failcase_panel(ip,gt,pred))
    fail_titles.append('(d) Low recall scene\n(occlusion / glare)')

# Pad to 4 panels if needed
if len(fn_cases)>1 and len(fail_panels)<4:
    for c in fn_cases[1:]:
        if len(fail_panels)>=4: break
        ip,missed_gt,pred,gt=c
        fail_panels.append(make_failcase_panel(ip,gt,pred,highlight_gt=missed_gt))
        fail_titles.append(f'(d) Missed detection\n(GT: {CLASS_SHORT[missed_gt[0]]})')

fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
fig.subplots_adjust(wspace=0.05, hspace=0.22, left=0.02, right=0.98, top=0.95, bottom=0.14)

for idx,(crop,title) in enumerate(zip(fail_panels[:4], fail_titles[:4])):
    r,c = idx//2, idx%2
    ax = axes[r][c]
    ax.imshow(resize_h(crop, FAIL_H))
    ax.set_title(title, fontsize=12, pad=5)
    ax.set_xticks([]); ax.set_yticks([])

for idx in range(len(fail_panels), 4):
    axes[idx//2][idx%2].axis('off')

legend_handles = [
    mpatches.Patch(color='silver',  label='GT box (gray)',   ec='gray'),
    mpatches.Patch(color='red',     label='Missed GT (red outline)'),
    mpatches.Patch(color='#FF8000', label='Highlighted Pred (orange)'),
] + [mpatches.Patch(color=BOX_RGB_HEX[i], label=f'Pred: {CLASS_SHORT[i]}') for i in range(4)]

fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=11,
           frameon=True, bbox_to_anchor=(0.5, -0.03))

for ext in ['pdf', 'svg']:
    fig.savefig(f'{FIG_DIR}/fig_failcase.{ext}', bbox_inches='tight', dpi=300)
plt.close()
print(f'\nfig_failcase: figsize=7×5.5 in, 2×2 grid, title=12pt, legend=11pt')
print(f'Saved: {FIG_DIR}/fig_failcase.pdf + .svg')
