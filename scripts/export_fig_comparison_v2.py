"""fig_comparison — 2 rows × 4 cols, column-width sizing (~7.5 in), font >= 13pt.
Rows: Net formation and Sweetening (most visually distinct stages).
Cols: Ground Truth | YOLOv11s(RGB) | EarlyFusion(4ch) | GMD-YOLO(Ours)
Each prediction cell annotated with 'TP: x / GT: y'.
"""
import os, sys, glob
from pathlib import Path
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
MODEL_V11 = '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolo11s_1280_balanced_b4/weights/best.pt'
MODEL_EF  = os.path.join(REPO_DIR, 'Multispectral_DuaLuoi_Project/Exp1_NoPretrain_1280px/weights/best.pt')
TEST_IMG_DIR = os.path.join(RAW_DIR, 'images', 'test')

os.chdir(REPO_DIR)
sys.path.insert(0, REPO_DIR)

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})

CLASS_NAMES  = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
CLASS_SHORT  = ['Pre-pol.', 'Post-pol.', 'Net form.', 'Sweetening']
BOX_BGR      = [(255, 0, 255), (255, 191, 0), (0, 215, 255), (64, 64, 255)]
BOX_RGB_HEX  = ['#FF00FF', '#00BFFF', '#FFD700', '#FF4040']
GT_BGR       = (255, 255, 255)

# Two curated scenes: net-formation (15 boxes) and sweetening (106 boxes)
COMP_IMGS = [
    os.path.join(TEST_IMG_DIR, '2023_0620_161134_014.jpg'),   # net formation
    os.path.join(TEST_IMG_DIR, '2023_0620_174748_521.jpg'),   # sweetening
]
ROW_LABELS = ['Net formation', 'Sweetening']
COL_TITLES = ['Ground Truth', 'YOLOv11s\n(R/G/NIR)', 'EarlyFusion\n(4ch)', 'GMD-YOLO\n(Ours)']
CROP_H_COMP = 380

def load_4ch(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None: return None, None
    fname = Path(img_path).stem + '.png'
    ndvi = None
    for sp in ['test', 'val', 'train']:
        p = os.path.join(NDVI_DIR, 'images', sp, fname)
        if os.path.exists(p):
            ndvi = cv2.imread(p, cv2.IMREAD_GRAYSCALE); break
    if ndvi is None:
        nir = bgr[:,:,2].astype(np.float32); red = bgr[:,:,1].astype(np.float32)
        ndvi = (((nir-red)/(nir+red+1e-6)+1)/2*255).clip(0,255).astype(np.uint8)
    if ndvi.shape != bgr.shape[:2]:
        ndvi = cv2.resize(ndvi, (bgr.shape[1], bgr.shape[0]))
    return bgr, np.dstack([bgr, ndvi])

def load_gt(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None: return None, []
    H, W = bgr.shape[:2]; stem = Path(img_path).stem; boxes = []
    for sp in ['test', 'val', 'train']:
        lp = os.path.join(RAW_DIR, 'labels', sp, stem + '.txt')
        if os.path.exists(lp):
            for line in open(lp):
                p = line.strip().split()
                if not p: continue
                cls = int(p[0]); cx,cy,bw,bh = float(p[1]),float(p[2]),float(p[3]),float(p[4])
                x1=int((cx-bw/2)*W); y1=int((cy-bh/2)*H)
                x2=int((cx+bw/2)*W); y2=int((cy+bh/2)*H)
                boxes.append((cls,x1,y1,x2,y2))
            break
    return bgr, boxes

def run_infer(model, img_path, is_4ch, conf=0.25, iou=0.6):
    bgr, img4 = load_4ch(img_path)
    if bgr is None: return None, []
    src = img4 if is_4ch else img_path
    res = model.predict(source=src, conf=conf, iou=iou, imgsz=1280, verbose=False)
    boxes = []
    for box in res[0].boxes:
        cls=int(box.cls[0]); cf=float(box.conf[0])
        x1,y1,x2,y2 = [int(v) for v in box.xyxy[0].tolist()]
        boxes.append((cls,x1,y1,x2,y2,cf))
    return bgr, boxes

def box_iou(b1, b2):
    xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    if inter==0: return 0.0
    a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter)

def count_tp(pred_boxes, gt_boxes, iou_thres=0.5):
    """Count TP predictions: best-match greedy, any class."""
    gt_matched = [False]*len(gt_boxes); tp = 0
    for (cls,px1,py1,px2,py2,cf) in pred_boxes:
        best_iou,best_gi = 0,-1
        for gi,(gc,gx1,gy1,gx2,gy2) in enumerate(gt_boxes):
            if gt_matched[gi]: continue
            v = box_iou((px1,py1,px2,py2),(gx1,gy1,gx2,gy2))
            if v > best_iou: best_iou=v; best_gi=gi
        if best_gi>=0 and best_iou>=iou_thres:
            gt_matched[best_gi]=True; tp+=1
    return tp

def draw_gt_vis(bgr, gt_boxes, lw=7):
    vis = bgr.copy()
    for (cls,x1,y1,x2,y2) in gt_boxes:
        cv2.rectangle(vis,(x1,y1),(x2,y2),GT_BGR,lw)
        lbl=CLASS_SHORT[cls]
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.55,1)
        cv2.rectangle(vis,(x1,y1-th-8),(x1+tw+6,y1),GT_BGR,-1)
        cv2.putText(vis,lbl,(x1+3,y1-3),cv2.FONT_HERSHEY_DUPLEX,0.55,(0,0,0),1,cv2.LINE_AA)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

def draw_pred(bgr, boxes, lw=7):
    vis = bgr.copy()
    for (cls,x1,y1,x2,y2,cf) in boxes:
        color=BOX_BGR[cls]
        cv2.rectangle(vis,(x1,y1),(x2,y2),color,lw)
        lbl=f'{CLASS_SHORT[cls]} {cf:.2f}'
        (tw,th),_=cv2.getTextSize(lbl,cv2.FONT_HERSHEY_DUPLEX,0.55,1)
        cv2.rectangle(vis,(x1,y1-th-8),(x1+tw+6,y1),color,-1)
        tc=(0,0,0) if sum(color)>400 else (255,255,255)
        cv2.putText(vis,lbl,(x1+3,y1-3),cv2.FONT_HERSHEY_DUPLEX,0.55,tc,1,cv2.LINE_AA)
    return cv2.cvtColor(vis,cv2.COLOR_BGR2RGB)

def crop_to_roi(img_rgb, gt_boxes, pad_ratio=0.35, min_size=300):
    if not gt_boxes: return img_rgb
    xs=[b[1] for b in gt_boxes]+[b[3] for b in gt_boxes]
    ys=[b[2] for b in gt_boxes]+[b[4] for b in gt_boxes]
    x1c,x2c=min(xs),max(xs); y1c,y2c=min(ys),max(ys)
    bw,bh=x2c-x1c,y2c-y1c
    pad_x=max(int(bw*pad_ratio),min_size//4)
    pad_y=max(int(bh*pad_ratio),min_size//4)
    H,W=img_rgb.shape[:2]
    c1=max(0,x1c-pad_x); c2=min(W,x2c+pad_x)
    r1=max(0,y1c-pad_y); r2=min(H,y2c+pad_y)
    crop=img_rgb[r1:r2,c1:c2]
    return crop if crop.size>0 else img_rgb

def resize_h(img, target_h=380):
    H,W=img.shape[:2]; W2=int(W*target_h/H)
    return cv2.resize(img,(W2,target_h),interpolation=cv2.INTER_AREA)

print("Loading models...")
from ultralytics import YOLO
m_cmx = YOLO(MODEL_CMX)
m_v11 = YOLO(MODEL_V11)
m_ef  = YOLO(MODEL_EF)
print("  All 3 loaded.")

fig, axes = plt.subplots(2, 4, figsize=(7.5, 4.3))
fig.subplots_adjust(wspace=0.04, hspace=0.12, left=0.11, right=0.99, top=0.88, bottom=0.12)

for col, title in enumerate(COL_TITLES):
    axes[0][col].set_title(title, fontsize=13, fontweight='bold', pad=6)

for row, ip in enumerate(COMP_IMGS):
    if not os.path.exists(ip):
        print(f"  WARNING: {ip} not found"); continue

    bgr, gt = load_gt(ip)
    gt_crop = gt[:8] if len(gt) > 8 else gt

    _, pred_v11 = run_infer(m_v11, ip, is_4ch=False)
    _, pred_ef  = run_infer(m_ef,  ip, is_4ch=True)
    _, pred_cmx = run_infer(m_cmx, ip, is_4ch=True)

    tp_v11 = count_tp(pred_v11, gt)
    tp_ef  = count_tp(pred_ef,  gt)
    tp_cmx = count_tp(pred_cmx, gt)
    n_gt   = len(gt)

    imgs_rgb = [
        draw_gt_vis(bgr, gt),
        draw_pred(bgr, pred_v11),
        draw_pred(bgr, pred_ef),
        draw_pred(bgr, pred_cmx),
    ]
    tp_labels = [None, f'TP: {tp_v11}/{n_gt}', f'TP: {tp_ef}/{n_gt}', f'TP: {tp_cmx}/{n_gt}']

    for col, (im, tp_lbl) in enumerate(zip(imgs_rgb, tp_labels)):
        crop = crop_to_roi(im, gt_crop)
        axes[row][col].imshow(resize_h(crop, CROP_H_COMP))
        axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
        if tp_lbl:
            axes[row][col].set_xlabel(tp_lbl, fontsize=11, labelpad=3)

    axes[row][0].set_ylabel(ROW_LABELS[row], fontsize=13, fontweight='bold',
                             rotation=90, va='center', labelpad=6)
    print(f"  Row {row} ({ROW_LABELS[row]}): GT={n_gt}  V11={tp_v11}  EF={tp_ef}  CMX={tp_cmx}")

handles = ([mpatches.Patch(color='white', label='GT box', ec='gray')] +
           [mpatches.Patch(color=BOX_RGB_HEX[i], label=CLASS_NAMES[i]) for i in range(4)])
fig.legend(handles=handles, loc='lower center', ncol=5, fontsize=11,
           frameon=True, bbox_to_anchor=(0.55, -0.02))

for ext in ['pdf', 'svg']:
    fig.savefig(f'{FIG_DIR}/fig_comparison.{ext}', bbox_inches='tight', dpi=300)
plt.close()
print(f'\nfig_comparison: figsize=7.5×4.3 in, 2×4, col_title=13pt, row_label=13pt')
print(f'Saved: {FIG_DIR}/fig_comparison.pdf + .svg')
