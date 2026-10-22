"""Regenerate fig_qualitative at 300 DPI. No suptitle. Layout identical to gen_figures.py."""
import os, sys, glob, random
import cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.chdir('/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')
sys.path.insert(0, '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')

FIG_DIR  = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics/paper_figures'
RAW_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'
MODEL_CMX = 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt'

CB = {
    'orange': '#E69F00', 'skyblue': '#56B4E9', 'green': '#009E73',
    'yellow': '#F0E442', 'blue': '#0072B2', 'red': '#D55E00',
    'pink': '#CC79A7',   'black': '#000000'
}
CLASS_COLORS = [CB['orange'], CB['skyblue'], CB['green'], CB['red']]
CLASS_NAMES  = ['Pre-pollination', 'Post-pollination', 'Net formation', 'Sweetening']
CLASS_SHORT  = ['truoc_thu_phan', 'sau_thu_phan', 'cay40', 'tao_ngot']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.titlesize': 9,
    'figure.dpi': 150,
})

def pick_images_per_class(split='test', n=6):
    lbl_dir = os.path.join(RAW_DIR, 'labels', split)
    img_dir = os.path.join(RAW_DIR, 'images', split)
    best = {c: [] for c in range(4)}
    for txt in glob.glob(os.path.join(lbl_dir, '*.txt')):
        counts = [0, 0, 0, 0]
        try:
            for line in open(txt):
                cls = int(line.split()[0])
                if cls < 4: counts[cls] += 1
        except: continue
        for c in range(4):
            if counts[c] > 0:
                img_path = os.path.join(img_dir,
                    os.path.basename(txt).replace('.txt', '.jpg'))
                if os.path.exists(img_path):
                    best[c].append((counts[c], txt, img_path))
    result = {}
    for c in range(4):
        ranked = sorted(best[c], key=lambda x: x[0], reverse=True)
        result[c] = [(r[1], r[2]) for r in ranked[:n]]
    return result

def run_inference_crop(model, img_path, conf=0.3, iou=0.6, scale=0.22):
    bgr = cv2.imread(img_path)
    if bgr is None: return np.zeros((200, 200, 3), dtype=np.uint8)
    H, W = bgr.shape[:2]
    results = model.predict(img_path, conf=conf, iou=iou, imgsz=1280, verbose=False)
    vis = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    for box in results[0].boxes:
        cls = int(box.cls[0])
        conf_val = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        import matplotlib.colors
        color_rgb = tuple(int(x * 255) for x in matplotlib.colors.to_rgb(CLASS_COLORS[cls]))
        cv2.rectangle(vis, (x1, y1), (x2, y2), color_rgb, 4)
        label = f'{CLASS_SHORT[cls][:4]} {conf_val:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(vis, (x1, y1 - th - 4), (x1 + tw + 4, y1), color_rgb, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    new_w, new_h = int(W * scale), int(H * scale)
    return cv2.resize(vis, (new_w, new_h), interpolation=cv2.INTER_AREA)

from ultralytics import YOLO
model_ours = YOLO(MODEL_CMX)

test_dir = os.path.join(RAW_DIR, 'images', 'test')
test_imgs = sorted(glob.glob(os.path.join(test_dir, '*.jpg')))

images_per_class_test = pick_images_per_class('test', n=6)
selected = []
for cls in range(4):
    samps = images_per_class_test[cls]
    for lbl, img in samps[:2]:
        if img not in [s[1] for s in selected]:
            selected.append((cls, lbl, img))
        if len(selected) >= 8: break
    if len(selected) >= 8: break

random.seed(7)
extra = [p for p in test_imgs if p not in [s[2] for s in selected]]
while len(selected) < 8:
    p = random.choice(extra)
    extra.remove(p)
    lbl = os.path.join(RAW_DIR, 'labels', 'test',
                       os.path.basename(p).replace('.jpg', '.txt'))
    selected.append((0, lbl, p))

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for idx, (cls, lbl, img_path) in enumerate(selected[:8]):
    ax = axes[idx // 4][idx % 4]
    thumb = run_inference_crop(model_ours, img_path)
    ax.imshow(thumb)
    ax.set_title(f'Stage: {CLASS_NAMES[cls]}', fontsize=8, pad=3)
    ax.axis('off')

handles = [mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(4)]
fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=8.5,
           frameon=True, bbox_to_anchor=(0.5, -0.03))
# No suptitle
fig.tight_layout(pad=0.4)

out_pdf = f'{FIG_DIR}/fig_qualitative.pdf'
out_svg = f'{FIG_DIR}/fig_qualitative.svg'
fig.savefig(out_pdf, bbox_inches='tight', dpi=300)
fig.savefig(out_svg, bbox_inches='tight', dpi=300)
plt.close()
print(f'Saved: {out_pdf}')
print(f'Saved: {out_svg}')
