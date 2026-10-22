"""fig_rgb_ndvi — 1×3 panels, column-width sizing (~7.5 in), font >= 13pt."""
import os, sys, glob, random
import cv2, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.chdir('/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')
sys.path.insert(0, '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')

FIG_DIR = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics/paper_figures'
RAW_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw'

plt.rcParams.update({
    'font.family':      'serif',
    'font.serif':       ['CMU Serif', 'DejaVu Serif', 'Times New Roman'],
    'mathtext.fontset': 'cm',
    'font.weight':      'normal',
    'axes.titleweight': 'normal',
    'font.size':        13,
    'axes.titlesize':   13,
})

def compute_ndvi(bgr):
    nir = bgr[:,:,2].astype(np.float32)
    red = bgr[:,:,1].astype(np.float32)
    return (nir - red) / (nir + red + 1e-6).clip(1e-6)

# Same image-selection logic as gen_figures.py (seed=42)
lbl_dir = os.path.join(RAW_DIR, 'labels', 'train')
img_dir = os.path.join(RAW_DIR, 'images', 'train')
best_img = None; best_score = 0
random.seed(42)
candidates = random.sample(os.listdir(lbl_dir), min(300, len(os.listdir(lbl_dir))))
for txt in candidates:
    try:
        lines = open(os.path.join(lbl_dir, txt)).readlines()
        counts = [0,0,0,0]
        for l in lines:
            c = int(l.split()[0])
            if c < 4: counts[c] += 1
        n_cls = sum(1 for c in counts if c > 0)
        score = n_cls * 10 + sum(counts[:3])
        if score > best_score:
            cand = os.path.join(img_dir, txt.replace('.txt', '.jpg'))
            if os.path.exists(cand):
                best_score = score; best_img = cand
    except: pass

bgr = cv2.imread(best_img)
H, W = bgr.shape[:2]
cy, cx = H//4, W//8
bgr_crop = bgr[cy:cy+H//2, cx:cx+3*W//4]

rgb_crop  = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
nir_crop  = bgr_crop[:,:,2]
ndvi_crop = compute_ndvi(bgr_crop)

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.8))

axes[0].imshow(rgb_crop)
axes[0].set_title('(a) Pseudo-color (R/G/NIR)', fontsize=13, pad=5)
axes[0].axis('off')

axes[1].imshow(nir_crop, cmap='gray')
axes[1].set_title('(b) NIR channel', fontsize=13, pad=5)
axes[1].axis('off')

ndvi_im = axes[2].imshow(ndvi_crop, cmap='RdYlGn', vmin=-0.5, vmax=0.8)
axes[2].set_title('(c) NDVI map', fontsize=13, pad=5)
axes[2].axis('off')
cb = fig.colorbar(ndvi_im, ax=axes[2], fraction=0.046, pad=0.02, shrink=0.85)
cb.ax.tick_params(labelsize=11)
cb.set_label('NDVI', fontsize=12)

fig.tight_layout(pad=0.3)
for ext in ['pdf', 'svg']:
    fig.savefig(f'{FIG_DIR}/fig_rgb_ndvi.{ext}', bbox_inches='tight', dpi=300)
plt.close()
print(f'fig_rgb_ndvi: figsize=7.5×2.8 in, title=13pt, colorbar=11pt')
print(f'  Saved: {FIG_DIR}/fig_rgb_ndvi.pdf + .svg')
print(f'  Source image: {os.path.basename(best_img)}')
