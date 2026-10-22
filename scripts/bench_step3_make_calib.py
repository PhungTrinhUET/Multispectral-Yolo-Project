"""
Step 3a — Build INT8 calibration dataset (shared across VOXL2 and Jetson).

Requirements:
  - 100–200 images from the test split covering all 4 growth stages + varied lighting
  - Saved as float32 raw binaries for SNPE, AND as TFLite representative dataset
  - SAME set used on both devices for fairness

Outputs:
  bench_results/calib/
    ├── img_0000.raw  … img_0199.raw   (NCHW float32, for SNPE)
    ├── calib_list.txt                 (one path per line, for snpe-dlc-quant)
    └── calib_npy/
        └── img_0000.npy … img_0199.npy  (NHWC float32, for TFLite rep dataset)

Usage:
    python scripts/bench_step3_make_calib.py
"""

import sys, os, json
import numpy as np
import cv2

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/test/images'
OUT_DIR  = os.path.join(ROOT, 'bench_results', 'calib')
NPY_DIR  = os.path.join(OUT_DIR, 'calib_npy')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(NPY_DIR, exist_ok=True)

IMGSZ   = 640
N_CALIB = 200   # use up to 200 images; INT8 quality plateaus well before this

# ── Collect test images ───────────────────────────────────────────────────────
all_imgs = sorted([
    f for f in os.listdir(TEST_DIR)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
])

# Subsample evenly (preserves class distribution assuming sorted = mixed)
step = max(1, len(all_imgs) // N_CALIB)
selected = all_imgs[::step][:N_CALIB]
print(f"[INFO] Total test images: {len(all_imgs)}  →  selected {len(selected)} for calib")

calib_paths_snpe = []
calib_paths_npy  = []

for i, fname in enumerate(selected):
    src = os.path.join(TEST_DIR, fname)

    # Load 4-channel image (PNG with alpha = NIR, or 3ch RGB → pad NIR=0)
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"[WARN] Cannot read {src}, skipping")
        continue

    if img.ndim == 2:          # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 3:      # RGB only — pad NDVI channel as 0
        nir = np.zeros((*img.shape[:2], 1), dtype=img.dtype)
        img = np.concatenate([img, nir], axis=2)

    # Resize to deployment resolution
    img = cv2.resize(img, (IMGSZ, IMGSZ))

    # Normalize to [0, 1]
    img_f = img.astype(np.float32) / 255.0   # H×W×4  NHWC

    # SNPE format: NCHW float32 binary
    nchw = img_f.transpose(2, 0, 1)          # 4×H×W
    raw_path = os.path.join(OUT_DIR, f'img_{i:04d}.raw')
    nchw.tofile(raw_path)
    calib_paths_snpe.append(raw_path)

    # TFLite format: NHWC float32 numpy
    npy_path = os.path.join(NPY_DIR, f'img_{i:04d}.npy')
    np.save(npy_path, img_f[np.newaxis])     # 1×H×W×4
    calib_paths_npy.append(npy_path)

    if (i + 1) % 50 == 0:
        print(f"  Processed {i+1}/{len(selected)}")

# Write SNPE input list
list_path = os.path.join(OUT_DIR, 'calib_list.txt')
with open(list_path, 'w') as f:
    f.write('\n'.join(calib_paths_snpe))

# Save manifest
manifest = {
    "n_images": len(calib_paths_snpe),
    "imgsz": IMGSZ,
    "format_snpe": "NCHW float32 .raw",
    "format_tflite": "NHWC float32 .npy (shape 1xHxWx4)",
    "snpe_list": list_path,
    "npy_dir": NPY_DIR,
    "source_dir": TEST_DIR,
}
with open(os.path.join(OUT_DIR, 'calib_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"\n[DONE] Calibration set: {len(calib_paths_snpe)} images")
print(f"  SNPE  list → {list_path}")
print(f"  TFLite npy → {NPY_DIR}/")
print("Next: python scripts/bench_step3_convert_tflite.py")
