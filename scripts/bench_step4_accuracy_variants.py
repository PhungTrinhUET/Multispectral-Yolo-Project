"""
Step 4 — Accuracy check: FP32 / FP16 / INT8 TFLite on PC.

Runs inference with each TFLite model on the FULL test set,
computes mAP50 and mAP50-95, Precision, Recall.

Gotcha #4 from spec: on-device accuracy must MATCH PC for the same variant.
This PC measurement is the reference. If VOXL2/Jetson numbers differ → bug.

Outputs: bench_results/step4_accuracy.json

Usage:
    python scripts/bench_step4_accuracy_variants.py
"""

import sys, os, json, time
import numpy as np

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT_DIR  = os.path.join(ROOT, 'bench_results')
TEST_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/test/images'
LBL_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/test/labels'
IMGSZ    = 640
CONF_THR = 0.25
IOU_THR  = 0.45

VARIANTS = {
    'FP32': os.path.join(OUT_DIR, 'cmx_yolo_fp32.tflite'),
    'FP16': os.path.join(OUT_DIR, 'cmx_yolo_fp16.tflite'),
    'INT8': os.path.join(OUT_DIR, 'cmx_yolo_int8.tflite'),
}

try:
    import tensorflow as tf
    import cv2
except ImportError as e:
    print(f"[ERROR] Missing: {e}")
    sys.exit(1)

# ── Helpers ───────────────────────────────────────────────────────────────────
def load_image(path, imgsz):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        nir = np.zeros((*img.shape[:2], 1), dtype=img.dtype)
        img = np.concatenate([img, nir], axis=2)
    img = cv2.resize(img, (imgsz, imgsz)).astype(np.float32) / 255.0
    return img[np.newaxis]  # 1×H×W×4  NHWC

def load_labels_yolo(label_path, imgsz):
    """Load YOLO-format labels → list of (cls, cx, cy, w, h) in pixel coords."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = [float(v) * imgsz for v in parts[1:5]]
            x1, y1 = cx - bw/2, cy - bh/2
            x2, y2 = cx + bw/2, cy + bh/2
            boxes.append({'cls': cls, 'box': [x1, y1, x2, y2]})
    return boxes

def iou(b1, b2):
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / (union + 1e-7)

def compute_map(all_preds, all_gts, n_classes, iou_thresholds):
    """Simplified mAP computation (sufficient for accuracy-retention check)."""
    aps = {iou_thr: [] for iou_thr in iou_thresholds}

    for iou_thr in iou_thresholds:
        for cls in range(n_classes):
            tp_list, fp_list, n_gt = [], [], 0
            for preds, gts in zip(all_preds, all_gts):
                gt_cls  = [g for g in gts  if g['cls'] == cls]
                pr_cls  = sorted([p for p in preds if p['cls'] == cls],
                                  key=lambda x: -x['conf'])
                n_gt   += len(gt_cls)
                matched = [False] * len(gt_cls)
                for pred in pr_cls:
                    best_iou, best_j = 0, -1
                    for j, gt in enumerate(gt_cls):
                        v = iou(pred['box'], gt['box'])
                        if v > best_iou:
                            best_iou, best_j = v, j
                    if best_iou >= iou_thr and not matched[best_j]:
                        tp_list.append(1); fp_list.append(0)
                        matched[best_j] = True
                    else:
                        tp_list.append(0); fp_list.append(1)

            if n_gt == 0:
                aps[iou_thr].append(0.0)
                continue
            tp = np.cumsum(tp_list).astype(float)
            fp = np.cumsum(fp_list).astype(float)
            rec = tp / n_gt
            prec = tp / (tp + fp + 1e-7)
            # 101-point interpolation
            ap = 0
            for thr in np.linspace(0, 1, 101):
                p = prec[rec >= thr].max() if (rec >= thr).any() else 0
                ap += p / 101
            aps[iou_thr].append(ap)

    map50     = np.mean(aps[0.50])
    map5095   = np.mean([np.mean(v) for v in aps.values()])
    return map50, map5095

def run_tflite_inference(model_path, img_nhwc, is_int8=False):
    """Return raw detections from TFLite model."""
    interp = tf.lite.Interpreter(model_path=model_path, num_threads=4)
    interp.allocate_tensors()

    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    # Log actual input dtype to catch silent fallbacks
    print(f"    Input dtype  : {inp_detail['dtype'].__name__}  "
          f"(expected {'uint8' if is_int8 else 'float32'})")

    if is_int8:
        # Dequantize: uint8 from [0,1] float → scale+zp
        scale, zp = inp_detail['quantization']
        inp_data = (img_nhwc / scale + zp).clip(0, 255).astype(np.uint8)
    else:
        inp_data = img_nhwc.astype(np.float32)

    interp.set_tensor(inp_detail['index'], inp_data)
    interp.invoke()
    return interp.get_tensor(out_detail['index'])  # [1, num_preds, 4+nc]

# ── Main evaluation loop ──────────────────────────────────────────────────────
img_files = sorted([f for f in os.listdir(TEST_DIR)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
print(f"[INFO] Test images: {len(img_files)}")

results_all = {}

for variant, model_path in VARIANTS.items():
    if not os.path.exists(model_path):
        print(f"[SKIP] {variant}: {model_path} not found")
        continue

    print(f"\n{'='*60}")
    print(f"[{variant}] {model_path}")
    is_int8 = variant == 'INT8'

    # Initialize once to log delegate info
    interp_info = tf.lite.Interpreter(model_path=model_path)
    interp_info.allocate_tensors()
    inp_dtype = interp_info.get_input_details()[0]['dtype']
    print(f"  Input dtype: {inp_dtype.__name__}")

    all_preds, all_gts = [], []
    latencies = []

    for fname in img_files:
        img_path = os.path.join(TEST_DIR, fname)
        lbl_path = os.path.join(LBL_DIR, os.path.splitext(fname)[0] + '.txt')

        img = load_image(img_path, IMGSZ)
        if img is None:
            continue

        gt = load_labels_yolo(lbl_path, IMGSZ)

        t0 = time.perf_counter()
        raw = run_tflite_inference(model_path, img, is_int8)
        latencies.append((time.perf_counter() - t0) * 1000)

        # raw: [1, N, 4+nc]  format: [cx, cy, w, h, c0, c1, c2, c3]
        preds = []
        if raw is not None and raw.ndim == 3:
            for det in raw[0]:
                scores = det[4:]
                cls    = int(np.argmax(scores))
                conf   = float(scores[cls])
                if conf < CONF_THR:
                    continue
                cx, cy, bw, bh = det[:4] * IMGSZ
                x1, y1 = cx - bw/2, cy - bh/2
                x2, y2 = cx + bw/2, cy + bh/2
                preds.append({'cls': cls, 'conf': conf, 'box': [x1, y1, x2, y2]})

        all_preds.append(preds)
        all_gts.append(gt)

    iou_thrs = [round(t, 2) for t in np.arange(0.50, 1.00, 0.05)]
    map50, map5095 = compute_map(all_preds, all_gts, n_classes=4, iou_thresholds=iou_thrs)

    lat = np.array(latencies)
    print(f"  mAP@0.5      = {map50:.4f}")
    print(f"  mAP@0.5:0.95 = {map5095:.4f}")
    print(f"  Latency mean = {lat.mean():.2f} ms  (CPU-only TFLite on PC, NOT edge)")

    results_all[variant] = {
        "model_path": model_path,
        "size_mb": round(os.path.getsize(model_path)/1e6, 1),
        "map50": round(map50, 4),
        "map5095": round(map5095, 4),
        "n_images": len(all_preds),
        "note": "PC accuracy reference — on-device must match within ±0.01 mAP",
    }

out_path = os.path.join(OUT_DIR, 'step4_accuracy.json')
with open(out_path, 'w') as f:
    json.dump(results_all, f, indent=2)

print(f"\n[DONE] → {out_path}")
print("\n--- Table 1 preview (accuracy retention) ---")
print(f"{'Variant':8s}  {'mAP50':8s}  {'mAP50-95':10s}  {'Size(MB)':8s}")
for v, r in results_all.items():
    print(f"{v:8s}  {r['map50']:.4f}    {r['map5095']:.4f}      {r['size_mb']:.1f}")
