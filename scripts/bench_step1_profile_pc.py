"""
Step 1 — Static model metrics + FP32 accuracy (run on PC).

Outputs: bench_results/step1_profile.json
  - Parameters (M), GFLOPs at deployment resolution
  - Model file size (MB)
  - mAP@0.5, mAP@0.5:0.95, Precision, Recall per-class
  - PC GPU latency / FPS (warmup-corrected)

Usage:
    cd /home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics
    python scripts/bench_step1_profile_pc.py
"""

import sys, os, json, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultralytics import YOLO
from ultralytics.utils.torch_utils import model_info

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PT   = os.path.join(ROOT, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
DATA_YAML  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/data.yaml'
IMGSZ      = 640          # deployment resolution — ALL latency/GFLOPs reported at this size
WARMUP_N   = 50           # discard first N inferences
BENCH_N    = 300          # benchmark iterations
OUT_DIR    = os.path.join(ROOT, 'bench_results')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Device: {DEVICE}  |  Input: 1×4×{IMGSZ}×{IMGSZ}")

# ── Load model ───────────────────────────────────────────────────────────────
yolo = YOLO(MODEL_PT)
net  = yolo.model.to(DEVICE).eval()

# ── A. Parameters ────────────────────────────────────────────────────────────
n_params = sum(p.numel() for p in net.parameters())
print(f"\n[A] Parameters: {n_params/1e6:.3f} M")

# ── A. GFLOPs ────────────────────────────────────────────────────────────────
try:
    from ptflops import get_model_complexity_info
    dummy_fn = lambda: torch.zeros(4, IMGSZ, IMGSZ)  # single sample, CHW
    macs, params_ptf = get_model_complexity_info(
        net, (4, IMGSZ, IMGSZ),
        as_strings=False, print_per_layer_stat=False, verbose=False,
        input_constructor=lambda s: torch.zeros(1, *s).to(DEVICE)
    )
    gflops = macs * 2 / 1e9
    print(f"[A] GFLOPs: {gflops:.2f}  (MACs={macs/1e9:.2f} G)  at {IMGSZ}px")
except Exception as e:
    print(f"[WARN] ptflops failed ({e}), using ultralytics model_info fallback")
    gflops = None
    model_info(net, detailed=False, verbose=True, imgsz=IMGSZ)

# ── A. File size (FP32 .pt) ────────────────────────────────────────────────
size_fp32_mb = os.path.getsize(MODEL_PT) / 1e6
print(f"[A] File size FP32: {size_fp32_mb:.1f} MB")

# ── B. Accuracy — FP32 ───────────────────────────────────────────────────────
print("\n[B] Running validation (FP32) ...")
metrics = yolo.val(
    data=DATA_YAML,
    imgsz=IMGSZ,
    batch=1,
    device=DEVICE,
    split='test',
    verbose=True,
    conf=0.001,   # low threshold so mAP sweep covers full curve
    iou=0.6,
)

map50    = float(metrics.box.map50)
map5095  = float(metrics.box.map)
prec     = float(metrics.box.mp)      # mean precision at optimal threshold
rec      = float(metrics.box.mr)      # mean recall
ap50_per = metrics.box.ap50.tolist()  # per-class AP50
names    = list(yolo.names.values())

print(f"\n  mAP@0.5      = {map50:.4f}")
print(f"  mAP@0.5:0.95 = {map5095:.4f}")
print(f"  Precision    = {prec:.4f}")
print(f"  Recall       = {rec:.4f}")
for cls, ap in zip(names, ap50_per):
    print(f"  AP50 [{cls}]: {ap:.4f}")

# ── C. PC GPU latency / FPS ──────────────────────────────────────────────────
print(f"\n[C] Latency benchmark  warmup={WARMUP_N}  bench={BENCH_N}  device={DEVICE}")
dummy = torch.zeros(1, 4, IMGSZ, IMGSZ, device=DEVICE)

def sync():
    if DEVICE == 'cuda':
        torch.cuda.synchronize()

with torch.no_grad():
    for _ in range(WARMUP_N):
        net(dummy)
    sync()

    latencies_ms = []
    for _ in range(BENCH_N):
        sync()
        t0 = time.perf_counter()
        net(dummy)
        sync()
        latencies_ms.append((time.perf_counter() - t0) * 1000)

lat = np.array(latencies_ms)
print(f"  Mean     : {lat.mean():.2f} ms")
print(f"  Std      : {lat.std():.2f} ms")
print(f"  p50      : {np.percentile(lat, 50):.2f} ms")
print(f"  p95      : {np.percentile(lat, 95):.2f} ms")
print(f"  p99      : {np.percentile(lat, 99):.2f} ms")
print(f"  FPS      : {1000/lat.mean():.1f}")

# ── Save results ─────────────────────────────────────────────────────────────
results = {
    "model": MODEL_PT,
    "imgsz": IMGSZ,
    "variant": "FP32",
    "device_profile": str(DEVICE),
    "params_M": round(n_params / 1e6, 3),
    "gflops": round(gflops, 2) if gflops else None,
    "size_mb": round(size_fp32_mb, 1),
    "accuracy": {
        "map50": round(map50, 4),
        "map5095": round(map5095, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "ap50_per_class": {n: round(v, 4) for n, v in zip(names, ap50_per)},
    },
    "pc_latency": {
        "mean_ms": round(lat.mean(), 2),
        "std_ms": round(lat.std(), 2),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "fps": round(1000 / lat.mean(), 1),
        "warmup_n": WARMUP_N,
        "bench_n": BENCH_N,
    }
}

out_path = os.path.join(OUT_DIR, 'step1_profile.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n[DONE] Saved → {out_path}")
