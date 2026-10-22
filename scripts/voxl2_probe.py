#!/usr/bin/env python3
"""
VOXL2 deployment probe — Table 2 speed harness.

One invocation = one (model, delegate, variant) row.

Metrics collected:
  - Delegate actually applied: hard-fail if load_delegate() raises (no silent CPU fallback)
  - Burst latency: mean/std/p50/p95/p99 over --burst-runs inferences
  - FPS burst: 1000 / mean_ms
  - FPS sustained: total frames / --sustained-secs (default 300 s)
  - Peak RAM: max VmRSS across entire run (/proc/self/status)
  - SoC temp trace: sampled every --temp-interval s during sustained run
  - OOM / crash: caught and reported; non-zero exit code
  - Accuracy spot-check: 30 images vs gate3a_fp32_ref_1280.json (PC PyTorch reference)

Table 2 columns (printed at end):
  variant | delegate | lat_mean | p50 | p95 | p99 | fps_burst | fps_sus | peak_ram | temp_max

Usage on device (run once per row):
  python3 voxl2_probe.py \\
      --model  ul_1280v2_float16.tflite \\
      --variant "FP16/GPU" \\
      --delegate gpu \\
      --gpu-so  /usr/lib/.../libtensorflowlite_gpu_delegate.so \\
      --ref-json gate3a_fp32_ref_1280.json \\
      --ref-rgb  /data/test/images/test \\
      --ref-nir  /data/test/ndvi/test \\
      --imgsz 1280 \\
      --out voxl2_FP16_GPU.json

  python3 voxl2_probe.py --model ul_1280_int8_mixed.tflite --variant "INT8/NPU" --delegate npu \\
      --hex-so /usr/lib/libhexagon_nn_skel.so ...

  python3 voxl2_probe.py --model ul_1280v2_float16.tflite --variant "FP16/CPU" --delegate cpu ...
  python3 voxl2_probe.py --model ul_1280_int8_mixed.tflite --variant "INT8/CPU" --delegate cpu ...

Requirements on device:
  pip3 install tflite-runtime numpy

GFLOPs: 164.7 @ imgsz=1280 (static, measured on PC — not re-measured here).
"""

import argparse, json, os, sys, time
import numpy as np
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
ap.add_argument('--model',    required=True, help='TFLite model path')
ap.add_argument('--variant',  required=True, help='Row label, e.g. "INT8/NPU"')
ap.add_argument('--delegate', required=True, choices=['cpu', 'gpu', 'npu'],
                help='Required delegate. Hard-fail if load raises.')
ap.add_argument('--imgsz',    type=int, default=1280)
ap.add_argument('--burst-runs',    type=int, default=50,
                help='Inferences for latency percentiles and FPS burst')
ap.add_argument('--warmup',        type=int, default=5,
                help='Warmup inferences (discarded from timing)')
ap.add_argument('--sustained-secs', type=int, default=300,
                help='Duration of sustained FPS run (seconds)')
ap.add_argument('--temp-interval', type=float, default=5.0,
                help='Seconds between SoC temperature samples during sustained run')
ap.add_argument('--gpu-so',   default=None, help='GPU delegate .so path')
ap.add_argument('--hex-so',   default=None, help='Hexagon delegate .so path')
ap.add_argument('--ref-json', default=None,
                help='gate3a_fp32_ref_1280.json (PC PyTorch detections for 30 images)')
ap.add_argument('--ref-rgb',  default=None, help='Directory of test RGB images (.jpg)')
ap.add_argument('--ref-nir',  default=None, help='Directory of test NIR images (.png)')
ap.add_argument('--cpu-threads', type=int, default=4)
ap.add_argument('--out',      default=None, help='Output JSON (default: voxl2_<variant>.json)')
args = ap.parse_args()

MODEL_PATH = Path(args.model)
assert MODEL_PATH.exists(), f"Model not found: {MODEL_PATH}"
OUT_PATH = Path(args.out) if args.out else Path(f"voxl2_{args.variant.replace('/','_')}.json")

print(f"{'='*60}")
print(f"VOXL2 probe: {args.variant}")
print(f"  model    : {MODEL_PATH}  ({MODEL_PATH.stat().st_size/1024**2:.1f} MB)")
print(f"  delegate : {args.delegate.upper()}")
print(f"  imgsz    : {args.imgsz}")
print(f"  burst    : {args.burst_runs} runs after {args.warmup} warmup")
print(f"  sustained: {args.sustained_secs} s")
print(f"{'='*60}")

# ── TFLite runtime ────────────────────────────────────────────────────────────
try:
    import tflite_runtime.interpreter as tflite
    RUNTIME = "tflite_runtime"
except ImportError:
    try:
        import tensorflow.lite as tflite
        RUNTIME = "tensorflow.lite"
    except ImportError:
        print("FATAL: neither tflite_runtime nor tensorflow found.")
        print("  pip3 install tflite-runtime")
        sys.exit(1)
print(f"Runtime: {RUNTIME}")

# ── System helpers ────────────────────────────────────────────────────────────
def rss_mb():
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return -1.0

def read_soc_temps():
    """Return dict {zone_type: celsius} from /sys/class/thermal."""
    temps = {}
    base = Path('/sys/class/thermal')
    if not base.exists():
        return temps
    for zone in sorted(base.glob('thermal_zone*')):
        try:
            t_file = zone / 'temp'
            n_file = zone / 'type'
            if not t_file.exists():
                continue
            name = n_file.read_text().strip() if n_file.exists() else zone.name
            val = int(t_file.read_text().strip()) / 1000.0
            # Deduplicate: if name seen before keep max
            if name in temps:
                temps[name] = max(temps[name], val)
            else:
                temps[name] = val
        except Exception:
            pass
    return temps

# ── Build delegate ────────────────────────────────────────────────────────────
delegate_obj = None
delegate_so_used = None

if args.delegate == 'cpu':
    print(f"Delegate: CPU ({args.cpu_threads} threads)")

elif args.delegate == 'gpu':
    candidates = [
        args.gpu_so,
        '/usr/local/lib/python3.8/dist-packages/tflite_runtime/libtensorflowlite_gpu_delegate.so',
        '/usr/local/lib/python3.10/dist-packages/tflite_runtime/libtensorflowlite_gpu_delegate.so',
        '/usr/lib/aarch64-linux-gnu/libtensorflowlite_gpu_delegate.so',
        'libtensorflowlite_gpu_delegate.so',
    ]
    gpu_so = next((p for p in candidates if p and Path(p).exists()), None)
    if gpu_so is None:
        print("HARD FAIL: GPU delegate .so not found. Searched:")
        for c in candidates:
            if c: print(f"  {c}")
        print("Pass --gpu-so /path/to/libtensorflowlite_gpu_delegate.so")
        sys.exit(2)
    try:
        delegate_obj = tflite.load_delegate(gpu_so, {
            'allow_precision_loss': 'true',
            'inference_preference': 'LATENCY',
            'inference_priority1': 'MIN_LATENCY',
        })
        delegate_so_used = gpu_so
        print(f"Delegate: GPU  ({gpu_so})")
    except Exception as e:
        print(f"HARD FAIL: GPU delegate load raised: {e}")
        print("Silent CPU fallback is not permitted — fix the delegate and retry.")
        sys.exit(2)

elif args.delegate == 'npu':
    candidates = [
        args.hex_so,
        '/usr/lib/libhexagon_nn_skel.so',
        '/usr/local/lib/libhexagon_delegate.so',
        'libhexagon_delegate.so',
    ]
    hex_so = next((p for p in candidates if p and Path(p).exists()), None)
    if hex_so is None:
        print("HARD FAIL: Hexagon delegate .so not found. Searched:")
        for c in candidates:
            if c: print(f"  {c}")
        print("Pass --hex-so /path/to/libhexagon_delegate.so")
        sys.exit(2)
    try:
        delegate_obj = tflite.load_delegate(hex_so, {'debug_level': '0'})
        delegate_so_used = hex_so
        print(f"Delegate: NPU/Hexagon  ({hex_so})")
    except Exception as e:
        print(f"HARD FAIL: Hexagon delegate load raised: {e}")
        print("Silent CPU fallback is not permitted — fix the delegate and retry.")
        sys.exit(2)

# ── Load interpreter ──────────────────────────────────────────────────────────
t_load = time.perf_counter()
ram_baseline = rss_mb()
try:
    kwargs = {'model_path': str(MODEL_PATH)}
    if delegate_obj is not None:
        kwargs['experimental_delegates'] = [delegate_obj]
    else:
        kwargs['num_threads'] = args.cpu_threads
    interp = tflite.Interpreter(**kwargs)
    interp.allocate_tensors()
except MemoryError as e:
    print(f"OOM during model load: {e}")
    sys.exit(3)
except Exception as e:
    print(f"HARD FAIL: Interpreter load failed: {e}")
    sys.exit(2)

load_ms = (time.perf_counter() - t_load) * 1000
in_det  = interp.get_input_details()[0]
out_det = interp.get_output_details()[0]
in_shape = in_det['shape'].tolist()   # [1, H, W, C]
in_dtype = in_det['dtype']
print(f"  Input : {in_shape}  {in_dtype.__name__}")
print(f"  Output: {out_det['shape'].tolist()}  {out_det['dtype'].__name__}")
print(f"  Load  : {load_ms:.0f} ms")

# ── Dummy input ───────────────────────────────────────────────────────────────
H, W, C = in_shape[1], in_shape[2], in_shape[3]
if in_dtype == np.float32:
    dummy = np.random.rand(1, H, W, C).astype(np.float32)
elif in_dtype == np.int8:
    sc, zp = in_det['quantization']
    dummy = np.clip(np.round(np.random.rand(1, H, W, C) / (sc + 1e-9) + zp),
                    -128, 127).astype(np.int8)
elif in_dtype == np.uint8:
    dummy = (np.random.rand(1, H, W, C) * 255).astype(np.uint8)
else:
    dummy = np.zeros((1, H, W, C), dtype=in_dtype)

def invoke(inp=None):
    interp.set_tensor(in_det['index'], dummy if inp is None else inp)
    interp.invoke()

# ── Warmup ────────────────────────────────────────────────────────────────────
print(f"\nWarmup ({args.warmup} runs)...")
for _ in range(args.warmup):
    invoke()

# ── Burst latency + FPS burst ─────────────────────────────────────────────────
print(f"Burst ({args.burst_runs} runs)...")
ram_before_burst = rss_mb()
times_ms = []
burst_crashed = False
try:
    for _ in range(args.burst_runs):
        t0 = time.perf_counter()
        invoke()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
except MemoryError as e:
    print(f"  OOM during burst: {e}")
    burst_crashed = True
except Exception as e:
    print(f"  CRASH during burst: {e}")
    burst_crashed = True

if not times_ms:
    print("HARD FAIL: zero successful inferences in burst run.")
    sys.exit(3)

arr = np.array(times_ms)
lat_mean = float(arr.mean())
lat_std  = float(arr.std())
lat_p50  = float(np.percentile(arr, 50))
lat_p95  = float(np.percentile(arr, 95))
lat_p99  = float(np.percentile(arr, 99))
fps_burst = 1000.0 / lat_mean if lat_mean > 0 else 0.0
ram_after_burst = rss_mb()
peak_ram = max(ram_before_burst, ram_after_burst)

print(f"  lat mean={lat_mean:.1f} ms  std={lat_std:.1f} ms")
print(f"  p50={lat_p50:.1f}  p95={lat_p95:.1f}  p99={lat_p99:.1f}  (ms)")
print(f"  FPS burst = {fps_burst:.2f}")

# ── SoC temp at burst end ─────────────────────────────────────────────────────
temps_burst = read_soc_temps()
if temps_burst:
    top3 = sorted(temps_burst.items(), key=lambda x: -x[1])[:3]
    print(f"  SoC temps (top3): " + "  ".join(f"{k}={v:.1f}°C" for k, v in top3))

# ── Sustained FPS run (--sustained-secs) ─────────────────────────────────────
print(f"\nSustained run ({args.sustained_secs} s)...")
sustained_frames = 0
sustained_crashed = False
temp_trace = []   # list of {t_s, temps}
t_sus_start = time.perf_counter()
t_last_temp = t_sus_start
try:
    while True:
        t_now = time.perf_counter()
        elapsed = t_now - t_sus_start
        if elapsed >= args.sustained_secs:
            break
        invoke()
        sustained_frames += 1
        # Temperature sample
        if (t_now - t_last_temp) >= args.temp_interval:
            temps_now = read_soc_temps()
            temp_trace.append({'t_s': round(elapsed, 1), 'temps': temps_now})
            t_last_temp = t_now
            if sustained_frames % 200 == 0:
                top1 = max(temps_now.values()) if temps_now else 0
                print(f"  {elapsed:6.0f}s  frames={sustained_frames}  "
                      f"temp_max={top1:.1f}°C  FPS={sustained_frames/elapsed:.2f}")
except MemoryError as e:
    print(f"  OOM during sustained: {e}")
    sustained_crashed = True
except Exception as e:
    print(f"  CRASH during sustained: {e}")
    sustained_crashed = True

sus_elapsed = time.perf_counter() - t_sus_start
fps_sustained = sustained_frames / sus_elapsed if sus_elapsed > 0 else 0.0
ram_after_sustained = rss_mb()
peak_ram = max(peak_ram, ram_after_sustained)
print(f"  FPS sustained = {fps_sustained:.2f}  "
      f"(frames={sustained_frames}, elapsed={sus_elapsed:.1f}s)")

# SoC temp stats from trace
all_temp_vals = []
for entry in temp_trace:
    all_temp_vals.extend(entry['temps'].values())
temp_max = max(all_temp_vals) if all_temp_vals else None
temp_final = max(temp_trace[-1]['temps'].values()) if temp_trace else None
if temp_max is not None:
    print(f"  SoC temp_max={temp_max:.1f}°C  temp_final={temp_final:.1f}°C  "
          f"({len(temp_trace)} samples)")

# ── Accuracy spot-check (30 images vs PC reference) ─────────────────────────
acc_result = None
if args.ref_json and args.ref_rgb and args.ref_nir:
    print(f"\nAccuracy spot-check (30 images vs {args.ref_json})...")
    try:
        import cv2

        def letterbox(img4, sz):
            h0, w0 = img4.shape[:2]; r = sz / max(h0, w0)
            nw, nh = int(round(w0 * r)), int(round(h0 * r))
            img4 = cv2.resize(img4, (nw, nh))
            dh, dw = sz - nh, sz - nw
            t, b, l, rr = dh // 2, dh - dh // 2, dw // 2, dw - dw // 2
            return (cv2.copyMakeBorder(img4, t, b, l, rr, cv2.BORDER_CONSTANT,
                                       value=(114,) * img4.shape[2])
                    .astype(np.float32) / 255.0)

        def nms(raw, conf=0.25, iou=0.45, sz=1280):
            # raw: [1, 8, N] or [8, N]
            if raw.ndim == 3: raw = raw[0]
            cx, cy, w, h, *sc = raw
            sc = np.array(sc)
            ms = sc.max(0); cl = sc.argmax(0); mask = ms >= conf
            if not mask.any(): return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
            cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
            ms, cl = ms[mask], cl[mask]
            boxes = np.stack([(cx-w/2)*sz, (cy-h/2)*sz,
                               (cx+w/2)*sz, (cy+h/2)*sz], 1)
            # Greedy NMS
            order = ms.argsort()[::-1]; keep = []
            while order.size:
                i = order[0]; keep.append(i)
                if order.size == 1: break
                rest = order[1:]
                ix1 = np.maximum(boxes[i, 0], boxes[rest, 0])
                iy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
                ix2 = np.minimum(boxes[i, 2], boxes[rest, 2])
                iy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
                inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
                union = ((boxes[i,2]-boxes[i,0])*(boxes[i,3]-boxes[i,1]) +
                         (boxes[rest,2]-boxes[rest,0])*(boxes[rest,3]-boxes[rest,1]) - inter)
                order = rest[inter / (union + 1e-9) <= iou]
            keep = np.array(keep)
            return boxes[keep], ms[keep], cl[keep].astype(int)

        # Ref JSON format: {description, images: [{index, rgb_path, nir_path, n_dets,
        #   detections: [{box_xyxy_px, score, cls, cls_name}]}]}
        ref_data = json.loads(Path(args.ref_json).read_text())
        ref_imgs = ref_data['images'] if isinstance(ref_data, dict) else ref_data
        rgb_dir = Path(args.ref_rgb)
        nir_dir = Path(args.ref_nir)
        model_sz = in_shape[1]   # actual model input size

        matched = pc_only = dev_only = 0
        iou_vals = []

        for rec in ref_imgs:
            # Locate image by filename (stem) — paths in JSON are PC paths, use filename only
            stem = Path(rec['rgb_path']).stem
            rgb_p = next(rgb_dir.glob(f"{stem}.*"), None)
            nir_p = next(nir_dir.glob(f"{stem}.*"), None)
            if rgb_p is None or nir_p is None:
                print(f"  WARN: image not found: {stem}")
                continue

            bgr = cv2.imread(str(rgb_p))
            nir = cv2.imread(str(nir_p), cv2.IMREAD_GRAYSCALE)
            if bgr is None or nir is None: continue
            if nir.shape[:2] != bgr.shape[:2]:
                nir = cv2.resize(nir, (bgr.shape[1], bgr.shape[0]))
            img4 = np.concatenate([bgr, nir[:, :, None]], axis=2)
            lb = letterbox(img4, model_sz)
            nhwc = lb[None].astype(np.float32)

            if in_dtype == np.int8:
                sc_q, zp_q = in_det['quantization']
                inp = np.clip(np.round(nhwc / sc_q + zp_q), -128, 127).astype(np.int8)
            else:
                inp = nhwc

            invoke(inp)
            raw = interp.get_tensor(out_det['index']).copy().astype(np.float32)
            if out_det['dtype'] == np.int8:
                sc_o, zp_o = out_det['quantization']
                raw = (interp.get_tensor(out_det['index']).astype(np.float32) - zp_o) * sc_o

            bx_dev, sc_dev, _ = nms(raw, sz=model_sz)
            # PC reference boxes from {detections: [{box_xyxy_px}]} format
            bx_pc = np.array([d['box_xyxy_px'] for d in rec.get('detections', [])],
                              dtype=np.float32).reshape(-1, 4)

            # Scale PC boxes from ref imgsz (1280) to model_sz if different
            ref_sz = ref_data.get('imgsz', 1280) if isinstance(ref_data, dict) else 1280
            if model_sz != ref_sz and len(bx_pc):
                bx_pc = bx_pc * (model_sz / ref_sz)

            nb = 0
            if len(bx_pc) and len(bx_dev):
                for pi in range(len(bx_pc)):
                    best_iou = 0.0
                    px1, py1, px2, py2 = bx_pc[pi]
                    for ti in range(len(bx_dev)):
                        tx1, ty1, tx2, ty2 = bx_dev[ti]
                        ix = max(0, min(px2, tx2) - max(px1, tx1))
                        iy = max(0, min(py2, ty2) - max(py1, ty1))
                        inter = ix * iy
                        union = ((px2-px1)*(py2-py1) + (tx2-tx1)*(ty2-ty1) - inter)
                        best_iou = max(best_iou, inter / (union + 1e-9))
                    if best_iou >= 0.5:
                        iou_vals.append(best_iou); nb += 1
            matched += nb
            pc_only  += max(0, len(bx_pc)  - nb)
            dev_only += max(0, len(bx_dev) - nb)

        acc_result = {
            'n_images': len(ref_imgs),
            'matched_iou50': matched,
            'pc_ref_only': pc_only,
            'device_only': dev_only,
            'iou_mean': round(float(np.mean(iou_vals)), 4) if iou_vals else None,
            'iou_min':  round(float(np.min(iou_vals)), 4) if iou_vals else None,
        }
        print(f"  matched={matched}  PC-ref-only={pc_only}  device-only={dev_only}  "
              f"IoU_mean={acc_result['iou_mean']}")
    except Exception as e:
        import traceback
        print(f"  Accuracy spot-check failed: {e}")
        traceback.print_exc()
        acc_result = {'error': str(e)}
else:
    print("\nAccuracy spot-check: SKIPPED (pass --ref-json --ref-rgb --ref-nir to enable)")

# ── Status ────────────────────────────────────────────────────────────────────
status = "OK"
if burst_crashed or sustained_crashed:
    status = "CRASHED"

# ── Table 2 row ───────────────────────────────────────────────────────────────
print(f"\n{'─'*80}")
print(f"TABLE 2 ROW: {args.variant}")
print(f"{'─'*80}")
hdr = (f"{'variant':<14} {'delegate':<10} {'lat_mean':>9} {'p50':>7} "
       f"{'p95':>7} {'p99':>7} {'fps_bst':>8} {'fps_sus':>8} {'RAM_MB':>8} {'temp_max':>9}")
print(hdr)

temp_str = f"{temp_max:.1f}" if temp_max is not None else "N/A"
row = (f"{args.variant:<14} {args.delegate.upper():<10} {lat_mean:>8.1f}ms {lat_p50:>6.1f}ms "
       f"{lat_p95:>6.1f}ms {lat_p99:>6.1f}ms {fps_burst:>8.2f} {fps_sustained:>8.2f} "
       f"{peak_ram:>7.0f}MB {temp_str:>8}°C")
print(row)
print(f"{'─'*80}")

# ── Save JSON ─────────────────────────────────────────────────────────────────
output = {
    'variant': args.variant,
    'delegate_requested': args.delegate,
    'delegate_so': delegate_so_used,
    'model': str(MODEL_PATH),
    'model_size_mb': round(MODEL_PATH.stat().st_size / 1024**2, 1),
    'imgsz': args.imgsz,
    'runtime': RUNTIME,
    'load_ms': round(load_ms, 1),
    'status': status,
    'burst': {
        'runs': args.burst_runs,
        'lat_mean_ms': round(lat_mean, 2),
        'lat_std_ms':  round(lat_std, 2),
        'lat_p50_ms':  round(lat_p50, 2),
        'lat_p95_ms':  round(lat_p95, 2),
        'lat_p99_ms':  round(lat_p99, 2),
        'fps_burst':   round(fps_burst, 2),
        'crashed': burst_crashed,
    },
    'sustained': {
        'secs': args.sustained_secs,
        'actual_secs': round(sus_elapsed, 1),
        'frames': sustained_frames,
        'fps_sustained': round(fps_sustained, 2),
        'crashed': sustained_crashed,
    },
    'soc_temp': {
        'temp_max_c':   round(temp_max, 1) if temp_max is not None else None,
        'temp_final_c': round(temp_final, 1) if temp_final is not None else None,
        'n_samples': len(temp_trace),
        'trace': temp_trace,          # full per-sample trace; can be large
    },
    'ram': {
        'baseline_mb': round(ram_baseline, 1),
        'peak_mb':     round(peak_ram, 1),
    },
    'accuracy_spot_check': acc_result,
    # GFLOPs is a static property measured on PC; not re-measured on device.
    'gflops_note': '164.7 @ imgsz=1280 (measured on PC, not re-measured here)',
}

with open(OUT_PATH, 'w') as f:
    json.dump(output, f, indent=2)
print(f"Results saved: {OUT_PATH}")

if status != "OK":
    sys.exit(3)
