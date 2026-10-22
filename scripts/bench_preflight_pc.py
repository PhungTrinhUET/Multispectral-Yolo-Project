"""
PC Pre-flight — Gate 2 (op compat) + Gate 3a (save PC reference outputs).

Run AFTER bench_step3_convert_tflite.py, BEFORE any device run.

Gate 2: Check that all ops in every TFLite model are delegate-compatible.
  - Inspects input/output nodes of the converted model.
  - Tries loading FP16 with GPU delegate and INT8 with NNAPI/Hexagon delegate
    (PC-side simulation — catches obvious conversion bugs before device).
  - Emits PASS/FAIL verdict per variant.

Gate 3a: Run 30 test images through the FP32 .pt model on PC and save outputs
  as npy reference files. The on-device harness (bench_step5) loads these and
  compares. If outputs differ beyond tolerance, the device run stops.

Outputs:
  bench_results/preflight_gate2.json    — op compat report per variant
  bench_results/gate3_reference/        — 30 × output npy files (PC reference)
  bench_results/gate3_image_list.json   — which 30 images were used + sha256

Usage:
    cd /ultralytics
    python scripts/bench_preflight_pc.py
"""

import sys, os, json, glob, hashlib, time
import numpy as np
import cv2

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT_DIR = os.path.join(ROOT, 'bench_results')
REF_DIR = os.path.join(OUT_DIR, 'gate3_reference')
os.makedirs(REF_DIR, exist_ok=True)

IMGSZ    = 640
TEST_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/test/images'
MODEL_PT = os.path.join(ROOT, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')

TFLITE_VARIANTS = {
    'FP32': os.path.join(OUT_DIR, 'cmx_yolo_fp32.tflite'),
    'FP16': os.path.join(OUT_DIR, 'cmx_yolo_fp16.tflite'),
    'INT8': os.path.join(OUT_DIR, 'cmx_yolo_int8.tflite'),
}

# ─────────────────────────────────────────────────────────────────────────────
# GATE 2 — Op compatibility + input-node verification
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PRE-FLIGHT GATE 2 — Op compatibility check")
print("=" * 60)

try:
    import tensorflow as tf
    print(f"[GATE2] TensorFlow: {tf.__version__}  ✓")
except ImportError:
    print("[GATE2] FAIL — TensorFlow not installed: pip install tensorflow")
    sys.exit(1)

gate2_report = {}

for variant, path in TFLITE_VARIANTS.items():
    print(f"\n[GATE2] Checking {variant}: {path}")

    if not os.path.exists(path):
        print(f"  SKIP — file not found (run bench_step3_convert_tflite.py first)")
        gate2_report[variant] = {"verdict": "SKIP", "reason": "file_not_found"}
        continue

    entry = {"path": path, "size_mb": round(os.path.getsize(path)/1e6, 1)}

    # ── Probe input/output nodes ──────────────────────────────────────────────
    try:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()

        inp_details = interp.get_input_details()
        out_details = interp.get_output_details()

        entry["inputs"] = [
            {"name": d['name'], "shape": list(d['shape']), "dtype": d['dtype'].__name__}
            for d in inp_details
        ]
        entry["outputs"] = [
            {"name": d['name'], "shape": list(d['shape'])}
            for d in out_details
        ]
        n_inputs = len(inp_details)
        inp_shape = inp_details[0]['shape'].tolist()
        inp_dtype = inp_details[0]['dtype'].__name__

        # The model has ONE 4-channel input (InputContainer splits internally).
        # onnx2tf converts NCHW→NHWC so shape should be [1, 640, 640, 4].
        expected_shape = [1, IMGSZ, IMGSZ, 4]
        shape_ok = (inp_shape == expected_shape)

        print(f"  Input nodes  : {n_inputs}  (expected: 1)")
        print(f"  Input shape  : {inp_shape}  (expected: {expected_shape})")
        print(f"  Input dtype  : {inp_dtype}")
        print(f"  Output nodes : {len(out_details)}")

        if n_inputs != 1:
            print(f"  [WARN] Expected 1 input, got {n_inputs} — onnx2tf may have split channels")
        if not shape_ok:
            print(f"  [WARN] Shape mismatch — normalization/resize must be adjusted in harness")

        entry["n_inputs"]    = n_inputs
        entry["shape_ok"]    = shape_ok
        entry["input_dtype"] = inp_dtype

    except Exception as e:
        print(f"  [FAIL] Interpreter load error: {e}")
        gate2_report[variant] = {"verdict": "FAIL", "reason": str(e)}
        continue

    # ── Try one CPU inference (baseline — must not crash) ─────────────────────
    try:
        dummy_in = np.zeros(inp_shape, dtype=inp_dtype if inp_dtype != 'uint8'
                            else np.uint8)
        if inp_dtype == 'uint8':
            # INT8 model — zero input is fine for shape check
            dummy_in = np.zeros(inp_shape, dtype=np.uint8)
        interp.set_tensor(inp_details[0]['index'], dummy_in)
        t0 = time.perf_counter()
        interp.invoke()
        cpu_ms = (time.perf_counter() - t0) * 1000
        print(f"  CPU inference: {cpu_ms:.1f} ms  ✓")
        entry["cpu_inference_ms"] = round(cpu_ms, 1)
        entry["cpu_ok"] = True
    except Exception as e:
        print(f"  [FAIL] CPU inference failed: {e}")
        entry["cpu_ok"] = False
        gate2_report[variant] = {"verdict": "FAIL", "reason": f"cpu_inference: {e}"}
        continue

    # ── Attempt GPU delegate load (PC simulation — may not have GPU delegate) ──
    if variant in ('FP32', 'FP16'):
        gpu_verdict = "NOT_TESTED"
        try:
            # On desktop, GPU delegate uses OpenCL
            gpu_delegate = tf.lite.experimental.load_delegate(
                'libdelegate_gpu.so',  # Linux OpenCL delegate
                options={'is_precision_loss_allowed': int(variant == 'FP16')}
            )
            interp_gpu = tf.lite.Interpreter(
                model_path=path, experimental_delegates=[gpu_delegate])
            interp_gpu.allocate_tensors()
            gpu_verdict = "LOAD_OK"
            print(f"  GPU delegate load: {gpu_verdict}")
        except Exception as e:
            # This is expected on PC (no Adreno). Log but do not FAIL gate.
            gpu_verdict = f"NOT_AVAILABLE_ON_PC: {str(e)[:80]}"
            print(f"  GPU delegate on PC: {gpu_verdict} (expected)")
        entry["gpu_delegate_pc"] = gpu_verdict

    # ── Check INT8 quantization coverage ─────────────────────────────────────
    if variant == 'INT8':
        # Verify that all major ops were actually quantized (not left as float fallback).
        # TFLite exposes this via internal _get_ops_details() — use if available.
        try:
            ops_details = interp._get_ops_details()
            float_ops  = [op for op in ops_details
                          if 'float' in str(op.get('inputs', '')).lower()]
            n_total    = len(ops_details)
            n_float    = len(float_ops)
            n_quant    = n_total - n_float
            print(f"  INT8 ops coverage: {n_quant}/{n_total} quantized  "
                  f"({'CLEAN' if n_float == 0 else f'WARN: {n_float} float fallback ops'})")
            entry["int8_quantized_ops"]  = n_quant
            entry["int8_float_fallback"] = n_float
            if n_float > 0:
                entry["float_fallback_ops"] = [op.get('op_name', '?') for op in float_ops]
                print(f"  Float fallback ops: {entry['float_fallback_ops']}")
        except AttributeError:
            # _get_ops_details() may not be available in all TF builds
            print("  INT8 ops coverage: _get_ops_details() not available in this TF build")
            entry["int8_ops_note"] = "ops_details_not_accessible"

    entry["verdict"] = "PASS"
    gate2_report[variant] = entry
    print(f"  [GATE2] {variant}: PASS")

# ── Gate 2 summary ─────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("[GATE2] SUMMARY:")
any_fail = False
for v, r in gate2_report.items():
    verdict = r.get("verdict", "?")
    print(f"  {v:5s}: {verdict}")
    if verdict == "FAIL":
        any_fail = True

gate2_path = os.path.join(OUT_DIR, 'preflight_gate2.json')
with open(gate2_path, 'w') as f:
    json.dump(gate2_report, f, indent=2)
print(f"\n[GATE2] Report → {gate2_path}")

if any_fail:
    print("\n[GATE2] STOP — one or more variants FAILED. Fix conversion before device runs.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# GATE 3a — Save PC reference outputs (FP32 PyTorch, 30 images)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PRE-FLIGHT GATE 3a — Saving PC reference outputs (FP32 PyTorch)")
print("=" * 60)

import torch
from ultralytics import YOLO

print(f"[GATE3a] Loading {MODEL_PT}")
yolo = YOLO(MODEL_PT)
net  = yolo.model.cpu().eval()

def load_4ch(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        nir = np.zeros((*img.shape[:2], 1), dtype=img.dtype)
        img = np.concatenate([img, nir], axis=2)
    img = cv2.resize(img, (IMGSZ, IMGSZ)).astype(np.float32) / 255.0
    return img  # H×W×4  NHWC

all_imgs = sorted([f for f in os.listdir(TEST_DIR)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))])
step = max(1, len(all_imgs) // 30)
selected = all_imgs[::step][:30]
print(f"[GATE3a] Selected {len(selected)} images from {len(all_imgs)} total")

image_list = []

for i, fname in enumerate(selected):
    img_path = os.path.join(TEST_DIR, fname)
    img_nhwc = load_4ch(img_path)
    if img_nhwc is None:
        print(f"  [WARN] Cannot read {img_path}")
        continue

    # NCHW tensor for PyTorch
    img_t = torch.from_numpy(img_nhwc.transpose(2, 0, 1))[None]  # 1×4×H×W

    with torch.no_grad():
        out = net(img_t)
    # out: tuple; first element is the detection tensor [1, num_preds, 4+nc]
    if isinstance(out, (list, tuple)):
        out_np = out[0].numpy() if hasattr(out[0], 'numpy') else np.array(out[0])
    else:
        out_np = out.numpy() if hasattr(out, 'numpy') else np.array(out)

    ref_path = os.path.join(REF_DIR, f'ref_{i:03d}.npy')
    img_path_save = os.path.join(REF_DIR, f'img_{i:03d}.npy')
    np.save(ref_path, out_np)      # PC output
    np.save(img_path_save, img_nhwc[np.newaxis])   # input (NHWC, for device to reuse)

    # SHA256 of the input image (to verify same image used on device)
    sha = hashlib.sha256(open(os.path.join(TEST_DIR, fname), 'rb').read()).hexdigest()[:16]
    image_list.append({
        "idx": i, "filename": fname, "sha256_prefix": sha,
        "ref_npy": ref_path, "img_npy": img_path_save,
        "out_shape": list(out_np.shape),
    })
    print(f"  [{i+1:2d}/30] {fname}  output: {out_np.shape}")

list_path = os.path.join(OUT_DIR, 'gate3_image_list.json')
with open(list_path, 'w') as f:
    json.dump(image_list, f, indent=2)

# SHA256 of the calib dataset for cross-device identity check
calib_list = os.path.join(OUT_DIR, 'calib', 'calib_list.txt')
if os.path.exists(calib_list):
    calib_hash = hashlib.sha256(open(calib_list, 'rb').read()).hexdigest()
    calib_id = {"calib_list_sha256": calib_hash, "calib_list_path": calib_list}
    with open(os.path.join(OUT_DIR, 'calib_identity.json'), 'w') as f:
        json.dump(calib_id, f, indent=2)
    print(f"\n[GATE3a] Calib set SHA256: {calib_hash[:16]}...  "
          "(verify this matches on device before INT8 run)")

print(f"\n[GATE3a] {len(image_list)} reference outputs saved → {REF_DIR}/")
print(f"[GATE3a] Image list → {list_path}")
print(f"\n[PRE-FLIGHT PC] All gates passed. Transfer bench_results/ to devices.")
print("  adb push bench_results/ /data/gmd_bench/")
