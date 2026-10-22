"""
Step 3b — Convert ONNX → TFLite (FP32 / FP16 / INT8) with Gate 2 op-compat check.

Conversion chain:
  ONNX → TF SavedModel (via onnx2tf)  [stderr captured + scanned for warnings]
  TF SavedModel → TFLite FP32 / FP16 / INT8

Gate 2 is EMBEDDED here: after each TFLite conversion, the model is probed for:
  - Input/output node count and shapes
  - INT8 quantization coverage (float fallback ops reported loudly)
  - One CPU inference to confirm no crash

Requires:
  pip install onnx2tf tensorflow>=2.13

Outputs (all in bench_results/):
  cmx_yolo_fp32.tflite
  cmx_yolo_fp16.tflite
  cmx_yolo_int8.tflite
  step3_sizes.json
  step3_opcompat.json    ← Gate 2 report (read by bench_preflight_pc.py)

Usage:
    cd /ultralytics
    python scripts/bench_step3_convert_tflite.py
"""

import sys, os, json, subprocess, io
import numpy as np

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, 'bench_results')
CALIB_NPY_DIR = os.path.join(OUT_DIR, 'calib', 'calib_npy')

ONNX_SIM        = os.path.join(OUT_DIR, 'cmx_yolo_fp32_sim.onnx')
SAVED_MODEL_DIR = os.path.join(OUT_DIR, 'cmx_saved_model')
TFL_FP32        = os.path.join(OUT_DIR, 'cmx_yolo_fp32.tflite')
TFL_FP16        = os.path.join(OUT_DIR, 'cmx_yolo_fp16.tflite')
TFL_INT8        = os.path.join(OUT_DIR, 'cmx_yolo_int8.tflite')
IMGSZ           = 640

try:
    import tensorflow as tf
    print(f"[INFO] TensorFlow {tf.__version__}  ✓")
except ImportError:
    print("[ERROR] TensorFlow not installed: pip install tensorflow")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Prerequisite check
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(ONNX_SIM):
    print(f"[ERROR] ONNX file not found: {ONNX_SIM}")
    print("  Run bench_step2_export_onnx.py first")
    sys.exit(1)

ret = subprocess.run(['onnx2tf', '--version'], capture_output=True, text=True)
if ret.returncode != 0:
    print("[ERROR] onnx2tf not found: pip install onnx2tf")
    sys.exit(1)
print(f"[INFO] onnx2tf version: {ret.stdout.strip() or ret.stderr.strip()}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. ONNX → TF SavedModel  (capture stderr to scan for warnings)
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.isdir(SAVED_MODEL_DIR):
    print(f"\n[STEP3] Converting ONNX → TF SavedModel ...")
    log_path = os.path.join(OUT_DIR, 'onnx2tf_log.txt')
    with open(log_path, 'w') as logf:
        result = subprocess.run(
            ['onnx2tf',
             '-i', ONNX_SIM,
             '-o', SAVED_MODEL_DIR,
             '--non_verbose',
             '--output_signaturedefs'],
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # Scan log for warning keywords
    WARN_KEYWORDS = [
        'unsupported', 'not supported', 'fallback', 'decomposed',
        'failed', 'error', 'skip', 'ignored',
    ]
    warnings_found = []
    with open(log_path) as lf:
        for lineno, line in enumerate(lf, 1):
            lower = line.lower()
            if any(k in lower for k in WARN_KEYWORDS):
                warnings_found.append(f"L{lineno}: {line.rstrip()}")

    if result.returncode != 0:
        print(f"[ERROR] onnx2tf failed (exit {result.returncode}) — check {log_path}")
        sys.exit(1)

    if warnings_found:
        print(f"\n{'!'*60}")
        print(f"[GATE2 WARNING] onnx2tf emitted {len(warnings_found)} suspicious lines:")
        for w in warnings_found[:20]:
            print(f"  {w}")
        if len(warnings_found) > 20:
            print(f"  ... and {len(warnings_found)-20} more (see {log_path})")
        print(f"{'!'*60}")
        print("[GATE2] Review warnings above. If they are op-decomposition warnings")
        print("        (not errors), conversion may still be correct — verify with")
        print("        bench_preflight_pc.py Gate 2 report before proceeding.")
    else:
        print(f"[GATE2] onnx2tf log: no warning keywords found  ✓")

    print(f"  Log → {log_path}")
else:
    print(f"[INFO] SavedModel already exists: {SAVED_MODEL_DIR} (delete to re-convert)")

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Gate 2 probe — run after EACH variant conversion
# ─────────────────────────────────────────────────────────────────────────────
def probe_tflite(path, variant_label):
    """Probe TFLite model and emit PASS/FAIL verdict. Returns report dict."""
    print(f"\n[GATE2 PROBE] {variant_label}: {path}")
    report = {"path": path, "size_mb": round(os.path.getsize(path)/1e6, 1)}

    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()

    inp = interp.get_input_details()
    out = interp.get_output_details()

    n_inputs    = len(inp)
    inp_shape   = inp[0]['shape'].tolist()
    inp_dtype   = inp[0]['dtype'].__name__
    out_shapes  = [o['shape'].tolist() for o in out]
    expected_in = [1, IMGSZ, IMGSZ, 4]   # NHWC after onnx2tf

    report.update({
        "n_inputs": n_inputs, "input_shape": inp_shape,
        "input_dtype": inp_dtype, "output_shapes": out_shapes,
    })

    shape_match = (inp_shape == expected_in)
    print(f"  Input  : {n_inputs} node(s)  shape={inp_shape}  dtype={inp_dtype}")
    print(f"  Output : {len(out)} node(s)  shapes={out_shapes}")

    if not shape_match:
        msg = f"Input shape {inp_shape} != expected {expected_in}"
        print(f"  [WARN] {msg}")
        print("  If onnx2tf split the 4ch input into separate nodes, update")
        print("  the harness feeding logic to match the ACTUAL input structure.")
        report["shape_warning"] = msg
    else:
        print(f"  Shape: {inp_shape} == expected {expected_in}  ✓")

    if n_inputs != 1:
        print(f"  [WARN] Expected 1 input, got {n_inputs}. Harness must feed all {n_inputs} inputs.")
        report["multi_input_warning"] = f"model has {n_inputs} inputs"

    # One CPU inference (shape/crash check)
    dummy = (np.zeros(inp_shape, dtype=np.uint8)
             if inp_dtype == 'uint8'
             else np.zeros(inp_shape, dtype=np.float32))
    interp.set_tensor(inp[0]['index'], dummy)
    interp.invoke()
    print(f"  CPU inference: OK  ✓")
    report["cpu_inference"] = "PASS"

    # INT8: check float fallback ops
    if variant_label == 'INT8':
        try:
            ops = interp._get_ops_details()
            n_total = len(ops)
            float_ops = [op for op in ops if 'float' in str(op).lower()]
            n_float = len(float_ops)
            if n_float == 0:
                print(f"  INT8 quantization: {n_total} ops, 0 float fallbacks  ✓")
                report["int8_ops_clean"] = True
            else:
                float_names = [op.get('op_name', '?') for op in float_ops]
                print(f"  [WARN] INT8 quantization: {n_float}/{n_total} ops run as float:")
                for nm in float_names:
                    print(f"    {nm}")
                report["int8_ops_clean"]    = False
                report["int8_float_count"]  = n_float
                report["int8_float_ops"]    = float_names
        except AttributeError:
            print("  INT8 ops coverage: _get_ops_details() not exposed in this TF build")
            report["int8_ops_note"] = "not_accessible"

    # Quantization scale/zp check for INT8 input
    if inp_dtype == 'uint8':
        scale, zp = inp[0]['quantization']
        print(f"  INT8 input quant: scale={scale:.6f}  zero_point={zp}")
        if scale == 0.0:
            print("  [FAIL] Input quantization scale=0 — INT8 conversion failed silently")
            report["verdict"] = "FAIL"
            report["fail_reason"] = "input_scale_zero"
            return report
        report["int8_input_scale"] = scale
        report["int8_input_zp"]    = zp

    report["verdict"] = "PASS"
    print(f"  [GATE2] {variant_label}: PASS  ✓")
    return report

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Calibration dataset generator
# ─────────────────────────────────────────────────────────────────────────────
def make_rep_dataset(npy_dir, n=200):
    files = sorted([f for f in os.listdir(npy_dir) if f.endswith('.npy')])[:n]
    def gen():
        for fname in files:
            arr = np.load(os.path.join(npy_dir, fname))  # 1×H×W×4 NHWC float32
            yield [arr]
    return gen

# ─────────────────────────────────────────────────────────────────────────────
# 2. FP32 TFLite
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP3] Converting FP32 TFLite ...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
tflite_model = converter.convert()
with open(TFL_FP32, 'wb') as f:
    f.write(tflite_model)
fp32_report = probe_tflite(TFL_FP32, 'FP32')
if fp32_report["verdict"] == "FAIL":
    print("[ERROR] FP32 probe FAIL — fix before continuing")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 3. FP16 TFLite
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP3] Converting FP16 TFLite ...")
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()
with open(TFL_FP16, 'wb') as f:
    f.write(tflite_model)
fp16_report = probe_tflite(TFL_FP16, 'FP16')
if fp16_report["verdict"] == "FAIL":
    print("[ERROR] FP16 probe FAIL — fix before continuing")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. INT8 TFLite
# ─────────────────────────────────────────────────────────────────────────────
print("\n[STEP3] Converting INT8 TFLite ...")
if not os.path.isdir(CALIB_NPY_DIR):
    print(f"[ERROR] Calibration data missing: {CALIB_NPY_DIR}")
    print("  Run bench_step3_make_calib.py first")
    sys.exit(1)

n_calib = len([f for f in os.listdir(CALIB_NPY_DIR) if f.endswith('.npy')])
print(f"  Calibration images: {n_calib}")
if n_calib < 50:
    print(f"  [WARN] Only {n_calib} calibration images — recommend ≥100 for stable INT8 quality")

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = make_rep_dataset(CALIB_NPY_DIR)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,   # float fallback for ops that can't quantize
]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.float32   # float output preserves mAP eval accuracy
tflite_model = converter.convert()
with open(TFL_INT8, 'wb') as f:
    f.write(tflite_model)

int8_report = probe_tflite(TFL_INT8, 'INT8')
if int8_report["verdict"] == "FAIL":
    print("[ERROR] INT8 probe FAIL — fix before continuing")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Save reports
# ─────────────────────────────────────────────────────────────────────────────
sizes = {}
opcompat = {}
for label, path, report in [
    ('FP32', TFL_FP32, fp32_report),
    ('FP16', TFL_FP16, fp16_report),
    ('INT8', TFL_INT8, int8_report),
]:
    if os.path.exists(path):
        sizes[label]    = {"path": path, "size_mb": round(os.path.getsize(path)/1e6, 1)}
        opcompat[label] = report

with open(os.path.join(OUT_DIR, 'step3_sizes.json'), 'w') as f:
    json.dump(sizes, f, indent=2)
with open(os.path.join(OUT_DIR, 'step3_opcompat.json'), 'w') as f:
    json.dump(opcompat, f, indent=2)

print("\n" + "="*60)
print("[STEP3] DONE — TFLite variants:")
print(f"  {'Variant':6s}  {'Size':8s}  {'Gate2':6s}")
for label, info in sizes.items():
    verdict = opcompat.get(label, {}).get('verdict', '?')
    print(f"  {label:6s}  {info['size_mb']:5.1f} MB  {verdict}")

print(f"\nNext: python scripts/bench_preflight_pc.py  (Gate 2 full report + save reference outputs)")
