"""
Step 2 — Export best.pt → ONNX (with tracing verification).

The dual-stream routing in _predict_once uses Python-level control flow
(m.i integer checks, isinstance checks on Python lists) — all resolve to
concrete values at trace time, so standard torch.onnx.export works.

Key: export at DEPLOYMENT resolution (640), not training resolution (1280).
     All downstream latency numbers must reference this size.

Outputs:
  bench_results/cmx_yolo_fp32.onnx   — opset 12, fixed shape 1×4×640×640
  bench_results/cmx_yolo_fp32_sim.onnx — simplified (onnxsim)

Usage:
    cd /home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics
    python scripts/bench_step2_export_onnx.py
"""

import sys, os, json
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ultralytics import YOLO

MODEL_PT  = os.path.join(ROOT, 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt')
OUT_DIR   = os.path.join(ROOT, 'bench_results')
IMGSZ     = 640
OPSET     = 12   # SNPE ≥2.x, TFLite onnx-tf, TRT ≥8 all support opset 12
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ONNX     = os.path.join(OUT_DIR, 'cmx_yolo_fp32.onnx')
OUT_ONNX_SIM = os.path.join(OUT_DIR, 'cmx_yolo_fp32_sim.onnx')

# ── Load model ────────────────────────────────────────────────────────────────
print(f"[INFO] Loading {MODEL_PT}")
yolo = YOLO(MODEL_PT)
net  = yolo.model.cpu().eval()

# ── Trace verification ────────────────────────────────────────────────────────
dummy = torch.zeros(1, 4, IMGSZ, IMGSZ)
print("[INFO] Trace verification pass...")
with torch.no_grad():
    out = net(dummy)
print(f"  Output shape: {out[0].shape if isinstance(out, (list,tuple)) else out.shape}")

# ── Export ─────────────────────────────────────────────────────────────────────
print(f"[INFO] Exporting ONNX opset={OPSET} → {OUT_ONNX}")

# Wrap to ensure single-tensor input and single-tensor output for ONNX graph
class ExportWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x: [1, 4, H, W]  →  model returns list of [boxes, scores, ...]
        # Flatten to single tensor output for ONNX (Detect head returns tuple)
        out = self.model(x)
        # ultralytics Detect returns (inference_output, [p3, p4, p5])
        # We want only the inference output: shape [B, num_preds, 4+nc]
        if isinstance(out, (list, tuple)):
            return out[0]
        return out

wrapper = ExportWrapper(net)
wrapper.eval()

with torch.no_grad():
    torch.onnx.export(
        wrapper,
        dummy,
        OUT_ONNX,
        opset_version=OPSET,
        input_names=['images'],
        output_names=['output0'],
        dynamic_axes=None,   # fixed shape — required for SNPE/TFLite static allocation
        verbose=False,
        do_constant_folding=True,
    )
size_onnx = os.path.getsize(OUT_ONNX) / 1e6
print(f"  Saved: {OUT_ONNX}  ({size_onnx:.1f} MB)")

# ── ONNX validation ───────────────────────────────────────────────────────────
try:
    import onnx
    model_onnx = onnx.load(OUT_ONNX)
    onnx.checker.check_model(model_onnx)
    print("[INFO] ONNX graph check: PASS")
except ImportError:
    print("[WARN] onnx not installed — skipping graph check")

# ── onnxsim simplification ────────────────────────────────────────────────────
try:
    from onnxsim import simplify
    import onnx
    model_onnx = onnx.load(OUT_ONNX)
    model_sim, ok = simplify(model_onnx)
    if ok:
        onnx.save(model_sim, OUT_ONNX_SIM)
        size_sim = os.path.getsize(OUT_ONNX_SIM) / 1e6
        print(f"[INFO] Simplified → {OUT_ONNX_SIM}  ({size_sim:.1f} MB)")
    else:
        print("[WARN] onnxsim: simplification check failed — use unsimplified")
        OUT_ONNX_SIM = OUT_ONNX
except ImportError:
    print("[WARN] onnxsim not installed: pip install onnxsim")
    OUT_ONNX_SIM = OUT_ONNX

# ── Runtime verification via onnxruntime ──────────────────────────────────────
try:
    import onnxruntime as ort
    import numpy as np
    sess = ort.InferenceSession(OUT_ONNX_SIM, providers=['CPUExecutionProvider'])
    inp = {sess.get_inputs()[0].name: np.zeros((1, 4, IMGSZ, IMGSZ), dtype=np.float32)}
    ort_out = sess.run(None, inp)
    print(f"[INFO] ORT verification: output shape {ort_out[0].shape}  ✓")
except ImportError:
    print("[WARN] onnxruntime not installed — skipping ORT check")

# ── Save manifest ─────────────────────────────────────────────────────────────
manifest = {
    "onnx_fp32": OUT_ONNX_SIM,
    "imgsz": IMGSZ,
    "opset": OPSET,
    "input_name": "images",
    "output_name": "output0",
    "size_mb": round(os.path.getsize(OUT_ONNX_SIM) / 1e6, 1),
}
manifest_path = os.path.join(OUT_DIR, 'export_manifest.json')
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)
print(f"\n[DONE] Manifest → {manifest_path}")
print("Next: python scripts/bench_step3_convert_tflite.py")
