#!/usr/bin/env python3
"""
STEP B parity: best_int8_ort_v3_perchannel.onnx vs PyTorch FP32 reference.
- Real 4ch input (BGR cv2 + NDVI grayscale PNG, filename-stem pairing)
- PyTorch reference: raw model forward (NCHW float [0,1]) → pixel-space XYWH → NMS
- ORT INT8: ONNX Runtime with INT8 weights → same output format → NMS
- Report: post-NMS IoU mean/min, score delta mean/max
"""
import sys, numpy as np, cv2, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import onnxruntime as ort
import torch
import torchvision.ops as tvops

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression as ul_nms

WEIGHTS   = 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt'
ORT_INT8  = Path('bench_results/best_int8_ort_v3_perchannel.onnx')
TEST_RGB  = Path('/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/test')
TEST_NDVI = Path('/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi/images/test')
IMGSZ = 640
N     = 30
CONF  = 0.01
IOU   = 0.45

# ── 1. Paired test images ──────────────────────────────────────────────────────
samples = []
for p in sorted(TEST_RGB.glob('*.jpg')):
    if (TEST_NDVI / (p.stem + '.png')).exists():
        samples.append(p)
    if len(samples) >= N:
        break
print(f"Paired images: {len(samples)}/{N}")
assert len(samples) >= N

def make_input(rgb_path):
    bgr = cv2.imread(str(rgb_path))
    nir = cv2.imread(str(TEST_NDVI / (rgb_path.stem + '.png')), cv2.IMREAD_GRAYSCALE)
    img4 = np.concatenate([bgr, nir[:, :, None]], axis=2)
    img4 = cv2.resize(img4, (IMGSZ, IMGSZ)).astype(np.float32) / 255.0
    nhwc = img4[None]                                              # [1,H,W,4]
    nchw = torch.from_numpy(img4.transpose(2, 0, 1)).unsqueeze(0) # [1,4,H,W]
    return nhwc, nchw

# ── 2. ORT INT8 probe ─────────────────────────────────────────────────────────
print(f"\nLoading ORT INT8: {ORT_INT8}")
assert ORT_INT8.exists(), f"Not found: {ORT_INT8}"
sess = ort.InferenceSession(str(ORT_INT8), providers=['CPUExecutionProvider'])
inp_name  = sess.get_inputs()[0].name
out_name  = sess.get_outputs()[0].name
inp_shape = sess.get_inputs()[0].shape
out_shape = sess.get_outputs()[0].shape
print(f"  Input  name={inp_name!r}, shape={inp_shape}")
print(f"  Output name={out_name!r}, shape={out_shape}")

# Probe on one image
nhwc_probe, _ = make_input(samples[0])
nchw_probe = nhwc_probe.transpose(0, 3, 1, 2)  # NHWC → NCHW for ORT
probe_out = sess.run([out_name], {inp_name: nchw_probe})[0]
print(f"  Probe output: shape={probe_out.shape}, range=[{probe_out.min():.4f},{probe_out.max():.4f}]")
print(f"  Box (rows 0-3): [{probe_out[0,:4,:].min():.4f},{probe_out[0,:4,:].max():.4f}]")
print(f"  Scores (rows 4+): [{probe_out[0,4:,:].min():.4f},{probe_out[0,4:,:].max():.4f}]")

# ── 3. PyTorch reference ───────────────────────────────────────────────────────
print("\nLoading PyTorch model...")
model = YOLO(WEIGHTS)
model.model.eval()

def run_pytorch(nchw):
    with torch.no_grad():
        raw = model.model(nchw)
    pred = raw[0] if isinstance(raw, (list, tuple)) else raw
    return pred  # [1, 8, N_anchors]

def nms_from_raw(raw_bcn, imgsz=IMGSZ, conf=CONF, iou=IOU):
    """NMS on PyTorch raw [B,C,N] output (XYWH pixel). Uses ul_nms which handles XYWH→XYXY."""
    preds = ul_nms(raw_bcn, conf_thres=conf, iou_thres=iou, max_det=300)
    det = preds[0]  # [K, 6+]: xyxy, conf, cls
    if det is None or len(det) == 0:
        return np.zeros((0,4)), np.zeros(0), np.zeros(0,int)
    return det[:,:4].cpu().numpy(), det[:,4].cpu().numpy(), det[:,5].cpu().numpy().astype(int)

def nms_ort(raw, imgsz=IMGSZ, conf=CONF, iou=IOU):
    """NMS on ORT INT8 output [1,8,N]. Output is XYWH normalized [0,1]."""
    out = raw[0]  # [8, N]
    # Convert XYWH normalized → XYXY pixel
    cx, cy, w, h = out[0], out[1], out[2], out[3]
    boxes_px = np.stack([
        (cx - w/2) * imgsz, (cy - h/2) * imgsz,
        (cx + w/2) * imgsz, (cy + h/2) * imgsz,
    ], axis=1)  # [N, 4] XYXY pixel
    scores = out[4:].T  # [N, nc]
    max_s  = scores.max(axis=1)
    cls_id = scores.argmax(axis=1)
    mask   = max_s >= conf
    if not mask.any():
        return np.zeros((0,4)), np.zeros(0), np.zeros(0,int)
    bx = boxes_px[mask]
    sc = max_s[mask]
    cl = cls_id[mask]
    keep = tvops.batched_nms(
        torch.from_numpy(bx.astype(np.float32)),
        torch.from_numpy(sc.astype(np.float32)),
        torch.from_numpy(cl),
        iou
    ).numpy()
    return bx[keep], sc[keep], cl[keep].astype(int)

# ── 4. Parity loop ─────────────────────────────────────────────────────────────
print(f"\nRunning parity on {N} images (CONF={CONF}, IOU={IOU})...")
iou_vals, score_deltas = [], []
n_both, n_only_pt, n_only_ort = 0, 0, 0

for i, path in enumerate(samples):
    nhwc, nchw = make_input(path)

    # PyTorch
    raw_pt = run_pytorch(nchw)
    bx_pt, sc_pt, cl_pt = nms_from_raw(raw_pt)

    # ORT INT8 — uses NCHW input
    nchw_ort = nhwc.transpose(0, 3, 1, 2)  # [1,H,W,4] → [1,4,H,W]
    raw_ort = sess.run([out_name], {inp_name: nchw_ort})[0]
    bx_ort, sc_ort, cl_ort = nms_ort(raw_ort)

    # Match detections
    if len(bx_pt) == 0 and len(bx_ort) == 0:
        continue
    n_both_img = 0
    if len(bx_pt) > 0 and len(bx_ort) > 0:
        # greedy match: for each PT box find best IoU ORT box
        from torchvision.ops import box_iou
        iou_mat = box_iou(
            torch.tensor(bx_pt, dtype=torch.float32),
            torch.tensor(bx_ort, dtype=torch.float32)
        )
        for pi in range(len(bx_pt)):
            best_iou, best_j = iou_mat[pi].max(0)
            if best_iou >= 0.5:
                iou_vals.append(float(best_iou))
                score_deltas.append(abs(float(sc_pt[pi]) - float(sc_ort[best_j])))
                n_both_img += 1
    n_both     += n_both_img
    n_only_pt  += max(0, len(bx_pt) - n_both_img)
    n_only_ort += max(0, len(bx_ort) - n_both_img)

    if i < 5:
        print(f"  [{i:2d}] PT={len(bx_pt)} ORT={len(bx_ort)} matched={n_both_img}")

# ── 5. Report ─────────────────────────────────────────────────────────────────
print(f"\n=== STEP B ORT INT8 PARITY REPORT ===")
print(f"N_images: {N}")
print(f"Matched detections (IoU≥0.5): {n_both}")
print(f"PT-only detections:            {n_only_pt}")
print(f"ORT-only detections:           {n_only_ort}")
if iou_vals:
    print(f"Post-NMS IoU  mean={np.mean(iou_vals):.6f}  min={np.min(iou_vals):.6f}")
    print(f"Score delta   mean={np.mean(score_deltas):.6f}  max={np.max(score_deltas):.6f}")
else:
    print("No matched detections found (possible model-level issue)")

result = {
    "artifact": str(ORT_INT8.name),
    "n_images": N,
    "matched": n_both,
    "pt_only": n_only_pt,
    "ort_only": n_only_ort,
    "iou_mean": round(float(np.mean(iou_vals)), 6) if iou_vals else None,
    "iou_min":  round(float(np.min(iou_vals)), 6)  if iou_vals else None,
    "score_delta_mean": round(float(np.mean(score_deltas)), 6) if score_deltas else None,
    "score_delta_max":  round(float(np.max(score_deltas)), 6) if score_deltas else None,
}
out_json = Path('bench_results/stepB_ort_int8_parity.json')
with open(out_json, 'w') as f:
    json.dump(result, f, indent=2)
print(f"\nSaved: {out_json}")
print(json.dumps(result, indent=2))
