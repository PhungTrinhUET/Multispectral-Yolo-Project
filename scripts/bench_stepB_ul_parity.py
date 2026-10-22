#!/usr/bin/env python3
"""
STEP B parity: ul_best_float16.tflite and ul_best_integer_quant.tflite vs PyTorch.
- Real 4ch input via dataset.py code path (BGR cv2 + NDVI grayscale PNG, filename-stem pairing)
- PyTorch reference: raw model forward (NCHW float [0,1]) → pixel-space decode → NMS
- TFLite: NHWC float [0,1] input → raw output → denormalize → NMS
- Report: FP16 CPU status, post-NMS IoU mean/min, score delta mean/max
"""
import sys, numpy as np, cv2, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import tensorflow as tf
import torch
import torchvision.ops as tvops

from ultralytics import YOLO
from ultralytics.utils.nms import non_max_suppression

WEIGHTS = 'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt'
TEST_RGB  = Path('/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/test')
TEST_NDVI = Path('/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi/images/test')
IMGSZ = 640
BENCH_DIR = Path('bench_results')
N = 30        # images
CONF = 0.01   # low threshold to catch all detections
IOU  = 0.45   # NMS iou

# ── 1. Collect N paired test images ──────────────────────────────────────────
samples = []
for p in sorted(TEST_RGB.glob('*.jpg')):
    ndvi_path = TEST_NDVI / (p.stem + '.png')
    if ndvi_path.exists():
        samples.append(p)
    if len(samples) >= N:
        break
print(f"Paired images found: {len(samples)}/{N}")
assert len(samples) >= N, "Not enough paired test images"

def make_input(rgb_path):
    """Returns (nhwc [1,H,W,4] float32 [0,1]), (nchw tensor [1,4,H,W] float32 [0,1])."""
    bgr = cv2.imread(str(rgb_path))           # H,W,3 BGR uint8
    nir = cv2.imread(str(TEST_NDVI / (rgb_path.stem + '.png')), cv2.IMREAD_GRAYSCALE)  # H,W uint8
    img4 = np.concatenate([bgr, nir[:, :, None]], axis=2)          # H,W,4 (B,G,R,NIR)
    img4 = cv2.resize(img4, (IMGSZ, IMGSZ)).astype(np.float32) / 255.0  # H,W,4 [0,1]
    nhwc = img4[None]                                               # 1,H,W,4
    nchw = torch.from_numpy(img4.transpose(2, 0, 1)).unsqueeze(0)  # 1,4,H,W (no channel flip: shape[0]==4)
    return nhwc, nchw

# ── 2. Probe both TFLite artifacts ───────────────────────────────────────────
print("\n=== PROBE ===")
probe_info = {}
for name in ['ul_best_float16.tflite', 'ul_best_integer_quant.tflite', 'ul_best_full_integer_quant.tflite']:
    path = BENCH_DIR / name
    if not path.exists():
        print(f"{name}: NOT FOUND"); continue
    print(f"\n{name} ({path.stat().st_size/1e6:.1f} MB):")
    try:
        interp = tf.lite.Interpreter(str(path))
        interp.allocate_tensors()
        ind  = interp.get_input_details()[0]
        outd = interp.get_output_details()[0]
        in_dtype  = np.dtype(ind['dtype'])
        out_dtype = np.dtype(outd['dtype'])
        print(f"  Input : shape={ind['shape'].tolist()}, dtype={in_dtype}")
        print(f"  Output: shape={outd['shape'].tolist()}, dtype={out_dtype}")
        # quantization params
        iq = ind.get('quantization_parameters', {})
        oq = outd.get('quantization_parameters', {})
        if len(iq.get('scales', [])):
            print(f"  Input  quant scale={iq['scales'][0]:.6f} zp={iq['zero_points'][0]}")
        if len(oq.get('scales', [])):
            print(f"  Output quant scale={oq['scales'][0]:.6f} zp={oq['zero_points'][0]}")
        probe_info[name] = {'in_dtype': str(in_dtype), 'out_shape': outd['shape'].tolist()}
    except Exception as e:
        print(f"  PROBE FAIL: {e}")
        probe_info[name] = {'error': str(e)}

# ── 3. PyTorch reference ──────────────────────────────────────────────────────
print("\n=== PyTorch reference ===")
yolo = YOLO(WEIGHTS)
pt_model = yolo.model.eval()

# Warm-up
_nhwc0, _nchw0 = make_input(samples[0])
with torch.no_grad():
    _ = pt_model(_nchw0)

pt_all = []   # list of (xyxy_px [K,4], scores [K], cls [K]) per image
t0 = time.time()
for p in samples:
    _, nchw = make_input(p)
    with torch.no_grad():
        raw = pt_model(nchw)   # [1, 8+, 8400] pixel-space coords (standard Detect head)
    # raw might be (preds, protos) for segment or just preds for detect
    if isinstance(raw, (list, tuple)):
        raw = raw[0]
    # Apply NMS: expects [batch, 4+nc, anchors] OR [batch, anchors, 4+nc]
    # ultralytics non_max_suppression expects [batch, anchors, 4+nc]
    pred_nms = non_max_suppression(raw, conf_thres=CONF, iou_thres=IOU, max_det=100)
    det = pred_nms[0]  # [K, 6]: xyxy, conf, cls
    if det is not None and len(det):
        xyxy  = det[:, :4].cpu().numpy()      # pixel space
        scores = det[:, 4].cpu().numpy()
        cls    = det[:, 5].cpu().numpy().astype(int)
    else:
        xyxy, scores, cls = np.zeros((0,4)), np.zeros(0), np.zeros(0, int)
    pt_all.append((xyxy, scores, cls))

pt_time = time.time() - t0
print(f"PyTorch: {N} images in {pt_time:.1f}s, total detections={sum(len(x[0]) for x in pt_all)}")

# ── 4. TFLite parity helper ───────────────────────────────────────────────────
def nms_tflite(raw_out, imgsz=640, conf=CONF, iou=IOU):
    """
    raw_out: numpy [1, 8, 8400] — tf_wrapper normalized output
      rows 0-3: cx,cy,w,h in [0,1] (XYWH, decoded by _tf_inference via decode_bboxes w/ use_dfl=True)
      rows 4-7: class sigmoid scores [0,1]
    Returns (xyxy_px [K,4], scores [K], cls [K])
    """
    out = raw_out[0]   # [8, 8400]
    boxes_norm  = out[:4].T   # [8400, 4] XYWH normalized
    class_probs = out[4:].T   # [8400, nc]
    scores      = class_probs.max(axis=1)        # [8400]
    cls_ids     = class_probs.argmax(axis=1)     # [8400]

    mask = scores >= conf
    if not mask.any():
        return np.zeros((0,4)), np.zeros(0), np.zeros(0, int)

    # XYWH normalized → XYXY pixel space
    bn = boxes_norm[mask]
    cx, cy, w, h = bn[:,0], bn[:,1], bn[:,2], bn[:,3]
    boxes_px = np.stack([
        (cx - w/2) * imgsz, (cy - h/2) * imgsz,
        (cx + w/2) * imgsz, (cy + h/2) * imgsz,
    ], axis=1)   # [K, 4] xyxy pixel space

    sc = scores[mask]
    cl = cls_ids[mask]

    # NMS via torchvision
    keep = tvops.batched_nms(
        torch.from_numpy(boxes_px.astype(np.float32)),
        torch.from_numpy(sc.astype(np.float32)),
        torch.from_numpy(cl.astype(np.int64)),
        iou_threshold=iou,
    ).numpy()
    return boxes_px[keep], sc[keep], cl[keep].astype(int)

def run_parity(name, interp, in_idx, out_idx, in_dtype, samples, pt_all, imgsz=640):
    """Run TFLite + compare with PyTorch post-NMS. Return parity dict."""
    iou_list, score_delta, matched_total, pt_total, tfl_total = [], [], 0, 0, 0
    errors = 0
    t0 = time.time()
    for i, p in enumerate(samples):
        nhwc, _ = make_input(p)
        inp = nhwc.astype(in_dtype)
        try:
            interp.set_tensor(in_idx, inp)
            interp.invoke()
            raw_out = interp.get_tensor(out_idx)  # [1, 8, 8400]
        except Exception as e:
            errors += 1
            print(f"  Image {i} invoke error: {e}")
            continue

        xyxy_tfl, sc_tfl, cl_tfl = nms_tflite(raw_out, imgsz=imgsz)

        xyxy_pt, sc_pt, cl_pt = pt_all[i]
        pt_total  += len(xyxy_pt)
        tfl_total += len(xyxy_tfl)

        # Match by IoU (greedy)
        if len(xyxy_pt) and len(xyxy_tfl):
            iou_mat = tvops.box_iou(
                torch.from_numpy(xyxy_pt.astype(np.float32)),
                torch.from_numpy(xyxy_tfl.astype(np.float32)),
            ).numpy()
            used_tfl = set()
            for j in range(len(xyxy_pt)):
                row = iou_mat[j].copy()
                row[list(used_tfl)] = 0
                best_k = row.argmax()
                if iou_mat[j, best_k] >= 0.5:
                    iou_list.append(iou_mat[j, best_k])
                    score_delta.append(abs(float(sc_pt[j]) - float(sc_tfl[best_k])))
                    matched_total += 1
                    used_tfl.add(best_k)

    elapsed = time.time() - t0
    result = {
        'artifact': name,
        'n_images': N - errors,
        'errors': errors,
        'elapsed_s': round(elapsed, 1),
        'pt_detections': int(pt_total),
        'tfl_detections': int(tfl_total),
        'matched_pairs': int(matched_total),
        'iou_mean': round(float(np.mean(iou_list)), 4)  if iou_list else None,
        'iou_min':  round(float(np.min(iou_list)), 4)   if iou_list else None,
        'score_delta_mean': round(float(np.mean(score_delta)), 4) if score_delta else None,
        'score_delta_max':  round(float(np.max(score_delta)), 4)  if score_delta else None,
    }
    return result

# ── 5. FP16 TFLite ───────────────────────────────────────────────────────────
print("\n=== FP16 TFLite (ul_best_float16.tflite) ===")
fp16_path = BENCH_DIR / 'ul_best_float16.tflite'
fp16_result = {'artifact': 'ul_best_float16.tflite', 'cpu_runnable': False}

try:
    interp_fp16 = tf.lite.Interpreter(str(fp16_path))
    interp_fp16.allocate_tensors()
    ind_fp16  = interp_fp16.get_input_details()[0]
    outd_fp16 = interp_fp16.get_output_details()[0]
    in_dtype_fp16 = np.dtype(ind_fp16['dtype'])
    print(f"Input dtype: {in_dtype_fp16}")

    # Probe: try one image
    nhwc_test, _ = make_input(samples[0])
    inp_test = nhwc_test.astype(in_dtype_fp16)
    interp_fp16.set_tensor(ind_fp16['index'], inp_test)
    interp_fp16.invoke()
    raw_test = interp_fp16.get_tensor(outd_fp16['index'])
    print(f"FP16 CPU run: OK. Output shape={raw_test.shape}, range=[{raw_test.min():.4f}, {raw_test.max():.4f}]")
    fp16_result['cpu_runnable'] = True
    fp16_result['probe_output_shape'] = list(raw_test.shape)
    fp16_result['probe_output_range'] = [round(float(raw_test.min()), 4), round(float(raw_test.max()), 4)]

    # Full parity
    fp16_parity = run_parity(
        'ul_best_float16.tflite', interp_fp16,
        ind_fp16['index'], outd_fp16['index'], in_dtype_fp16,
        samples, pt_all
    )
    fp16_result.update(fp16_parity)
    print(f"  PT={fp16_parity['pt_detections']} TFL={fp16_parity['tfl_detections']} matched={fp16_parity['matched_pairs']}")
    print(f"  IoU mean={fp16_parity['iou_mean']} min={fp16_parity['iou_min']}")
    print(f"  Score Δ mean={fp16_parity['score_delta_mean']} max={fp16_parity['score_delta_max']}")

except Exception as e:
    print(f"FP16 CPU run BLOCKED: {e}")
    fp16_result['cpu_blocked_reason'] = str(e)
    fp16_result['cpu_runnable'] = False

# ── 6. INT8 TFLite (float32 i/o = integer_quant) ─────────────────────────────
print("\n=== INT8 TFLite (ul_best_integer_quant.tflite) ===")
int8_path = BENCH_DIR / 'ul_best_integer_quant.tflite'
int8_result = {'artifact': 'ul_best_integer_quant.tflite'}

try:
    interp_int8 = tf.lite.Interpreter(str(int8_path))
    interp_int8.allocate_tensors()
    ind_int8  = interp_int8.get_input_details()[0]
    outd_int8 = interp_int8.get_output_details()[0]
    in_dtype_int8 = np.dtype(ind_int8['dtype'])
    print(f"Input dtype: {in_dtype_int8}, Output dtype: {np.dtype(outd_int8['dtype'])}")

    # Probe output range
    nhwc_test, _ = make_input(samples[0])
    inp_test = nhwc_test.astype(in_dtype_int8)
    interp_int8.set_tensor(ind_int8['index'], inp_test)
    interp_int8.invoke()
    raw_test = interp_int8.get_tensor(outd_int8['index'])
    print(f"INT8 CPU run: OK. Output shape={raw_test.shape}, range=[{raw_test.min():.4f}, {raw_test.max():.4f}]")

    # Sanity check: box coord range (rows 0-3 should be [0,1] if tf_wrapper is working)
    box_range = raw_test[0, :4, :]
    score_range = raw_test[0, 4:, :]
    print(f"  Box coord range: [{box_range.min():.4f}, {box_range.max():.4f}]")
    print(f"  Score range: [{score_range.min():.4f}, {score_range.max():.4f}]")

    int8_result['probe_box_range'] = [round(float(box_range.min()),4), round(float(box_range.max()),4)]
    int8_result['probe_score_range'] = [round(float(score_range.min()),4), round(float(score_range.max()),4)]

    # Full parity
    int8_parity = run_parity(
        'ul_best_integer_quant.tflite', interp_int8,
        ind_int8['index'], outd_int8['index'], in_dtype_int8,
        samples, pt_all
    )
    int8_result.update(int8_parity)
    print(f"  PT={int8_parity['pt_detections']} TFL={int8_parity['tfl_detections']} matched={int8_parity['matched_pairs']}")
    print(f"  IoU mean={int8_parity['iou_mean']} min={int8_parity['iou_min']}")
    print(f"  Score Δ mean={int8_parity['score_delta_mean']} max={int8_parity['score_delta_max']}")

except Exception as e:
    print(f"INT8 FAIL: {e}")
    int8_result['error'] = str(e)

# ── 7. Full-integer quant (int8 i/o) ─────────────────────────────────────────
print("\n=== Full INT8 TFLite (ul_best_full_integer_quant.tflite) ===")
full_int8_path = BENCH_DIR / 'ul_best_full_integer_quant.tflite'
full_int8_result = {'artifact': 'ul_best_full_integer_quant.tflite'}

try:
    interp_fi = tf.lite.Interpreter(str(full_int8_path))
    interp_fi.allocate_tensors()
    ind_fi  = interp_fi.get_input_details()[0]
    outd_fi = interp_fi.get_output_details()[0]
    in_dtype_fi = np.dtype(ind_fi['dtype'])
    out_dtype_fi = np.dtype(outd_fi['dtype'])
    print(f"Input dtype: {in_dtype_fi}, Output dtype: {out_dtype_fi}")

    in_scale  = ind_fi.get('quantization_parameters', {}).get('scales', [1.0])[0]
    in_zp     = ind_fi.get('quantization_parameters', {}).get('zero_points', [0])[0]
    out_scale = outd_fi.get('quantization_parameters', {}).get('scales', [1.0])[0]
    out_zp    = outd_fi.get('quantization_parameters', {}).get('zero_points', [0])[0]
    print(f"  In quant: scale={in_scale:.6f} zp={in_zp}")
    print(f"  Out quant: scale={out_scale:.6f} zp={out_zp}")

    # Probe with quantized input
    nhwc_test, _ = make_input(samples[0])
    if in_dtype_fi == np.int8:
        inp_test = np.clip(np.round(nhwc_test / in_scale + in_zp), -128, 127).astype(np.int8)
    else:
        inp_test = nhwc_test.astype(in_dtype_fi)

    interp_fi.set_tensor(ind_fi['index'], inp_test)
    interp_fi.invoke()
    raw_test_qi = interp_fi.get_tensor(outd_fi['index'])
    if out_dtype_fi == np.int8:
        raw_test = (raw_test_qi.astype(np.float32) - out_zp) * out_scale
    else:
        raw_test = raw_test_qi.astype(np.float32)

    print(f"Full INT8 run: OK. Output shape={raw_test.shape}, range=[{raw_test.min():.4f}, {raw_test.max():.4f}]")
    box_range = raw_test[0, :4, :]
    score_range = raw_test[0, 4:, :]
    print(f"  Box coord range: [{box_range.min():.4f}, {box_range.max():.4f}]")
    print(f"  Score range: [{score_range.min():.4f}, {score_range.max():.4f}]")
    full_int8_result['note'] = 'int8 i/o — device NPU path; CPU parity skipped (non-standard)'

except Exception as e:
    print(f"Full INT8 FAIL: {e}")
    full_int8_result['error'] = str(e)

# ── 8. Save results ───────────────────────────────────────────────────────────
results = {
    'fp16': fp16_result,
    'int8_float_io': int8_result,
    'int8_full': full_int8_result,
    'probe': probe_info,
}
out_path = BENCH_DIR / 'stepB_ul_parity.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {out_path}")

# ── 9. Print summary table ────────────────────────────────────────────────────
print("\n=== STEP B SUMMARY (ul_best_*.tflite) ===")
print(f"{'Artifact':<35} {'CPU?':<8} {'IoU mean':<10} {'IoU min':<10} {'Score Δ mean':<14} {'Score Δ max'}")
for tag, r in [('ul_best_float16.tflite', fp16_result), ('ul_best_integer_quant.tflite', int8_result)]:
    cpu = 'YES' if r.get('cpu_runnable', False) else 'BLOCKED'
    if not r.get('cpu_runnable', True) and 'cpu_blocked_reason' in r:
        cpu = 'BLOCKED'
    iou_m  = r.get('iou_mean', 'N/A')
    iou_mn = r.get('iou_min', 'N/A')
    sd_m   = r.get('score_delta_mean', 'N/A')
    sd_mx  = r.get('score_delta_max', 'N/A')
    print(f"{tag:<35} {cpu:<8} {str(iou_m):<10} {str(iou_mn):<10} {str(sd_m):<14} {sd_mx}")
