#!/usr/bin/env python3
"""
INT8 TFLite re-export with corrected calibration range.

ROOT CAUSE OF v1 FAILURE:
  onnx2tf normalizes calibration images as: (img - mean) / std
  Previous export used std=255 for images already in [0,1] range:
    (img_01 - 0) / 255 = img_01/255 ≈ [0, 0.004]  ← WRONG
  All quantization scales were calibrated for near-zero activations.
  At inference with [0,1] inputs, actual activations were 255x the calibrated range → crushed to 0.

FIX:
  Use std=1 so: (img_01 - 0) / 1 = img_01 → [0,1]  ← CORRECT
"""
import sys, shutil, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import onnx2tf

ONNX = Path('Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.onnx')
CALIB_NPY = Path('bench_results/calib/calib_stacked_nhwc.npy')
OUTPUT_DIR = Path('bench_results/ul_int8v2_saved_model')
BENCH_DIR = Path('bench_results')

assert ONNX.exists(), f"ONNX not found: {ONNX}"
assert CALIB_NPY.exists(), f"Calibration npy not found: {CALIB_NPY}"

calib = np.load(str(CALIB_NPY))
print(f"Calibration: shape={calib.shape}, range=[{calib.min():.4f},{calib.max():.4f}]")
assert calib.max() <= 1.1, "Calibration images should be in [0,1] range"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tmp_npy = str(OUTPUT_DIR / 'tmp_calib.npy')
shutil.copy(str(CALIB_NPY), tmp_npy)
print(f"Calibration npy → {tmp_npy}")

# Corrected np_data: std=1 since calibration images are already in [0,1]
np_data = [
    ["images",
     tmp_npy,
     [[[[0, 0, 0, 0]]]],    # mean = 0 (no offset)
     [[[[1, 1, 1, 1]]]]]    # std = 1 (FIXED: was 255, causing img/255 ≈ 0.004 scale error)
]

print(f"\nConverting {ONNX} → INT8 TFLite (corrected calibration std=1)...")
print(f"Output dir: {OUTPUT_DIR}")
t0 = time.time()

try:
    keras_model = onnx2tf.convert(
        input_onnx_file_path=str(ONNX),
        output_folder_path=str(OUTPUT_DIR),
        not_use_onnxsim=True,
        verbosity="error",
        output_integer_quantized_tflite=True,
        custom_input_op_name_np_data_path=np_data,
        enable_batchmatmul_unfold=False,  # Disable for INT8 (avoid large intermediate tensors)
        output_signaturedefs=True,
        disable_group_convolution=False,
    )
    elapsed = time.time() - t0
    print(f"Conversion done in {elapsed:.1f}s")
except Exception as e:
    print(f"Conversion FAILED: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# Rename output TFLite files to v2 variants
import glob
for src in glob.glob(str(OUTPUT_DIR / '*.tflite')):
    dst = BENCH_DIR / ('ul_best_int8v2_' + Path(src).name.replace(ONNX.stem + '_', ''))
    import shutil as sh; sh.copy(src, str(dst))
    print(f"Copied: {dst.name} ({dst.stat().st_size/1e6:.1f} MB)")

# Quick sanity probe
import tensorflow as tf
for name in sorted((BENCH_DIR).glob('ul_best_int8v2_*.tflite')):
    print(f"\n=== PROBE: {name.name} ===")
    try:
        interp = tf.lite.Interpreter(str(name))
        interp.allocate_tensors()
        ins = interp.get_input_details()[0]
        outs = interp.get_output_details()[0]
        in_dtype = np.dtype(ins['dtype'])
        print(f"  In: shape={ins['shape'].tolist()}, dtype={in_dtype}")
        print(f"  Out: shape={outs['shape'].tolist()}, dtype={outs['dtype']}")

        import cv2
        from pathlib import Path as P
        test_p = next(P('/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/test').glob('*.jpg'))
        test_ndvi = P('/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi/images/test') / (test_p.stem + '.png')
        bgr = cv2.imread(str(test_p))
        nir = cv2.imread(str(test_ndvi), cv2.IMREAD_GRAYSCALE)
        img4 = np.concatenate([bgr, nir[:,:,None]], axis=2)
        img4 = cv2.resize(img4, (640,640)).astype(np.float32)/255.0
        nhwc = img4[None].astype(in_dtype)

        interp.set_tensor(ins['index'], nhwc)
        interp.invoke()
        raw = interp.get_tensor(outs['index']).astype(np.float32)
        if outs['quantization_parameters'].get('scales', []):
            raw = (raw - outs['quantization_parameters']['zero_points'][0]) * outs['quantization_parameters']['scales'][0]
        print(f"  Output range: [{raw.min():.6f}, {raw.max():.6f}]")
        print(f"  Box (rows 0-3): [{raw[0,:4,:].min():.6f}, {raw[0,:4,:].max():.6f}]")
        print(f"  Scores (rows 4+): [{raw[0,4:,:].min():.6f}, {raw[0,4:,:].max():.6f}]")
    except Exception as e:
        print(f"  PROBE FAIL: {e}")
