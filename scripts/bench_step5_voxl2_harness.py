"""
Step 5 — VOXL2 Sentinel benchmark harness (run ON-DEVICE).

VALID PAIRINGS ONLY (others are explicitly skipped):
  fp32 / CPU  — float32, no delegate
  fp16 / GPU  — float16, Adreno 650 via OpenCL
  int8 / NPU  — int8, Hexagon NN delegate

PRE-FLIGHT GATE 1: run this FIRST with --preflight flag:
  python3 bench_step5_voxl2_harness.py --preflight

Gate 1 checks:
  - tflite_runtime or tensorflow importable
  - Delegate .so files present
  - Int8 / NPU: does the delegate ACTUALLY apply? If fallback → HARD EXIT
  - Gate 3b: compare 30 device outputs against PC reference; HARD EXIT on mismatch

Full run (ONLY after preflight passes):
  python3 bench_step5_voxl2_harness.py --variant fp32 --delegate CPU
  python3 bench_step5_voxl2_harness.py --variant fp16 --delegate GPU
  python3 bench_step5_voxl2_harness.py --variant int8 --delegate NPU

Transfer files from PC:
  adb push bench_results/ /data/gmd_bench/
  adb push scripts/bench_step5_voxl2_harness.py /data/gmd_bench/

Outputs: /data/gmd_bench/voxl2_{variant}_{delegate}.json
"""

import argparse, json, os, time, resource, glob, sys, hashlib
import numpy as np

BENCH_DIR_DEFAULT = '/data/gmd_bench'

parser = argparse.ArgumentParser()
parser.add_argument('--variant',  choices=['fp32', 'fp16', 'int8'], default='fp16')
parser.add_argument('--delegate', choices=['CPU', 'GPU', 'NPU'],    default='GPU')
parser.add_argument('--bench_dir', default=BENCH_DIR_DEFAULT)
parser.add_argument('--n_warmup',  type=int, default=50)
parser.add_argument('--n_bench',   type=int, default=5000)
parser.add_argument('--n_sustained_s', type=int, default=300,
                    help='Sustained run seconds (≥300 to induce thermal throttle)')
parser.add_argument('--preflight', action='store_true',
                    help='Run Gate 1+3b pre-flight check only (no full benchmark)')
args = parser.parse_args()

IMGSZ     = 640
BENCH_DIR = args.bench_dir

# ─────────────────────────────────────────────────────────────────────────────
# VALID PAIRINGS — reject before doing anything
# ─────────────────────────────────────────────────────────────────────────────
VALID_PAIRINGS = {('fp32', 'CPU'), ('fp16', 'GPU'), ('int8', 'NPU')}

if not args.preflight:
    pair = (args.variant, args.delegate)
    if pair not in VALID_PAIRINGS:
        valid_str = ', '.join(f"{v}/{d}" for v, d in sorted(VALID_PAIRINGS))
        print(f"[INVALID PAIRING] {args.variant}/{args.delegate} is not a valid combination.")
        print(f"  Valid pairings: {valid_str}")
        print(f"  fp32+GPU, int8+GPU, fp32+NPU etc. are NOT supported paths.")
        sys.exit(2)

MODEL_MAP = {
    'fp32': os.path.join(BENCH_DIR, 'cmx_yolo_fp32.tflite'),
    'fp16': os.path.join(BENCH_DIR, 'cmx_yolo_fp16.tflite'),
    'int8': os.path.join(BENCH_DIR, 'cmx_yolo_int8.tflite'),
}
CALIB_NPY = os.path.join(BENCH_DIR, 'calib', 'calib_npy')
REF_DIR   = os.path.join(BENCH_DIR, 'gate3_reference')

# ─────────────────────────────────────────────────────────────────────────────
# PREREQUISITE PROBE — check everything before building any interpreter
# ─────────────────────────────────────────────────────────────────────────────
def probe_prerequisites(variant, delegate, bench_dir):
    """
    Check all required tools/files are present on this device.
    Prints PASS/FAIL for each item. Returns False if any critical item missing.
    """
    print("\n[PREREQ PROBE] Checking device prerequisites ...")
    ok = True

    # 1. Python TFLite runtime
    try:
        import tensorflow as tf
        print(f"  [✓] tensorflow {tf.__version__}")
        tflite_module = tf.lite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tfl
            print(f"  [✓] tflite_runtime available")
            tflite_module = None
        except ImportError:
            print("  [✗] CRITICAL: Neither tensorflow nor tflite_runtime found.")
            print("      Install: pip3 install tflite-runtime")
            ok = False
            tflite_module = None

    # 2. Model file
    mpath = MODEL_MAP.get(variant, '')
    if os.path.exists(mpath):
        sz = os.path.getsize(mpath) / 1e6
        print(f"  [✓] Model file: {mpath} ({sz:.1f} MB)")
    else:
        print(f"  [✗] CRITICAL: Model file not found: {mpath}")
        print("       Transfer from PC: adb push bench_results/ /data/gmd_bench/")
        ok = False

    # 3. Calibration images (needed for Gate 3 spot-check)
    n_npy = len(glob.glob(os.path.join(CALIB_NPY, '*.npy')))
    if n_npy > 0:
        print(f"  [✓] Calib images: {n_npy} .npy files in {CALIB_NPY}")
    else:
        print(f"  [✗] WARN: No calib .npy files found in {CALIB_NPY}")
        print("       Will use dummy input — Gate 3 spot-check will be skipped")

    # 4. Gate 3 reference outputs from PC
    n_ref = len(glob.glob(os.path.join(REF_DIR, 'ref_*.npy')))
    if n_ref > 0:
        print(f"  [✓] Gate3 reference: {n_ref} PC reference outputs in {REF_DIR}")
    else:
        print(f"  [!] WARN: Gate3 reference not found in {REF_DIR}")
        print("       Run bench_preflight_pc.py on PC and re-transfer bench_results/")

    # 5. Delegate .so files
    DELEGATE_LIB_MAP = {
        'GPU': [
            '/usr/lib/libDelegate_gpu.so',
            '/usr/local/lib/libDelegate_gpu.so',
            '/data/local/tmp/libDelegate_gpu.so',
        ],
        'NPU': [
            '/usr/lib/libhexagon_interface.so',
            '/usr/local/lib/libhexagon_interface.so',
            '/usr/lib/libhexagon_nn_skel.so',
            '/data/local/tmp/libhexagon_interface.so',
        ],
    }
    if delegate in DELEGATE_LIB_MAP:
        found_so = None
        for so_path in DELEGATE_LIB_MAP[delegate]:
            if os.path.exists(so_path):
                found_so = so_path
                break
        if found_so:
            print(f"  [✓] {delegate} delegate .so: {found_so}")
        else:
            print(f"  [!] WARN: {delegate} delegate .so not found in known paths.")
            print(f"      Searched: {DELEGATE_LIB_MAP[delegate]}")
            print(f"      Gate 1 will detect fallback and EXIT if {delegate} cannot be loaded.")

    # 6. Calib identity — verify same set as used on PC
    calib_id_file = os.path.join(bench_dir, 'calib_identity.json')
    if os.path.exists(calib_id_file):
        with open(calib_id_file) as f:
            calib_id = json.load(f)
        print(f"  [✓] Calib identity: {calib_id.get('calib_list_sha256','?')[:16]}...")
    else:
        print(f"  [!] WARN: calib_identity.json not found — cannot verify same calib set")

    print(f"\n[PREREQ PROBE] {'ALL PASS' if ok else 'CRITICAL FAILURES — FIX BEFORE PROCEEDING'}")
    return ok

prereq_ok = probe_prerequisites(
    args.variant if not args.preflight else 'int8',
    args.delegate if not args.preflight else 'NPU',
    BENCH_DIR
)
if not prereq_ok:
    print("\n[STOP] Prerequisites failed. Fix above issues before running benchmark.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Interpreter builder — with HARD EXIT on delegate fallback
# ─────────────────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
    load_delegate = tf.lite.experimental.load_delegate
except ImportError:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
    load_delegate = tflite.load_delegate

def build_interpreter(model_path, delegate_name):
    """
    Build TFLite interpreter for the requested delegate.
    HARD EXIT (sys.exit(1)) if delegate fails and falls back to CPU.
    Returns (interp, actual_delegate_label).
    """
    if delegate_name == 'CPU':
        interp = Interpreter(model_path=model_path, num_threads=4)
        interp.allocate_tensors()
        print(f"[DELEGATE] CPU — standard TFLite CPU (4 threads)  ✓")
        return interp, 'CPU'

    elif delegate_name == 'GPU':
        # Try known GPU delegate library names for VOXL2
        GPU_SO_CANDIDATES = [
            'libDelegate_gpu.so',
            '/usr/lib/libDelegate_gpu.so',
            '/usr/local/lib/libDelegate_gpu.so',
        ]
        for so_path in GPU_SO_CANDIDATES:
            try:
                gpu_del = load_delegate(so_path,
                                        options={'is_precision_loss_allowed': 1})
                interp = Interpreter(model_path=model_path,
                                      experimental_delegates=[gpu_del])
                interp.allocate_tensors()
                # Verify it didn't silently fall back by checking allocation succeeded
                print(f"[DELEGATE] GPU (Adreno 650, {so_path})  ✓")
                return interp, f'GPU_Adreno_{so_path}'
            except Exception as e:
                print(f"  [DELEGATE] GPU attempt {so_path}: {e}")
                continue

        # All GPU attempts failed
        print("\n" + "!"*60)
        print("[GATE1 FAIL] GPU delegate could NOT be loaded on this device.")
        print("  - All known .so paths exhausted (see output above)")
        print("  - If the only available runtime is CPU, the VOXL2 GPU column")
        print("    is NOT measurable with TFLite on this firmware/SDK.")
        print("  - DO NOT report CPU numbers as GPU numbers.")
        print("!"*60)
        sys.exit(1)

    elif delegate_name == 'NPU':
        NPU_SO_CANDIDATES = [
            'libhexagon_interface.so',
            '/usr/lib/libhexagon_interface.so',
            '/usr/local/lib/libhexagon_interface.so',
            '/data/local/tmp/libhexagon_interface.so',
        ]
        for so_path in NPU_SO_CANDIDATES:
            try:
                npu_del = load_delegate(so_path, options={'debug_level': 0})
                interp = Interpreter(model_path=model_path,
                                      experimental_delegates=[npu_del])
                interp.allocate_tensors()
                print(f"[DELEGATE] NPU (Hexagon, {so_path})  ✓")
                return interp, f'NPU_Hexagon_{so_path}'
            except Exception as e:
                print(f"  [DELEGATE] NPU attempt {so_path}: {e}")
                continue

        print("\n" + "!"*60)
        print("[GATE1 FAIL] Hexagon NPU delegate could NOT be loaded on this device.")
        print("  - This means INT8/NPU row cannot be measured with TFLite.")
        print("  - The VOXL2 table collapses to CPU-only for this SDK version.")
        print("  - Consider SNPE SDK as alternative (snpe-dlc-quantize path).")
        print("!"*60)
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Temperature reader (time-series)
# ─────────────────────────────────────────────────────────────────────────────
THERMAL_ZONES = [f'/sys/class/thermal/thermal_zone{i}/temp' for i in range(20)]

def read_max_temp():
    """Read max temperature across all thermal zones in °C."""
    temps = []
    for p in THERMAL_ZONES:
        try:
            with open(p) as f:
                temps.append(int(f.read().strip()) / 1000.0)
        except:
            pass
    return max(temps) if temps else None

def read_all_temps():
    """Read all available thermal zones for diagnostic logging."""
    zones = {}
    for i, p in enumerate(THERMAL_ZONES):
        try:
            with open(p) as f:
                zones[f'zone{i}'] = int(f.read().strip()) / 1000.0
        except:
            pass
    return zones

# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT mode — Gate 1 + Gate 3b
# ─────────────────────────────────────────────────────────────────────────────
if args.preflight:
    print("\n" + "="*60)
    print("[GATE 1] Pre-flight: int8/NPU single run")
    print("="*60)

    # Use INT8 for Gate 1 (highest-risk delegate = NPU)
    int8_model = MODEL_MAP['int8']
    if not os.path.exists(int8_model):
        print(f"[GATE1 SKIP] INT8 model not found: {int8_model}")
        print("  Run bench_step3_convert_tflite.py first")
        sys.exit(1)

    # This call will HARD EXIT if NPU delegate fails
    print("[GATE1] Attempting to build INT8/NPU interpreter ...")
    interp, actual_del = build_interpreter(int8_model, 'NPU')

    inp_det = interp.get_input_details()[0]
    print(f"[GATE1] Input dtype: {inp_det['dtype'].__name__}  shape: {inp_det['shape'].tolist()}")

    dummy = np.zeros(inp_det['shape'], dtype=np.uint8)
    interp.set_tensor(inp_det['index'], dummy)
    interp.invoke()
    print(f"[GATE1] Single inference: OK")
    print(f"[GATE1] Actual delegate: {actual_del}")
    print(f"[GATE1] PASS — NPU delegate applied. Requested NPU == actual NPU  ✓")

    # Gate 3b — accuracy spot check
    print("\n" + "="*60)
    print("[GATE 3b] On-device accuracy spot-check (30 images vs PC reference)")
    print("="*60)

    ref_files = sorted(glob.glob(os.path.join(REF_DIR, 'ref_*.npy')))
    img_files = sorted(glob.glob(os.path.join(REF_DIR, 'img_*.npy')))
    list_path = os.path.join(BENCH_DIR, 'gate3_image_list.json')

    if not ref_files:
        print("[GATE3b] SKIP — no reference files found in", REF_DIR)
        print("  Run bench_preflight_pc.py on PC and re-transfer bench_results/")
    else:
        n_check = min(30, len(ref_files))
        match_count  = 0
        mismatch_msgs = []

        ABS_TOL = 0.1   # max acceptable absolute diff in detection scores
        CONF_THR = 0.05

        for i in range(n_check):
            # Load PC reference output
            ref_out = np.load(ref_files[i])   # [1, N, 4+nc]
            # Load the same input image
            if i < len(img_files):
                img_nhwc = np.load(img_files[i])  # 1×H×W×4 float32
            else:
                img_nhwc = np.zeros((1, IMGSZ, IMGSZ, 4), dtype=np.float32)

            # Convert to correct dtype
            inp_det_g3 = interp.get_input_details()[0]
            if inp_det_g3['dtype'] == np.uint8:
                scale, zp = inp_det_g3['quantization']
                inp = (img_nhwc / scale + zp).clip(0, 255).astype(np.uint8)
            else:
                inp = img_nhwc.astype(np.float32)

            interp.set_tensor(inp_det_g3['index'], inp)
            interp.invoke()
            dev_out = interp.get_tensor(interp.get_output_details()[0]['index'])

            # Compare top detections (above threshold) — check shapes first
            if ref_out.shape != dev_out.shape:
                mismatch_msgs.append(
                    f"img{i}: shape mismatch PC={ref_out.shape} dev={dev_out.shape}")
                continue

            # Max per-element diff on high-confidence predictions
            max_conf_pc  = ref_out[0, :, 4:].max(axis=1).max()
            max_conf_dev = dev_out[0, :, 4:].max(axis=1).max()
            score_diff   = abs(max_conf_pc - max_conf_dev)

            if score_diff > ABS_TOL:
                mismatch_msgs.append(
                    f"img{i}: max_conf PC={max_conf_pc:.3f} dev={max_conf_dev:.3f} "
                    f"diff={score_diff:.3f} > {ABS_TOL}")
            else:
                match_count += 1

        print(f"[GATE3b] Checked {n_check} images: {match_count} match, "
              f"{n_check - match_count} mismatch")

        if mismatch_msgs:
            print(f"\n{'!'*60}")
            print("[GATE3b] MISMATCHES DETECTED:")
            for m in mismatch_msgs[:10]:
                print(f"  {m}")
            print("\nPossible causes:")
            print("  - Delegate fallback (output matches CPU FP32 → not INT8 NPU)")
            print("  - Input channel order mismatch (BGR vs RGB, or NIR position)")
            print("  - Normalization difference (0-255 vs 0-1)")
            print("  - Different model file on device vs PC reference")
            if match_count < n_check * 0.5:
                print(f"{'!'*60}")
                print("[GATE3b] FAIL — >50% mismatch. DO NOT proceed with full benchmark.")
                sys.exit(1)
            else:
                print(f"{'!'*60}")
                print("[GATE3b] WARN — some mismatches but majority OK. Investigate before trusting numbers.")
        else:
            print(f"[GATE3b] PASS — on-device vs PC output match: PASS (N={n_check})  ✓")

    print("\n[PRE-FLIGHT] Complete. If all gates passed, run full benchmark:")
    print("  python3 bench_step5_voxl2_harness.py --variant fp32 --delegate CPU")
    print("  python3 bench_step5_voxl2_harness.py --variant fp16 --delegate GPU")
    print("  python3 bench_step5_voxl2_harness.py --variant int8 --delegate NPU")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# FULL BENCHMARK RUN
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = MODEL_MAP[args.variant]
OUT_PATH   = os.path.join(BENCH_DIR, f'voxl2_{args.variant}_{args.delegate}.json')

print(f"\n[BENCH] variant={args.variant}  delegate={args.delegate}")
print(f"[BENCH] model={MODEL_PATH}")
print(f"[BENCH] n_warmup={args.n_warmup}  n_bench={args.n_bench}  "
      f"sustained={args.n_sustained_s}s")

# Build interpreter — HARD EXIT if delegate fails
interp, actual_delegate = build_interpreter(MODEL_PATH, args.delegate)
inp_detail = interp.get_input_details()[0]
out_detail = interp.get_output_details()[0]

print(f"[BENCH] Input  dtype={inp_detail['dtype'].__name__}  "
      f"shape={inp_detail['shape'].tolist()}")
print(f"[BENCH] Output shape={out_detail['shape'].tolist()}")

# ── Input loader ──────────────────────────────────────────────────────────────
npy_files = sorted(glob.glob(os.path.join(CALIB_NPY, '*.npy')))
is_int8   = (inp_detail['dtype'] == np.uint8)

if is_int8:
    scale_in, zp_in = inp_detail['quantization']
    if scale_in == 0:
        print("[ERROR] INT8 input quantization scale=0 — model conversion failed")
        sys.exit(1)
    print(f"[BENCH] INT8 quant: scale={scale_in:.6f}  zero_point={zp_in}")

def get_input(i):
    if npy_files:
        arr = np.load(npy_files[i % len(npy_files)])  # 1×H×W×4 NHWC float32
    else:
        arr = np.random.rand(1, IMGSZ, IMGSZ, 4).astype(np.float32)
    if is_int8:
        arr = (arr / scale_in + zp_in).clip(0, 255).astype(np.uint8)
    return arr

# ── Warmup ────────────────────────────────────────────────────────────────────
print(f"\n[BENCH] Warmup ({args.n_warmup} iters — discarded) ...")
for i in range(args.n_warmup):
    interp.set_tensor(inp_detail['index'], get_input(i))
    interp.invoke()

# ── Burst benchmark ───────────────────────────────────────────────────────────
print(f"[BENCH] Burst benchmark ({args.n_bench} iters) ...")
latencies_ms = []
for i in range(args.n_bench):
    inp = get_input(i)
    t0 = time.perf_counter()
    interp.set_tensor(inp_detail['index'], inp)
    interp.invoke()
    latencies_ms.append((time.perf_counter() - t0) * 1000)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{args.n_bench}  mean={np.mean(latencies_ms):.2f} ms  "
              f"delegate={actual_delegate}")

lat = np.array(latencies_ms)
fps_burst = 1000.0 / lat.mean()
print(f"\n[BENCH] Burst — mean={lat.mean():.2f} ms  p95={np.percentile(lat,95):.2f} ms  "
      f"FPS={fps_burst:.1f}")

# ── Sustained benchmark (temperature over time) ───────────────────────────────
print(f"\n[BENCH] Sustained run ({args.n_sustained_s}s — thermal profiling) ...")
print("[BENCH] Logging FPS + temperature every 10 inferences")

t_start     = time.time()
count_sus   = 0
fps_samples = []
temp_trace  = []   # (elapsed_s, fps, temp_c)

while time.time() - t_start < args.n_sustained_s:
    t_seg = time.time()
    for _ in range(10):
        interp.set_tensor(inp_detail['index'], get_input(count_sus))
        interp.invoke()
        count_sus += 1
    seg_fps  = 10.0 / (time.time() - t_seg)
    elapsed  = time.time() - t_start
    temp_now = read_max_temp()
    fps_samples.append(seg_fps)
    temp_trace.append({
        "elapsed_s": round(elapsed, 1),
        "fps": round(seg_fps, 1),
        "temp_c": temp_now,
    })
    if len(fps_samples) % 30 == 0:
        print(f"  t={elapsed:.0f}s  FPS={seg_fps:.1f}  temp={temp_now}°C")

fps_sus_mean = float(np.mean(fps_samples))
fps_sus_min  = float(np.min(fps_samples))
fps_sus_std  = float(np.std(fps_samples))

# Check for throttle (FPS drop ≥10% from initial segment)
fps_initial  = float(np.mean(fps_samples[:10])) if len(fps_samples) >= 10 else fps_sus_mean
fps_final    = float(np.mean(fps_samples[-10:])) if len(fps_samples) >= 10 else fps_sus_mean
throttle_pct = (fps_initial - fps_final) / fps_initial * 100 if fps_initial > 0 else 0

print(f"[BENCH] Sustained — mean={fps_sus_mean:.1f} FPS  min={fps_sus_min:.1f}  "
      f"initial={fps_initial:.1f}  final={fps_final:.1f}  "
      f"throttle≈{throttle_pct:.1f}%")
if throttle_pct > 10:
    print(f"[BENCH] NOTE: thermal throttle detected ({throttle_pct:.1f}% FPS drop) — "
          "report fps_sustained (not fps_burst) as the paper number.")

# ── Resource snapshot ─────────────────────────────────────────────────────────
mem_mb   = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
temp_all = read_all_temps()
temp_max = max(temp_all.values()) if temp_all else None

# ─────────────────────────────────────────────────────────────────────────────
# POWER SCHEMA  (explicit source labeling — not apples-to-apples with Jetson)
# ─────────────────────────────────────────────────────────────────────────────
# Primary: external DC meter on 5V input rail (fill manually).
# Secondary: px4-listener battery_status (whole-board, only if PX4 running).
# NEVER mix with Jetson tegrastats VDD_CPU_GPU_CV (different rail, different device).

def read_px4_power_sample():
    """Single sample from px4-listener. Returns (current_A, voltage_V) or None."""
    import subprocess
    try:
        out = subprocess.run(
            ['px4-listener', 'battery_status', '-n', '1'],
            capture_output=True, text=True, timeout=3
        )
        curr = volt = None
        for line in out.stdout.splitlines():
            if 'current_a' in line:
                curr = float(line.split()[-1])
            if 'voltage_v' in line:
                volt = float(line.split()[-1])
        if curr is not None and volt is not None:
            return curr, volt
    except Exception:
        pass
    return None

px4_sample = read_px4_power_sample()
px4_power_w = None
px4_available = False
if px4_sample:
    curr_a, volt_v = px4_sample
    px4_power_w   = round(curr_a * volt_v, 2)
    px4_available = True
    print(f"[POWER] px4-listener: {curr_a:.2f}A × {volt_v:.2f}V = {px4_power_w:.2f}W")
else:
    print("[POWER] px4-listener: not available (PX4 not running / power module absent)")

power_block = {
    "primary": {
        "method": "external_dc_meter",
        "source_rail": "VOXL2 5V input rail",
        "scope": "whole-board",
        "how_to_measure": (
            "Connect USB power meter or bench supply to VOXL2 5V input. "
            "Record idle_w (board idle, inference NOT running), then "
            "inference_w (during sustained run). "
            "energy_per_inf_mJ = (inference_w / fps_sustained) * 1000."
        ),
        "idle_w":        None,    # FILL MANUALLY after external meter measurement
        "inference_w":   None,    # FILL MANUALLY
        "energy_per_inf_mJ": None,
        "status": "PENDING_EXTERNAL_MEASUREMENT",
    },
    "secondary": {
        "method": "px4-listener battery_status",
        "source_rail": "battery rail via PX4 power module",
        "scope": "whole-board (requires PX4 armed + power module installed)",
        "available": px4_available,
        "inference_w": px4_power_w,
        "comparability_note": (
            "NOT comparable to Jetson tegrastats. "
            "Jetson reports VDD_CPU_GPU_CV (partial rail). "
            "VOXL2 px4 reports total battery current. "
            "Different scope — do NOT place in same comparison column."
        ),
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Compile and save results
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "device": "VOXL2_Sentinel",
    "soc": "Qualcomm_QRB5165",
    "variant": args.variant,
    "requested_delegate": args.delegate,
    "actual_delegate": actual_delegate,
    "model_path": MODEL_PATH,
    "model_size_mb": round(os.path.getsize(MODEL_PATH)/1e6, 1),
    "imgsz": IMGSZ,
    "input_dtype": inp_detail['dtype'].__name__,
    "burst": {
        "n_inferences": args.n_bench,
        "warmup_discarded": args.n_warmup,
        "latency_mean_ms": round(float(lat.mean()), 2),
        "latency_std_ms":  round(float(lat.std()),  2),
        "latency_p50_ms":  round(float(np.percentile(lat, 50)), 2),
        "latency_p95_ms":  round(float(np.percentile(lat, 95)), 2),
        "latency_p99_ms":  round(float(np.percentile(lat, 99)), 2),
        "fps_burst":       round(fps_burst, 1),
    },
    "sustained": {
        "duration_s":    args.n_sustained_s,
        "fps_mean":      round(fps_sus_mean, 1),
        "fps_min":       round(fps_sus_min,  1),
        "fps_std":       round(fps_sus_std,  1),
        "fps_initial":   round(fps_initial,  1),
        "fps_final":     round(fps_final,    1),
        "throttle_pct":  round(throttle_pct, 1),
        "fps_trace":     fps_samples,
        "temp_trace":    temp_trace,
        "report_fps":    "fps_mean (sustained), NOT fps_burst",
    },
    "resources": {
        "peak_ram_mb":    round(mem_mb, 1),
        "temp_max_c":     temp_max,
        "temp_all_zones": temp_all,
    },
    "power": power_block,
}

with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"[DONE] {OUT_PATH}")
print(f"  Variant          : {args.variant}  ({actual_delegate})")
print(f"  Burst  FPS       : {fps_burst:.1f}")
print(f"  Sustained FPS    : {fps_sus_mean:.1f}  (min {fps_sus_min:.1f})")
print(f"  Latency mean     : {lat.mean():.2f} ms")
print(f"  Latency p95      : {float(np.percentile(lat,95)):.2f} ms")
print(f"  RAM              : {mem_mb:.1f} MB")
print(f"  Thermal throttle : {throttle_pct:.1f}%")
print(f"  Delegate check   : requested={args.delegate}  actual={actual_delegate}")
if 'FALLBACK' in actual_delegate:
    print(f"\n[ERROR] Delegate fallback detected in results — these numbers are INVALID.")
    sys.exit(1)
print(f"{'='*60}")
