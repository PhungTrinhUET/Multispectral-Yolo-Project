"""
Step 6 — Jetson Orin Nano benchmark harness (TensorRT, run ON-DEVICE).

Valid pairings (same constraint as VOXL2):
  fp32 / CPU (TRT FP32)
  fp16 / GPU (TRT FP16, native Ampere FP16)
  int8 / GPU (TRT INT8 + calibration)

POWER NOTE (CRITICAL for paper comparability):
  tegrastats on Jetson Orin Nano reports VDD_CPU_GPU_CV (CPU+GPU+CV combined rail)
  and VDD_SOC. Some Orin Nano carrier boards LACK the INA3221 sensor — tegrastats
  will then show 0mW or be absent. This is NOT total board power and is NOT
  comparable to VOXL2's px4-listener battery_status (whole-board, different rail).

  PRIMARY power number = external DC meter on input rail (12V or 5V depending on carrier).
  SECONDARY = tegrastats (label the rail name, NOT just "power").

Transfer:
  scp -r bench_results/ <jetson_ip>:/data/gmd_bench/
  scp scripts/bench_step6_jetson_trt.py <jetson_ip>:/data/gmd_bench/

On Jetson (JetPack 5.x / 6.x, jtop or tegrastats available):
  python3 bench_step6_jetson_trt.py --build_only   # build engines once (~5 min each)
  tegrastats --interval 500 --logfile tgs_fp16.log &
  python3 bench_step6_jetson_trt.py --variant fp16
  kill %1
  python3 bench_step6_jetson_trt.py --parse_power tgs_fp16.log

Outputs: /data/gmd_bench/jetson_{variant}.json
"""

import argparse, json, os, time, resource, glob, sys, subprocess, re
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--variant',       choices=['fp32', 'fp16', 'int8'], default='fp16')
parser.add_argument('--bench_dir',     default='/data/gmd_bench')
parser.add_argument('--n_warmup',      type=int, default=50)
parser.add_argument('--n_bench',       type=int, default=5000)
parser.add_argument('--n_sustained_s', type=int, default=300)
parser.add_argument('--build_only',    action='store_true')
parser.add_argument('--parse_power',   type=str, default=None)
args = parser.parse_args()

IMGSZ     = 640
BENCH_DIR = args.bench_dir
ONNX_PATH = os.path.join(BENCH_DIR, 'cmx_yolo_fp32_sim.onnx')
CALIB_NPY = os.path.join(BENCH_DIR, 'calib', 'calib_npy')
ENGINE_MAP = {
    'fp32': os.path.join(BENCH_DIR, 'cmx_trt_fp32.engine'),
    'fp16': os.path.join(BENCH_DIR, 'cmx_trt_fp16.engine'),
    'int8': os.path.join(BENCH_DIR, 'cmx_trt_int8.engine'),
}
OUT_PATH = os.path.join(BENCH_DIR, f'jetson_{args.variant}.json')

# ─────────────────────────────────────────────────────────────────────────────
# VALID PAIRINGS (TRT maps all three to GPU-class hardware)
# ─────────────────────────────────────────────────────────────────────────────
# All TRT variants run on Jetson GPU (Ampere). The "CPU" label is not applicable
# to TRT. For paper Table 2, Jetson column shows TRT precision, not delegate name.
VALID_VARIANTS = {'fp32', 'fp16', 'int8'}
if args.variant not in VALID_VARIANTS and not args.build_only and not args.parse_power:
    print(f"[ERROR] Invalid variant: {args.variant}")
    sys.exit(2)

# ─────────────────────────────────────────────────────────────────────────────
# tegrastats parser — Orin Nano specific (NOT Xavier NX VDD_IN parser)
# ─────────────────────────────────────────────────────────────────────────────
def parse_tegrastats_orin(log_path):
    """
    Parse tegrastats log for Jetson Orin Nano (JetPack 5/6).

    Orin Nano tegrastats format (JetPack 5):
      ... VDD_CPU_GPU_CV 1234mW/1234mW VDD_SOC 567mW/567mW ...

    VDD_IN (total board power as seen on Xavier NX) does NOT appear on Orin Nano
    dev kit — that field requires an INA3221 on the carrier board which is absent
    on the Orin Nano dev kit.

    Returns dict with per-rail mean/max (W) and availability note.
    """
    RAILS = {
        'VDD_CPU_GPU_CV': [],
        'VDD_SOC': [],
        'VDD_IN': [],           # may be present on third-party carriers
    }
    n_lines = 0
    try:
        with open(log_path) as f:
            for line in f:
                n_lines += 1
                for rail in RAILS:
                    # Pattern: "RAIL_NAME 1234mW/1234mW" or "RAIL_NAME 1234mW"
                    m = re.search(rf'{rail}\s+(\d+)mW', line)
                    if m:
                        RAILS[rail].append(int(m.group(1)) / 1000.0)  # mW → W
    except Exception as e:
        return {"error": str(e)}

    result = {"n_log_lines": n_lines}
    for rail, vals in RAILS.items():
        if vals:
            result[rail] = {
                "mean_w": round(float(np.mean(vals)), 2),
                "max_w":  round(float(np.max(vals)),  2),
                "n_samples": len(vals),
            }
        else:
            result[rail] = {"available": False}

    # Presence check and scope label
    has_vdd_in = bool(RAILS['VDD_IN'])
    result["primary_rail"] = "VDD_IN" if has_vdd_in else "VDD_CPU_GPU_CV"
    result["scope_note"] = (
        "VDD_IN = total board power (present on some carrier boards). "
        "VDD_CPU_GPU_CV = CPU+GPU+CV combined rail (NOT total board). "
        "VDD_SOC = SoC subsystem rail (NOT total board). "
        "If VDD_IN is absent (most Orin Nano dev kits): use external DC meter "
        "for total board power — that is the ONLY comparable number to VOXL2."
    )
    result["comparability_warning"] = (
        "VDD_CPU_GPU_CV is NOT comparable to VOXL2 px4-listener battery_status "
        "(different rails, different scope). Label each column separately in the paper table."
    )
    return result

if args.parse_power:
    print(f"[POWER] Parsing tegrastats (Orin Nano): {args.parse_power}")
    power_parsed = parse_tegrastats_orin(args.parse_power)
    print(json.dumps(power_parsed, indent=2))
    out_power = os.path.join(BENCH_DIR, f'jetson_power_{args.variant}.json')
    with open(out_power, 'w') as f:
        json.dump(power_parsed, f, indent=2)
    print(f"[POWER] Saved → {out_power}")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# PREREQUISITE PROBE
# ─────────────────────────────────────────────────────────────────────────────
print("[PREREQ PROBE] Checking Jetson prerequisites ...")

def probe_prerequisites_jetson(variant, bench_dir):
    ok = True

    # TensorRT
    try:
        import tensorrt as trt
        print(f"  [✓] TensorRT {trt.__version__}")
    except ImportError:
        print("  [✗] CRITICAL: tensorrt not found — JetPack must be installed")
        ok = False

    # PyCUDA
    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
        dev = cuda.Device(0)
        print(f"  [✓] PyCUDA — GPU: {dev.name()}")
    except ImportError:
        print("  [✗] CRITICAL: pycuda not found: pip3 install pycuda")
        ok = False
    except Exception as e:
        print(f"  [✗] CRITICAL: PyCUDA init error: {e}")
        ok = False

    # ONNX file
    onnx_ok = os.path.exists(ONNX_PATH)
    if onnx_ok:
        print(f"  [✓] ONNX: {ONNX_PATH} ({os.path.getsize(ONNX_PATH)/1e6:.1f} MB)")
    else:
        print(f"  [✗] CRITICAL: ONNX not found: {ONNX_PATH}")
        ok = False

    # Engine file (may not exist yet — build_only creates it)
    engine_path = ENGINE_MAP.get(variant)
    if engine_path and os.path.exists(engine_path):
        print(f"  [✓] Engine: {engine_path} ({os.path.getsize(engine_path)/1e6:.1f} MB)")
    else:
        print(f"  [!] Engine not built yet: {engine_path} — run with --build_only first")

    # Calib files (for INT8)
    if variant == 'int8':
        n_npy = len(glob.glob(os.path.join(CALIB_NPY, '*.npy')))
        if n_npy > 0:
            print(f"  [✓] Calib: {n_npy} .npy files")
        else:
            print(f"  [✗] CRITICAL: No calib .npy files for INT8 quantization")
            ok = False
        # Calib identity check
        ci = os.path.join(bench_dir, 'calib_identity.json')
        if os.path.exists(ci):
            with open(ci) as f:
                cid = json.load(f)
            print(f"  [✓] Calib identity SHA256: {cid.get('calib_list_sha256','?')[:16]}...")
        else:
            print("  [!] WARN: calib_identity.json not found — cannot confirm same set as VOXL2")

    # tegrastats availability
    ret = subprocess.run(['which', 'tegrastats'], capture_output=True)
    if ret.returncode == 0:
        print(f"  [✓] tegrastats: {ret.stdout.strip().decode()}")
    else:
        print("  [!] WARN: tegrastats not found — power measurement via log will not work")
        print("       Use external DC meter as primary power measurement")

    # jtop (optional, richer stats)
    try:
        import jtop
        print("  [✓] jtop available (optional, for richer stats)")
    except ImportError:
        print("  [!] jtop not available (optional): pip3 install jetson-stats")

    print(f"\n[PREREQ PROBE] {'ALL PASS' if ok else 'FAILURES — fix before building engine'}")
    return ok

if not probe_prerequisites_jetson(args.variant, BENCH_DIR):
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# TensorRT engine build
# ─────────────────────────────────────────────────────────────────────────────
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_path, engine_path, variant, calib_npy_dir=None):
    if os.path.exists(engine_path):
        print(f"  [INFO] Engine already exists: {engine_path}")
        return True

    print(f"  [BUILD] {variant} engine from {onnx_path} ...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print(f"  [ERROR] ONNX parse failed:")
            for i in range(parser.num_errors):
                print(f"    {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if variant == 'fp16':
        if not builder.platform_has_fast_fp16:
            print("  [WARN] Platform does not advertise fast FP16 — engine may use FP32")
        config.set_flag(trt.BuilderFlag.FP16)
        print("  [BUILD] FP16 mode enabled")

    elif variant == 'int8':
        if not builder.platform_has_fast_int8:
            print("  [WARN] Platform does not advertise fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allow FP16 for ops that don't support INT8

        if calib_npy_dir and os.path.isdir(calib_npy_dir):
            class NpyCalibrator(trt.IInt8EntropyCalibrator2):
                def __init__(self, npy_dir, imgsz):
                    super().__init__()
                    self.files = sorted(glob.glob(os.path.join(npy_dir, '*.npy')))[:200]
                    self.idx   = 0
                    self.nbytes = 4 * imgsz * imgsz * 4  # 4ch float32
                    self.d_input = cuda.mem_alloc(self.nbytes)
                    self.cache_file = os.path.join(npy_dir, 'trt_calib.cache')

                def get_batch_size(self): return 1

                def get_batch(self, names):
                    if self.idx >= len(self.files):
                        return None
                    arr = np.load(self.files[self.idx])  # 1×H×W×4 NHWC
                    arr = arr.transpose(0, 3, 1, 2).astype(np.float32)  # → NCHW
                    cuda.memcpy_htod(self.d_input, np.ascontiguousarray(arr))
                    self.idx += 1
                    return [int(self.d_input)]

                def read_calibration_cache(self):
                    if os.path.exists(self.cache_file):
                        with open(self.cache_file, 'rb') as f:
                            return f.read()
                    return None

                def write_calibration_cache(self, cache):
                    with open(self.cache_file, 'wb') as f:
                        f.write(cache)

            config.int8_calibrator = NpyCalibrator(calib_npy_dir, IMGSZ)
            print(f"  [BUILD] INT8 calibration from {calib_npy_dir}")
        else:
            print("  [ERROR] INT8 requires calibration data — not found")
            return False

    print(f"  [BUILD] Building engine (this may take several minutes) ...")
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        print("  [ERROR] Engine build failed")
        return False

    with open(engine_path, 'wb') as f:
        f.write(engine_bytes)
    sz = os.path.getsize(engine_path) / 1e6
    print(f"  [BUILD] Saved: {engine_path}  ({sz:.1f} MB)")
    return True

if args.build_only:
    print("[BUILD] Building all engines ...")
    for v in ['fp32', 'fp16', 'int8']:
        ok = build_engine(ONNX_PATH, ENGINE_MAP[v], v, CALIB_NPY)
        print(f"  {v}: {'OK' if ok else 'FAILED'}")
    sys.exit(0)

# ─────────────────────────────────────────────────────────────────────────────
# Load engine + allocate buffers
# ─────────────────────────────────────────────────────────────────────────────
engine_path = ENGINE_MAP[args.variant]
if not os.path.exists(engine_path):
    print(f"[ERROR] Engine not found: {engine_path}")
    print(f"  Run: python3 bench_step6_jetson_trt.py --build_only")
    sys.exit(1)

runtime = trt.Runtime(TRT_LOGGER)
with open(engine_path, 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

# Get tensor names and shapes (TRT 8.x API)
n_tensors = engine.num_io_tensors
inp_name = engine.get_tensor_name(0)
out_name = engine.get_tensor_name(1)
inp_shape = tuple(context.get_tensor_shape(inp_name))
out_shape  = tuple(context.get_tensor_shape(out_name))
print(f"[BENCH] Engine loaded: {args.variant}")
print(f"[BENCH] Input  [{inp_name}]: {inp_shape}")
print(f"[BENCH] Output [{out_name}]: {out_shape}")

inp_host = np.zeros(inp_shape, dtype=np.float32)
out_host = np.zeros(out_shape, dtype=np.float32)
inp_dev  = cuda.mem_alloc(inp_host.nbytes)
out_dev  = cuda.mem_alloc(out_host.nbytes)
stream   = cuda.Stream()
context.set_tensor_address(inp_name, int(inp_dev))
context.set_tensor_address(out_name, int(out_dev))

# ─────────────────────────────────────────────────────────────────────────────
# Input loader
# ─────────────────────────────────────────────────────────────────────────────
npy_files = sorted(glob.glob(os.path.join(CALIB_NPY, '*.npy')))
print(f"[BENCH] Input images: {len(npy_files)} .npy files")

def get_input_nchw(i):
    if npy_files:
        arr = np.load(npy_files[i % len(npy_files)])  # 1×H×W×4 NHWC
        return arr.transpose(0, 3, 1, 2).astype(np.float32)  # → NCHW
    return np.random.rand(1, 4, IMGSZ, IMGSZ).astype(np.float32)

def infer(i):
    inp_host[:] = get_input_nchw(i)
    cuda.memcpy_htod_async(inp_dev, inp_host, stream)
    context.execute_async_v3(stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(out_host, out_dev, stream)
    stream.synchronize()

# ─────────────────────────────────────────────────────────────────────────────
# Temperature reader (Jetson sysfs)
# ─────────────────────────────────────────────────────────────────────────────
def read_jetson_temps():
    zones = {}
    for i in range(15):
        p = f'/sys/class/thermal/thermal_zone{i}/temp'
        type_p = f'/sys/class/thermal/thermal_zone{i}/type'
        try:
            temp_val = int(open(p).read().strip()) / 1000.0
            zone_type = open(type_p).read().strip() if os.path.exists(type_p) else f'zone{i}'
            zones[zone_type] = temp_val
        except:
            pass
    return zones

def read_max_temp():
    zones = read_jetson_temps()
    return max(zones.values()) if zones else None

# ─────────────────────────────────────────────────────────────────────────────
# Warmup
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[BENCH] Warmup ({args.n_warmup} iters — discarded) ...")
for i in range(args.n_warmup):
    infer(i)

# ─────────────────────────────────────────────────────────────────────────────
# Burst benchmark
# ─────────────────────────────────────────────────────────────────────────────
print(f"[BENCH] Burst ({args.n_bench} iters) ...")
latencies_ms = []
for i in range(args.n_bench):
    t0 = time.perf_counter()
    infer(i)
    latencies_ms.append((time.perf_counter() - t0) * 1000)
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{args.n_bench}  mean={np.mean(latencies_ms):.2f} ms")

lat = np.array(latencies_ms)
fps_burst = 1000.0 / lat.mean()

# ─────────────────────────────────────────────────────────────────────────────
# Sustained benchmark — temperature over time
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[BENCH] Sustained run ({args.n_sustained_s}s) — logging FPS + temperature ...")
t_start     = time.time()
count_sus   = 0
fps_samples = []
temp_trace  = []

while time.time() - t_start < args.n_sustained_s:
    t_seg = time.time()
    for _ in range(10):
        infer(count_sus); count_sus += 1
    seg_fps = 10.0 / (time.time() - t_seg)
    elapsed = time.time() - t_start
    temps   = read_jetson_temps()
    temp_max = max(temps.values()) if temps else None
    fps_samples.append(seg_fps)
    temp_trace.append({
        "elapsed_s": round(elapsed, 1),
        "fps": round(seg_fps, 1),
        "temp_max_c": temp_max,
        "temps": temps,
    })
    if len(fps_samples) % 30 == 0:
        print(f"  t={elapsed:.0f}s  FPS={seg_fps:.1f}  temp={temp_max}°C")

fps_sus_mean = float(np.mean(fps_samples))
fps_sus_min  = float(np.min(fps_samples))
fps_sus_std  = float(np.std(fps_samples))
fps_initial  = float(np.mean(fps_samples[:10])) if len(fps_samples) >= 10 else fps_sus_mean
fps_final    = float(np.mean(fps_samples[-10:])) if len(fps_samples) >= 10 else fps_sus_mean
throttle_pct = (fps_initial - fps_final) / fps_initial * 100 if fps_initial > 0 else 0

print(f"[BENCH] Sustained FPS={fps_sus_mean:.1f}  min={fps_sus_min:.1f}  "
      f"throttle≈{throttle_pct:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# POWER SCHEMA — correct for Orin Nano
# ─────────────────────────────────────────────────────────────────────────────
power_block = {
    "primary": {
        "method": "external_dc_meter",
        "source_rail": "Jetson Orin Nano input rail (12V or 5V per carrier)",
        "scope": "whole-board",
        "how_to_measure": (
            "Connect DC meter to carrier board power input. "
            "Record idle_w (inference NOT running) then inference_w "
            "(during sustained run). "
            "energy_per_inf_mJ = (inference_w / fps_sustained) * 1000."
        ),
        "idle_w":    None,   # FILL MANUALLY
        "inference_w": None, # FILL MANUALLY
        "energy_per_inf_mJ": None,
        "status": "PENDING_EXTERNAL_MEASUREMENT",
    },
    "secondary_tegrastats": {
        "method": "tegrastats",
        "log_file": f"parse with: python3 bench_step6_jetson_trt.py --parse_power <log>",
        "rails_available": {
            "VDD_CPU_GPU_CV": "CPU+GPU+CV combined rail — NOT total board",
            "VDD_SOC": "SoC subsystem — NOT total board",
            "VDD_IN": "Total board (only present if INA3221 on carrier — ABSENT on Orin Nano dev kit)",
        },
        "comparability_warning": (
            "VDD_CPU_GPU_CV ≠ VOXL2 battery_status. "
            "Different devices, different rails. "
            "Do NOT place in same comparison column without labeling source."
        ),
        "data": None,  # filled by --parse_power after run
    }
}

mem_mb   = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
temp_all = read_jetson_temps()
temp_max = max(temp_all.values()) if temp_all else None

# ─────────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────────
results = {
    "device": "Jetson_Orin_Nano",
    "runtime": "TensorRT",
    "variant": args.variant,
    "engine_path": engine_path,
    "model_size_mb": round(os.path.getsize(engine_path)/1e6, 1),
    "imgsz": IMGSZ,
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
        "peak_ram_mb": round(mem_mb, 1),
        "temp_max_c":  temp_max,
        "temp_zones":  temp_all,
    },
    "power": power_block,
}

with open(OUT_PATH, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"[DONE] {OUT_PATH}")
print(f"  Variant        : {args.variant} (TensorRT)")
print(f"  Burst FPS      : {fps_burst:.1f}")
print(f"  Sustained FPS  : {fps_sus_mean:.1f}  (min {fps_sus_min:.1f})")
print(f"  Latency p95    : {float(np.percentile(lat,95)):.2f} ms")
print(f"  RAM            : {mem_mb:.1f} MB")
print(f"  Thermal throttle: {throttle_pct:.1f}%")
print(f"{'='*60}")
print(f"\nPower (secondary): run tegrastats in parallel, then:")
print(f"  python3 bench_step6_jetson_trt.py --parse_power <tegrastats_log>")
print(f"Power (primary):   use external DC meter — see 'power.primary' in output JSON")
