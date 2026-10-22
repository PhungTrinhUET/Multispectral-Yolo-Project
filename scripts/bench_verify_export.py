"""
Export fidelity verification — CMX dual-stream model.

Checks three things (numbers only, no verdicts):

1. InputContainer split in ONNX:
   - Which Slice ops consume the graph input directly?
   - What axes/starts/ends do they use? (NCHW axis=1 expected)
   - Expected NHWC equivalents after onnx2tf (axis=3)

2. FusionRectify attention ops in ONNX:
   - Channel branch: ReduceMean + ReduceMax + Concat + Gemm + Sigmoid present?
   - Spatial branch: Concat + Conv(k=1) + Relu + Sigmoid + Slice present?
   - All three FusionRectify instances (P5=layers21, P4=layers22, P3=layers23)?
   - Any op that onnx2tf is known to decompose (GlobalAveragePool, etc.)

3. Numerical parity:
   - FP32 PyTorch vs FP32 ONNX-RT: max/mean abs diff on raw output [1,8,8400]
   - FP32 PyTorch vs FP16 ONNX-RT (cast): max/mean abs diff
   - Detection-level: top-k detections (conf>0.1), box IoU deltas, score deltas
   - TFLite FP16: if tensorflow installed, same comparison

Usage:
    cd /home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics
    python scripts/bench_verify_export.py
    python scripts/bench_verify_export.py --input real  # use a real test image
"""

import sys, os, argparse, json
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import onnx
import onnxruntime as ort

parser = argparse.ArgumentParser()
parser.add_argument('--input', choices=['synthetic', 'real'], default='synthetic',
                    help='synthetic: random+structured 4ch tensor; real: first test image')
parser.add_argument('--onnx', default=os.path.join(ROOT, 'bench_results/cmx_yolo_fp32_sim.onnx'))
parser.add_argument('--tflite_fp16', default=os.path.join(ROOT, 'bench_results/cmx_yolo_fp16.tflite'))
parser.add_argument('--model_pt', default=os.path.join(ROOT,
    'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt'))
parser.add_argument('--test_dir', default=
    '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/test/images')
args = parser.parse_args()

IMGSZ = 640
NC    = 4   # number of classes

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def sep(title=''):
    w = 60
    if title:
        pad = max(0, w - len(title) - 4)
        print(f"\n{'─'*2} {title} {'─'*pad}")
    else:
        print('─' * w)

def onnx_initializer_map(model_onnx):
    """Return dict: name → numpy array for all ONNX initializers."""
    init_map = {}
    for init in model_onnx.graph.initializer:
        arr = np.frombuffer(init.raw_data, dtype=np.float32) if init.raw_data else None
        if arr is None:
            try:
                arr = np.array(list(init.int64_data), dtype=np.int64)
            except:
                arr = None
        init_map[init.name] = arr
    return init_map

def get_slice_axes(node, init_map):
    """
    Extract (starts, ends, axes) from an ONNX Slice node (opset ≥9 dynamic format).
    Returns None for any input that is not a constant initializer.
    """
    inputs = node.input  # [data, starts, ends, axes, steps]
    def fetch(name):
        if not name:
            return None
        arr = init_map.get(name)
        return arr.tolist() if arr is not None else None

    starts = fetch(inputs[1]) if len(inputs) > 1 else None
    ends   = fetch(inputs[2]) if len(inputs) > 2 else None
    axes   = fetch(inputs[3]) if len(inputs) > 3 else None
    steps  = fetch(inputs[4]) if len(inputs) > 4 else None
    return starts, ends, axes, steps

def nchw_to_nhwc_axis(axes):
    """Map NCHW axis list to NHWC axis list (N=0,C=1,H=2,W=3 → N=0,H=1,W=2,C=3)."""
    if axes is None:
        return None
    mapping = {0: 0, 1: 3, 2: 1, 3: 2}
    return [mapping.get(a, a) for a in axes]

def iou_box(b1, b2):
    """b1, b2 = [cx, cy, w, h] normalized."""
    b1x1, b1y1 = b1[0]-b1[2]/2, b1[1]-b1[3]/2
    b1x2, b1y2 = b1[0]+b1[2]/2, b1[1]+b1[3]/2
    b2x1, b2y1 = b2[0]-b2[2]/2, b2[1]-b2[3]/2
    b2x2, b2y2 = b2[0]+b2[2]/2, b2[1]+b2[3]/2
    ix = max(0, min(b1x2,b2x2) - max(b1x1,b2x1))
    iy = max(0, min(b1y2,b2y2) - max(b1y1,b2y1))
    inter = ix * iy
    union = b1[2]*b1[3] + b2[2]*b2[3] - inter
    return inter / (union + 1e-7)

# ─────────────────────────────────────────────────────────────────────────────
# Step 0 — Export ONNX if needed
# ─────────────────────────────────────────────────────────────────────────────
sep("STEP 0 — ONNX export")

if not os.path.exists(args.onnx):
    print(f"[INFO] ONNX not found at {args.onnx}")
    print(f"[INFO] Exporting now ...")

    from ultralytics import YOLO

    class ExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            out = self.model(x)
            return out[0] if isinstance(out, (list, tuple)) else out

    yolo = YOLO(args.model_pt)
    net  = yolo.model.cpu().eval()
    wrapper = ExportWrapper(net)
    dummy   = torch.zeros(1, 4, IMGSZ, IMGSZ)

    os.makedirs(os.path.dirname(args.onnx), exist_ok=True)
    torch.onnx.export(
        wrapper, dummy, args.onnx,
        opset_version=12,
        input_names=['images'], output_names=['output0'],
        dynamic_axes=None,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"  Exported: {args.onnx}  ({os.path.getsize(args.onnx)/1e6:.1f} MB)")

    # Simplify
    try:
        from onnxsim import simplify
        m_onnx = onnx.load(args.onnx)
        m_sim, ok = simplify(m_onnx)
        if ok:
            onnx.save(m_sim, args.onnx)
            print(f"  Simplified  ({os.path.getsize(args.onnx)/1e6:.1f} MB)")
    except ImportError:
        pass
else:
    print(f"  Using existing: {args.onnx}  ({os.path.getsize(args.onnx)/1e6:.1f} MB)")

model_onnx = onnx.load(args.onnx)
onnx.checker.check_model(model_onnx)
print(f"  ONNX graph check: PASS")
print(f"  Nodes: {len(model_onnx.graph.node)}")
print(f"  Inputs: {[i.name for i in model_onnx.graph.input]}")
print(f"  Input shape: {[d.dim_value for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]}")

init_map    = onnx_initializer_map(model_onnx)
graph_input = {i.name for i in model_onnx.graph.input}

# Build tensor→producer map for graph traversal
tensor_to_producer = {}
for node in model_onnx.graph.node:
    for outp in node.output:
        tensor_to_producer[outp] = node

def upstream_op(tensor_name, depth=5):
    """Return chain of op_types from a tensor back depth steps."""
    chain = []
    name = tensor_name
    for _ in range(depth):
        prod = tensor_to_producer.get(name)
        if prod is None:
            break
        chain.append(prod.op_type)
        name = prod.input[0] if prod.input else ''
    return chain

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — InputContainer: Slice ops on graph input
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 1 — InputContainer: channel split in ONNX")

graph_input_names = {i.name for i in model_onnx.graph.input}

# Walk all Slice nodes, find those whose data input originates from graph input
slice_nodes = [n for n in model_onnx.graph.node if n.op_type == 'Slice']
print(f"Total Slice nodes in graph: {len(slice_nodes)}")

input_slices = []
for node in slice_nodes:
    data_inp = node.input[0] if node.input else ''
    # Follow chain back to see if it reaches the graph input
    chain = [data_inp]
    cur = data_inp
    for _ in range(6):
        if cur in graph_input_names:
            chain.append('<GRAPH_INPUT>')
            break
        prod = tensor_to_producer.get(cur)
        if prod is None:
            break
        cur = prod.input[0] if prod.input else ''
        chain.append(f"{prod.op_type}({cur})")

    is_from_input = any('<GRAPH_INPUT>' in str(c) for c in chain) or data_inp in graph_input_names

    starts, ends, axes, steps = get_slice_axes(node, init_map)
    rec = {
        'node_name': node.name,
        'data_input': data_inp,
        'from_graph_input': is_from_input,
        'starts': starts, 'ends': ends, 'axes': axes, 'steps': steps,
    }
    if is_from_input:
        input_slices.append(rec)

print(f"\nSlice nodes consuming graph input (InputContainer):")
if not input_slices:
    # May be onnxsim-merged — look for any Slice with axis=1 and small ends
    candidates = []
    for node in slice_nodes:
        s, e, ax, _ = get_slice_axes(node, init_map)
        if ax and 1 in ax:
            candidates.append({'node': node.name, 'starts': s, 'ends': e, 'axes': ax})
    print(f"  (No direct graph-input Slices found after simplification)")
    print(f"  Candidates with axis=1 (channel): {len(candidates)}")
    for c in candidates[:6]:
        print(f"    {c['node']:40s}  starts={c['starts']}  ends={c['ends']}  axes={c['axes']}")
else:
    for rec in input_slices:
        print(f"  node  : {rec['node_name']}")
        print(f"  starts: {rec['starts']}  ends: {rec['ends']}  axes: {rec['axes']}")
        if rec['axes'] is not None:
            nhwc_ax = nchw_to_nhwc_axis(rec['axes'])
            print(f"  → expected axes after onnx2tf NCHW→NHWC: {nhwc_ax}")
        print()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FusionRectify attention ops
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 2 — FusionRectify attention ops in ONNX")

# Count op types across the whole graph
from collections import Counter, defaultdict
op_counts = Counter(n.op_type for n in model_onnx.graph.node)
print("All op types in graph:")
for op, cnt in sorted(op_counts.items()):
    print(f"  {op:30s}: {cnt}")

# Identify FusionRectify channel branch ops
# Expected: ReduceMean(axes=[2,3]), ReduceMax(axes=[2,3]), Concat(axis=1), Gemm/MatMul, Sigmoid
sep("  Channel branch (GlobalPool + MLP + Sigmoid)")
for op_type in ['ReduceMean', 'ReduceMax', 'GlobalAveragePool', 'GlobalMaxPool']:
    nodes = [n for n in model_onnx.graph.node if n.op_type == op_type]
    for n in nodes:
        # Get axes attribute
        axes_attr = next((a for a in n.attribute if a.name == 'axes'), None)
        keepdims  = next((a for a in n.attribute if a.name == 'keepdims'), None)
        ax_val    = list(axes_attr.ints) if axes_attr else None
        kd_val    = keepdims.i if keepdims else None
        print(f"  {op_type:25s}: axes={ax_val}  keepdims={kd_val}  "
              f"→ expected NHWC axes: {nchw_to_nhwc_axis(ax_val) if ax_val else '?'}")

gemm_nodes = [n for n in model_onnx.graph.node if n.op_type in ('Gemm', 'MatMul')]
print(f"  Gemm/MatMul: {len(gemm_nodes)} nodes (Linear layers in channel MLP)")

sigmoid_nodes = [n for n in model_onnx.graph.node if n.op_type == 'Sigmoid']
print(f"  Sigmoid: {len(sigmoid_nodes)} nodes total")

sep("  Spatial branch (Concat + Conv1x1 + Relu + Sigmoid)")
# Find Conv nodes with kernel_shape=[1,1]
conv1x1_nodes = []
for n in model_onnx.graph.node:
    if n.op_type == 'Conv':
        ks_attr = next((a for a in n.attribute if a.name == 'kernel_shape'), None)
        if ks_attr and list(ks_attr.ints) == [1, 1]:
            # Check if it could be spatial conv by output channel count
            # Get weight shape from initializer
            w_name = n.input[1] if len(n.input) > 1 else None
            w_arr  = init_map.get(w_name) if w_name else None
            conv1x1_nodes.append({'node': n.name, 'weight': w_name})

print(f"  Conv(1×1) nodes: {len(conv1x1_nodes)}  "
      "(includes spatial_conv + any other pointwise convs in network)")

# Concat nodes
concat_nodes = [n for n in model_onnx.graph.node if n.op_type == 'Concat']
concat_axes  = defaultdict(int)
for n in concat_nodes:
    ax = next((a.i for a in n.attribute if a.name == 'axis'), None)
    concat_axes[ax] += 1
print(f"  Concat nodes by axis: {dict(concat_axes)}")
print(f"    axis=1 (channel in NCHW): {concat_axes[1]}  "
      f"→ axis=3 in NHWC expected after onnx2tf")

relu_nodes = [n for n in model_onnx.graph.node if n.op_type in ('Relu', 'LeakyRelu')]
print(f"  Relu nodes: {len(relu_nodes)}")

# Check for potential problem ops (onnx2tf decomposition risks)
RISKY_OPS = {'GridSample', 'NonMaxSuppression', 'ScatterND',
             'DeformConv', 'RoiAlign'}
found_risky = {op: op_counts[op] for op in RISKY_OPS if op in op_counts}
print(f"\n  Known-problematic ops for onnx2tf: "
      f"{'none' if not found_risky else found_risky}")

# Unsqueeze axes (important: Unsqueeze after Sigmoid for w_c reshape)
unsqueeze_nodes = [n for n in model_onnx.graph.node if n.op_type == 'Unsqueeze']
print(f"  Unsqueeze nodes: {len(unsqueeze_nodes)}  "
      f"(reshape attention weights from [B,C] to [B,C,1,1])")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Numerical parity
# ─────────────────────────────────────────────────────────────────────────────
sep("SECTION 3 — Numerical parity")

# ── Build input ────────────────────────────────────────────────────────────────
import cv2

if args.input == 'real':
    img_files = sorted([f for f in os.listdir(args.test_dir)
                        if f.lower().endswith(('.png','.jpg','.tif'))])
    if img_files:
        img_path = os.path.join(args.test_dir, img_files[0])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 3:
                nir = np.zeros((*img.shape[:2], 1), dtype=img.dtype)
                img = np.concatenate([img, nir], axis=2)
            img = cv2.resize(img, (IMGSZ, IMGSZ)).astype(np.float32) / 255.0
            inp_np = img.transpose(2, 0, 1)[np.newaxis]  # 1×4×H×W
            print(f"  Input: real image {img_files[0]}  shape={inp_np.shape}")
        else:
            print(f"  [WARN] Cannot read {img_path}, falling back to synthetic")
            args.input = 'synthetic'

if args.input == 'synthetic':
    # Structured: channels 0-2 are natural RGB ranges, channel 3 is NDVI [-1,1]→[0,1]
    np.random.seed(42)
    inp_np = np.random.rand(1, 4, IMGSZ, IMGSZ).astype(np.float32)
    # Simulate NDVI contrast between vegetation and background
    inp_np[0, 3, IMGSZ//4:3*IMGSZ//4, IMGSZ//4:3*IMGSZ//4] = 0.8   # high NDVI center
    inp_np[0, 3, :IMGSZ//4, :] = 0.2   # low NDVI edges
    print(f"  Input: synthetic 4ch  shape={inp_np.shape}  "
          f"ch3(NDVI) range=[{inp_np[0,3].min():.2f},{inp_np[0,3].max():.2f}]")

inp_torch = torch.from_numpy(inp_np)

# ── FP32 PyTorch ───────────────────────────────────────────────────────────────
from ultralytics import YOLO

yolo = YOLO(args.model_pt)
net  = yolo.model.cpu().eval()

with torch.no_grad():
    pt_out = net(inp_torch)

# pt_out is tuple; [0] is [1, 8, 8400] = [batch, 4+nc, n_preds]
pt_raw = pt_out[0].numpy()  # [1, 8, 8400]
print(f"\n  PyTorch FP32 output shape: {pt_raw.shape}")

# ── FP32 ONNX-RT ──────────────────────────────────────────────────────────────
sess_fp32 = ort.InferenceSession(
    args.onnx,
    providers=['CPUExecutionProvider']
)
inp_name_ort = sess_fp32.get_inputs()[0].name
ort_out_fp32 = sess_fp32.run(None, {inp_name_ort: inp_np})[0]   # [1, 8, 8400]
print(f"  ONNX-RT FP32 output shape: {ort_out_fp32.shape}")

sep("  3a. FP32 PyTorch vs FP32 ONNX-RT")
diff_fp32 = np.abs(pt_raw - ort_out_fp32)
print(f"  Max abs diff  : {diff_fp32.max():.6f}")
print(f"  Mean abs diff : {diff_fp32.mean():.8f}")
print(f"  99th pctile   : {np.percentile(diff_fp32, 99):.6f}")
# Per-dimension breakdown
print(f"  Max diff by dim:")
print(f"    box coords ([:, :4, :])  : {np.abs(pt_raw[:,:4,:] - ort_out_fp32[:,:4,:]).max():.6f}")
print(f"    class scores ([:,4:,:])  : {np.abs(pt_raw[:,4:,:] - ort_out_fp32[:,4:,:]).max():.6f}")

# ── FP16 ONNX-RT (proxy for TFLite FP16 arithmetic) ──────────────────────────
# Method: cast model inputs+outputs to FP16 via ORT FP16 session
try:
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import copy

    model_fp16 = copy.deepcopy(model_onnx)
    model_fp16 = convert_float_to_float16(model_fp16, keep_io_types=True)
    onnx_fp16_path = args.onnx.replace('.onnx', '_fp16_proxy.onnx')
    onnx.save(model_fp16, onnx_fp16_path)

    sess_fp16 = ort.InferenceSession(onnx_fp16_path,
                                      providers=['CPUExecutionProvider'])
    ort_out_fp16 = sess_fp16.run(None, {inp_name_ort: inp_np})[0]

    sep("  3b. FP32 PyTorch vs FP16 ONNX-RT (FP16 proxy for TFLite FP16)")
    diff_fp16 = np.abs(pt_raw - ort_out_fp16)
    print(f"  Max abs diff  : {diff_fp16.max():.6f}")
    print(f"  Mean abs diff : {diff_fp16.mean():.8f}")
    print(f"  99th pctile   : {np.percentile(diff_fp16, 99):.6f}")
    print(f"  Max diff by dim:")
    print(f"    box coords ([:, :4, :])  : {np.abs(pt_raw[:,:4,:] - ort_out_fp16[:,:4,:]).max():.6f}")
    print(f"    class scores ([:,4:,:])  : {np.abs(pt_raw[:,4:,:] - ort_out_fp16[:,4:,:]).max():.6f}")

except Exception as e:
    sep("  3b. FP16 ONNX-RT")
    print(f"  [SKIP] FP16 via convert_float_to_float16 failed: {e}")
    print(f"  Fallback: manual FP32→FP16→FP32 round-trip on PyTorch output")
    pt_fp16_roundtrip = pt_raw.astype(np.float16).astype(np.float32)
    ort_fp16_roundtrip = ort_out_fp32.astype(np.float16).astype(np.float32)
    diff_rt = np.abs(pt_fp16_roundtrip - ort_fp16_roundtrip)
    print(f"  (FP16 round-trip diff, shows FP16 quantization noise floor):")
    print(f"  Max abs diff (FP32 vs FP16 round-trip): {np.abs(pt_raw - pt_fp16_roundtrip).max():.6f}")
    print(f"  Mean abs diff                          : {np.abs(pt_raw - pt_fp16_roundtrip).mean():.8f}")

# ── TFLite FP16 (if tensorflow available) ─────────────────────────────────────
tflite_ok = False
if os.path.exists(args.tflite_fp16):
    try:
        import tensorflow as tf

        interp = tf.lite.Interpreter(model_path=args.tflite_fp16)
        interp.allocate_tensors()
        inp_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        # TFLite FP16 input: NHWC float32 (the delegate converts internally)
        inp_nhwc = inp_np.transpose(0, 2, 3, 1)  # NCHW → NHWC
        interp.set_tensor(inp_det['index'], inp_nhwc)
        interp.invoke()
        tfl_out = interp.get_tensor(out_det['index'])  # may be [1, 8400, 8] or [1, 8, 8400]

        sep("  3c. FP32 PyTorch vs TFLite FP16")
        print(f"  TFLite FP16 output shape: {tfl_out.shape}")

        # Align shape with PyTorch [1, 8, 8400]
        if tfl_out.shape == pt_raw.shape:
            diff_tfl = np.abs(pt_raw - tfl_out)
        elif tfl_out.shape[::-1] == pt_raw.shape or (
                tfl_out.ndim == 3 and tfl_out.shape[0] == 1
                and tfl_out.shape[2] == pt_raw.shape[1]
                and tfl_out.shape[1] == pt_raw.shape[2]):
            diff_tfl = np.abs(pt_raw - tfl_out.transpose(0, 2, 1))
            print(f"  (TFLite output transposed [1,8400,8]→[1,8,8400] for comparison)")
        else:
            print(f"  [WARN] Shape mismatch: PyTorch {pt_raw.shape} vs TFLite {tfl_out.shape}")
            diff_tfl = None

        if diff_tfl is not None:
            print(f"  Max abs diff  : {diff_tfl.max():.6f}")
            print(f"  Mean abs diff : {diff_tfl.mean():.8f}")
            print(f"  99th pctile   : {np.percentile(diff_tfl, 99):.6f}")
            print(f"  Max diff by dim:")
            try:
                print(f"    box coords ([:,:4,:])  : "
                      f"{np.abs(pt_raw[:,:4,:] - (tfl_out if diff_tfl is not None else pt_raw)[:,:4,:]).max():.6f}")
            except:
                pass

        tflite_ok = True
    except ImportError:
        sep("  3c. TFLite FP16")
        print(f"  [SKIP] tensorflow not installed (pip install tensorflow)")
    except Exception as e:
        sep("  3c. TFLite FP16")
        print(f"  [SKIP] TFLite comparison error: {e}")
else:
    sep("  3c. TFLite FP16")
    print(f"  [SKIP] TFLite FP16 model not found: {args.tflite_fp16}")
    print(f"         Run bench_step3_convert_tflite.py first, then re-run this script.")

# ── Detection-level parity (FP32 PyTorch vs FP32 ONNX) ───────────────────────
sep("  3d. Detection-level parity (FP32 PyTorch vs FP32 ONNX)")

CONF_THR = 0.05
def extract_detections(raw, conf_thr=CONF_THR):
    """
    raw: [1, 8, 8400] — channels: [cx, cy, w, h, c0, c1, c2, c3] (YOLO format)
    Returns list of (score, cls, box=[cx,cy,w,h])
    """
    preds = raw[0]  # [8, 8400]
    boxes  = preds[:4, :].T   # [8400, 4]
    scores = preds[4:, :].T   # [8400, NC]
    max_scores = scores.max(axis=1)
    cls        = scores.argmax(axis=1)
    mask       = max_scores > conf_thr
    return list(zip(max_scores[mask].tolist(),
                    cls[mask].tolist(),
                    boxes[mask].tolist()))

dets_pt  = extract_detections(pt_raw)
dets_ort = extract_detections(ort_out_fp32)
print(f"  PyTorch  detections > {CONF_THR}: {len(dets_pt)}")
print(f"  ONNX-RT  detections > {CONF_THR}: {len(dets_ort)}")

if dets_pt and dets_ort:
    # Sort by score descending, compare top-k
    dets_pt_s  = sorted(dets_pt,  key=lambda x: -x[0])
    dets_ort_s = sorted(dets_ort, key=lambda x: -x[0])
    k = min(10, len(dets_pt_s), len(dets_ort_s))
    print(f"\n  Top-{k} detections (score | cls | IoU_PT_vs_ORT | score_delta):")
    print(f"  {'#':3s}  {'score_PT':8s}  {'score_ORT':9s}  {'cls_PT':6s}  "
          f"{'cls_ORT':7s}  {'IoU':5s}  {'Δscore':7s}")
    for i in range(k):
        s_pt, c_pt, b_pt = dets_pt_s[i]
        s_or, c_or, b_or = dets_ort_s[i]
        box_iou = iou_box(b_pt, b_or)
        print(f"  {i+1:3d}  {s_pt:8.4f}  {s_or:9.4f}  {c_pt:6d}  "
              f"{c_or:7d}  {box_iou:5.3f}  {abs(s_pt-s_or):7.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Summary JSON
# ─────────────────────────────────────────────────────────────────────────────
sep("SUMMARY")

results = {
    "model_pt": args.model_pt,
    "onnx": args.onnx,
    "input_type": args.input,
    "input_shape_nchw": list(inp_np.shape),
    "section1_input_slices": input_slices,
    "section2_op_counts": {k: v for k, v in op_counts.items()},
    "section3": {
        "pytorch_vs_onnx_fp32": {
            "max_abs_diff":  round(float(diff_fp32.max()),  6),
            "mean_abs_diff": round(float(diff_fp32.mean()), 8),
            "p99_abs_diff":  round(float(np.percentile(diff_fp32, 99)), 6),
        },
        "detections": {
            "pytorch_n": len(dets_pt),
            "onnx_n":    len(dets_ort),
        }
    }
}

out_json = os.path.join(ROOT, 'bench_results', 'verify_export.json')
os.makedirs(os.path.dirname(out_json), exist_ok=True)
with open(out_json, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved → {out_json}")
print("\nKey numbers to put in paper (Section 1 + 2):")
print(f"  InputContainer Slice axes (NCHW): see section1_input_slices above")
print(f"  FP32 PT vs FP32 ONNX max abs diff: {diff_fp32.max():.6f}")
print(f"  FP32 PT vs FP32 ONNX mean abs diff: {diff_fp32.mean():.8f}")
print(f"  TFLite FP16 comparison: {'completed' if tflite_ok else 'SKIPPED (install tensorflow)'}")
