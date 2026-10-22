"""
Step 8 — Collect all bench_results JSON files → Paper Tables 1 & 2.

Run on PC after all measurements are complete.

Usage:
    python scripts/bench_step8_collect_tables.py

Outputs:
  bench_results/paper_table1_accuracy.md
  bench_results/paper_table2_edge.md
  bench_results/paper_tables.json   (machine-readable, all numbers)
"""

import json, os, glob
from pathlib import Path

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH    = os.path.join(ROOT, 'bench_results')

# ── Load helpers ──────────────────────────────────────────────────────────────
def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def fmt(v, decimals=4):
    if v is None: return "—"
    return f"{v:.{decimals}f}"

# ── Table 1 — Accuracy retention (PC) ────────────────────────────────────────
step1   = load(os.path.join(BENCH, 'step1_profile.json'))
step4   = load(os.path.join(BENCH, 'step4_accuracy.json'))
sizes   = load(os.path.join(BENCH, 'step3_sizes.json'))

rows_t1 = []

# FP32 from step1 (PyTorch evaluation)
if step1:
    a = step1['accuracy']
    rows_t1.append({
        'Variant': 'FP32 (.pt)',
        'Precision': a.get('precision'),
        'Recall':    a.get('recall'),
        'mAP@0.5':  a.get('map50'),
        'mAP@0.5:0.95': a.get('map5095'),
        'Size_MB':  step1.get('size_mb'),
        'Params_M': step1.get('params_M'),
        'GFLOPs':   step1.get('gflops'),
        'note':     'PyTorch (.pt) — reference',
    })

# FP32 / FP16 / INT8 from step4 (TFLite evaluation)
if step4:
    for variant in ['FP32', 'FP16', 'INT8']:
        r = step4.get(variant)
        if not r: continue
        sz = sizes.get(variant, {}).get('size_mb') if sizes else None
        rows_t1.append({
            'Variant': f'{variant} (.tflite)',
            'Precision': None,  # not computed in step4 (add if needed)
            'Recall':    None,
            'mAP@0.5':  r.get('map50'),
            'mAP@0.5:0.95': r.get('map5095'),
            'Size_MB':  sz or r.get('size_mb'),
            'Params_M': step1.get('params_M') if step1 else None,
            'GFLOPs':   step1.get('gflops')   if step1 else None,
            'note':     'TFLite (PC CPU evaluation)',
        })

# ── Table 2 — Edge deployment ─────────────────────────────────────────────────
rows_t2 = []

# Collect VOXL2 results
for json_path in sorted(glob.glob(os.path.join(BENCH, 'voxl2_*.json'))):
    r = load(json_path)
    if not r: continue
    b = r.get('burst', {})
    s = r.get('sustained', {})
    res = r.get('resources', {})
    # Load corresponding power file
    power_path = json_path.replace('voxl2_', 'power_').replace('.json', '.json')
    # Try alternate naming: power_{variant}_{delegate}.json
    v = r.get('variant', '')
    d = r.get('actual_delegate', r.get('requested_delegate', ''))
    power = load(os.path.join(BENCH, f"power_{v}_{d.split('_')[0]}.json"))
    pwr_w    = power.get('inference_power_mean_w') if power else None
    sus_fps  = s.get('fps_mean')
    energy_mj = round(pwr_w / sus_fps * 1000, 1) if (pwr_w and sus_fps) else None

    rows_t2.append({
        'Device':   'VOXL2 Sentinel',
        'Variant':  v.upper(),
        'Runtime':  d,
        'Lat_mean': b.get('latency_mean_ms'),
        'Lat_p95':  b.get('latency_p95_ms'),
        'FPS_burst': b.get('fps_burst'),
        'FPS_sust': s.get('fps_mean'),
        'RAM_MB':   res.get('peak_ram_mb'),
        'Power_W':  pwr_w,
        'Energy_mJ': energy_mj,
        'Temp_C':   res.get('soc_temp_c'),
        'Size_MB':  r.get('model_size_mb'),
    })

# Collect Jetson results
for json_path in sorted(glob.glob(os.path.join(BENCH, 'jetson_*.json'))):
    r = load(json_path)
    if not r: continue
    b = r.get('burst', {})
    s = r.get('sustained', {})
    res = r.get('resources', {})
    v = r.get('variant', '')
    power = load(os.path.join(BENCH, f"jetson_power_{v}.json"))
    pwr_w   = power.get('inference_power_mean_w') if power else None
    sus_fps = s.get('fps_mean')
    energy_mj = round(pwr_w / sus_fps * 1000, 1) if (pwr_w and sus_fps) else None

    rows_t2.append({
        'Device':   'Jetson Orin Nano',
        'Variant':  v.upper(),
        'Runtime':  'TensorRT',
        'Lat_mean': b.get('latency_mean_ms'),
        'Lat_p95':  b.get('latency_p95_ms'),
        'FPS_burst': b.get('fps_burst'),
        'FPS_sust': s.get('fps_mean'),
        'RAM_MB':   res.get('peak_ram_mb'),
        'Power_W':  pwr_w,
        'Energy_mJ': energy_mj,
        'Temp_C':   res.get('soc_temp_c'),
        'Size_MB':  r.get('model_size_mb'),
    })

# ── Desktop GPU reference row (from step1) ────────────────────────────────────
if step1 and step1.get('pc_latency'):
    pc = step1['pc_latency']
    rows_t2.insert(0, {
        'Device':   f"Desktop GPU ({step1.get('device_profile','GPU')})",
        'Variant':  'FP32 (PyTorch)',
        'Runtime':  'PyTorch CUDA',
        'Lat_mean': pc.get('mean_ms'),
        'Lat_p95':  pc.get('p95_ms'),
        'FPS_burst': pc.get('fps'),
        'FPS_sust': '—',
        'RAM_MB':   '—',
        'Power_W':  '—',
        'Energy_mJ': '—',
        'Temp_C':   '—',
        'Size_MB':  step1.get('size_mb'),
        '_note':    'Upper-bound reference, not edge',
    })

# ── Render Table 1 ─────────────────────────────────────────────────────────────
def f(v, d=3):
    if v is None or v == '—': return '—'
    try: return f"{float(v):.{d}f}"
    except: return str(v)

t1_lines = [
    "## Table 1 — Accuracy Retention (measured on PC, test split)\n",
    f"Input resolution: 640×640 px  |  Dataset: Dưa Lưới melon, 4 classes\n",
    "",
    "| Variant | Params (M) | GFLOPs | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Size (MB) |",
    "|---------|-----------|--------|---------|-------------|-----------|--------|-----------|",
]
for r in rows_t1:
    t1_lines.append(
        f"| {r['Variant']:20s} | {f(r['Params_M'],2):10s} | "
        f"{f(r['GFLOPs'],2):6s} | {f(r['mAP@0.5'],4):7s} | "
        f"{f(r['mAP@0.5:0.95'],4):11s} | {f(r['Precision'],4):9s} | "
        f"{f(r['Recall'],4):6s} | {f(r['Size_MB'],1):9s} |"
    )

# ── Render Table 2 ─────────────────────────────────────────────────────────────
t2_lines = [
    "\n## Table 2 — Edge Deployment Benchmark\n",
    f"Input: 640×640 px  |  N=5000 inferences  |  Warmup=50 discarded\n",
    "Power: whole-board (NOT NPU-isolated) — stated explicitly.\n",
    "",
    "| Device | Variant | Runtime | Lat mean (ms) | p95 (ms) | FPS burst | FPS sustained | RAM (MB) | Power (W) | Energy/inf (mJ) | Temp (°C) | Size (MB) |",
    "|--------|---------|---------|-------------|---------|-----------|--------------|---------|---------|----------------|---------|---------|",
]
for r in rows_t2:
    t2_lines.append(
        f"| {str(r['Device']):22s} | {str(r['Variant']):7s} | "
        f"{str(r['Runtime']):15s} | {f(r['Lat_mean'],2):13s} | "
        f"{f(r['Lat_p95'],2):8s} | {f(r['FPS_burst'],1):9s} | "
        f"{f(r['FPS_sust'],1):12s} | {f(r['RAM_MB'],1):8s} | "
        f"{f(r['Power_W'],2):9s} | {f(r['Energy_mJ'],1):15s} | "
        f"{f(r['Temp_C'],1):8s} | {f(r['Size_MB'],1):9s} |"
    )

# ── Write files ────────────────────────────────────────────────────────────────
t1_path = os.path.join(BENCH, 'paper_table1_accuracy.md')
t2_path = os.path.join(BENCH, 'paper_table2_edge.md')
all_path = os.path.join(BENCH, 'paper_tables.json')

with open(t1_path, 'w') as f:
    f.write('\n'.join(t1_lines))
with open(t2_path, 'w') as f:
    f.write('\n'.join(t2_lines))
with open(all_path, 'w') as f:
    json.dump({'table1': rows_t1, 'table2': rows_t2}, f, indent=2)

print('\n'.join(t1_lines))
print('\n'.join(t2_lines))
print(f"\n[DONE] Tables saved:")
print(f"  {t1_path}")
print(f"  {t2_path}")
print(f"  {all_path}")
