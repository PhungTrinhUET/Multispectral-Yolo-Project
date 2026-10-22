"""
Benchmark script for:
1. AgriFusion (Exp7) - patches InputContainer to [x, x] (both branches take full 4ch)
2. Base YOLO models - gets Params/GFLOPs via model.info()
"""
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from ultralytics.nn.modules.block import InputContainer


def patch_input_container_full4ch():
    """Patch InputContainer to pass full 4ch to both branches (for Exp7/Exp9 trained weights)."""
    def forward_full4ch(self, x):
        if x.shape[1] < 4:
            extra = torch.zeros_like(x[:, 0:1, ...])
            x = torch.cat([x, extra], dim=1)
        return [x, x]
    InputContainer.forward = forward_full4ch


def restore_input_container():
    """Restore original InputContainer.forward."""
    def forward_original(self, x):
        if x.shape[1] < 4:
            nir = torch.zeros_like(x[:, 0:1, ...])
        else:
            nir = x[:, 3:4, ...]
        return [x[:, :3, ...], nir]
    InputContainer.forward = forward_original


def benchmark_speed_only(weights_path, imgsz=1280, device='0', n_channels=4, n_runs=100):
    """Speed test only (no val). Returns inference_ms, fps, vram_mb, params, gflops."""
    print(f"\n{'='*20} SPEED TEST {'='*20}")
    print(f"Model: {weights_path}")

    dev = select_device(device)
    model = YOLO(weights_path)
    model.to(dev)

    info = model.info(detailed=False, verbose=False)
    # Extract params and gflops from model
    try:
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
    except:
        params = None
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(model.model, imgsz=imgsz)
    except:
        gflops = None

    dummy = torch.randn(1, n_channels, imgsz, imgsz).to(dev)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            model(dummy, verbose=False)

    # Speed test
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy, verbose=False)
    t1 = time.time()

    avg_ms = (t1 - t0) / n_runs * 1000
    fps = 1000 / avg_ms
    vram = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    print(f"Params: {params:.2f}M  |  GFLOPs: {gflops}  |  Inf: {avg_ms:.2f}ms  |  FPS: {fps:.2f}  |  VRAM: {vram:.1f}MB")
    return dict(
        params_M=round(params, 3) if params else None,
        gflops=round(gflops, 1) if gflops else None,
        inference_ms=round(avg_ms, 2),
        fps=round(fps, 2),
        vram_mb=round(vram, 1),
    )


def benchmark_base_yolo_info(weights_path, imgsz=1280, device='0'):
    """Get params/GFLOPs for base YOLO models (3-channel input)."""
    print(f"\n{'='*20} BASE YOLO INFO {'='*20}")
    print(f"Model: {weights_path}")

    dev = select_device(device)
    model = YOLO(weights_path)
    model.to(dev)
    model.info(detailed=False, verbose=True)

    try:
        params = sum(p.numel() for p in model.model.parameters()) / 1e6
    except:
        params = None
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(model.model, imgsz=imgsz)
    except:
        gflops = None

    dummy = torch.randn(1, 3, imgsz, imgsz).to(dev)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    n_runs = 100
    t0 = time.time()
    with torch.no_grad():
        for _ in range(10):
            model(dummy, verbose=False)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model(dummy, verbose=False)
    t1 = time.time()
    avg_ms = (t1 - t0) / n_runs * 1000
    fps = 1000 / avg_ms
    vram = torch.cuda.max_memory_allocated(dev) / (1024 ** 2)

    print(f"Params: {params:.2f}M | GFLOPs: {gflops} | Inf: {avg_ms:.2f}ms | FPS: {fps:.2f} | VRAM: {vram:.1f}MB")
    return dict(params_M=round(params, 3) if params else None,
                gflops=round(gflops, 1) if gflops else None,
                inference_ms=round(avg_ms, 2), fps=round(fps, 2), vram_mb=round(vram, 1))


if __name__ == '__main__':
    # ============================================================
    # 1. AgriFusion (Exp7) - patch InputContainer to [x, x]
    # ============================================================
    print("\n\n>>> BENCHMARKING AgriFusion (Exp7) with InputContainer patch [x,x]")
    patch_input_container_full4ch()
    try:
        r7 = benchmark_speed_only(
            'Multispectral_DuaLuoi_Project/Exp7_Dual_AFF_CustomPretrain2/weights/best.pt',
            imgsz=1280, n_channels=4
        )
        # Merge with existing mAP from results.csv (best epoch)
        existing = json.loads(open(
            'Multispectral_DuaLuoi_Project/Exp7_Dual_AFF_CustomPretrain2/benchmark_results.json'
        ).read()) if Path('Multispectral_DuaLuoi_Project/Exp7_Dual_AFF_CustomPretrain2/benchmark_results.json').exists() else {}

        result7 = {
            'weights': 'Multispectral_DuaLuoi_Project/Exp7_Dual_AFF_CustomPretrain2/weights/best.pt',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'imgsz': 1280,
            'params_M': r7.get('params_M'),
            'gflops': r7.get('gflops'),
            'mAP50':    existing.get('mAP50', 0.5872),
            'mAP50-95': existing.get('mAP50-95', 0.2483),
            'precision': existing.get('precision', 0.6650),
            'recall':    existing.get('recall', 0.5334),
            'inference_ms': r7['inference_ms'],
            'fps':          r7['fps'],
            'vram_mb':      r7['vram_mb'],
            'note': 'mAP from training results.csv best epoch; speed from batch=1 synthetic 4ch input; InputContainer patched to [x,x] matching trained weights'
        }
        with open('Multispectral_DuaLuoi_Project/Exp7_Dual_AFF_CustomPretrain2/benchmark_results.json', 'w') as f:
            json.dump(result7, f, indent=2)
        print(f"Saved AgriFusion benchmark: FPS={r7['fps']}, VRAM={r7['vram_mb']}MB")
    except Exception as e:
        print(f"[ERROR] Exp7: {e}")
    finally:
        restore_input_container()

    # ============================================================
    # 2. Base YOLO models - params/GFLOPs
    # ============================================================
    base_dir = Path('/home/tower2080/Documents/DuaLuoi/yolov5/runs_train')
    base_models = {
        'YOLOv5s': base_dir / 'yolov5s_1280_balanced_b4/weights/best.pt',
        'YOLOv8s': base_dir / 'yolov8s_1280_balanced_b4/weights/best.pt',
        'YOLOv10s': base_dir / 'yolov10s_1280_balanced_b4/weights/best.pt',
        'YOLOv11s': base_dir / 'yolo11s_1280_balanced_b4/weights/best.pt',
    }
    base_results = {}
    for name, wp in base_models.items():
        print(f"\n>>> {name}")
        try:
            r = benchmark_base_yolo_info(str(wp), imgsz=1280)
            base_results[name] = r
            # Update existing benchmark_results.json
            bpath = wp.parent.parent / 'benchmark_results.json'
            if bpath.exists():
                existing = json.loads(bpath.read_text())
                existing['params_M'] = r['params_M']
                existing['gflops'] = r['gflops']
                bpath.write_text(json.dumps(existing, indent=2))
                print(f"Updated {bpath}")
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print("\n\n=== FINAL SUMMARY ===")
    for k, v in base_results.items():
        print(f"{k}: {v}")
