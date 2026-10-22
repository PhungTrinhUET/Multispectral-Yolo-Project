import json
import os
import torch
import time
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device


def benchmark_model(weights_path, data_path, imgsz=1280, device='0'):
    print(f"\n{'='*20} BENCHMARKING STARTED {'='*20}")
    print(f"Model: {weights_path}")

    device = select_device(device)
    model = YOLO(weights_path)
    model.to(device)

    print(f"\n--- 1. COMPUTATIONAL COST ---")
    model.info(detailed=False)

    # Warmup
    print(f"\n--- 2. WARMUP ---")
    dummy_input = torch.randn(1, 4, imgsz, imgsz).to(device)
    for _ in range(10):
        model(dummy_input, verbose=False)

    # Speed test
    print(f"\n--- 3. SPEED TEST (Batch=1) ---")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    num_samples = 100
    t_start = time.time()
    for _ in range(num_samples):
        model(dummy_input, verbose=False)
    t_end = time.time()
    avg_ms = (t_end - t_start) / num_samples * 1000
    fps = 1000 / avg_ms
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"Inference: {avg_ms:.2f} ms  |  FPS: {fps:.2f}  |  VRAM: {max_memory:.2f} MB")

    # Accuracy
    print(f"\n--- 4. ACCURACY (val set) ---")
    metrics = model.val(data=data_path, split='val', imgsz=imgsz, batch=4, verbose=False)

    results = {
        "weights":       weights_path,
        "date":          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "imgsz":         imgsz,
        "mAP50":         round(float(metrics.box.map50), 4),
        "mAP50-95":      round(float(metrics.box.map), 4),
        "precision":     round(float(metrics.box.mp), 4),
        "recall":        round(float(metrics.box.mr), 4),
        "inference_ms":  round(avg_ms, 2),
        "fps":           round(fps, 2),
        "vram_mb":       round(max_memory, 2),
    }

    # Lưu vào thư mục experiment (cùng cấp với best.pt)
    exp_dir = Path(weights_path).parent.parent
    out_path = exp_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*20} SUMMARY {'='*20}")
    for k, v in results.items():
        print(f"  {k:<15}: {v}")
    print(f"\n  Saved -> {out_path}")
    print("="*50)

    return results


if __name__ == '__main__':
    DATA_YAML = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/data.yaml'
    WEIGHTS_LIST = [
        'Multispectral_DuaLuoi_Project/Exp1_NoPretrain_1280px/weights/best.pt',
        'Multispectral_DuaLuoi_Project/Exp7_Dual_AFF_CustomPretrain2/weights/best.pt',
        'Multispectral_DuaLuoi_Project/Exp9_Dual_Rectify_CustomPretrain/weights/best.pt',
        'Multispectral_DuaLuoi_Project/Exp11_CrossCBAM_Spatial/weights/best.pt',
        'Multispectral_DuaLuoi_Project/Exp12_Rectify_Feedback_Final/weights/best.pt',
        'Multispectral_DuaLuoi_Project/Exp13_GMD_Modulation/weights/best.pt',
        'Multispectral_DuaLuoi_Project/Exp09_CMX_Redux/weights/best.pt',
        'Multispectral_DuaLuoi_Project/Exp16_AgriFusion_Fixed/weights/best.pt',
    ]

    for w in WEIGHTS_LIST:
        try:
            benchmark_model(w, DATA_YAML, imgsz=1280)
        except Exception as e:
            print(f"[ERROR] {w}: {e}")