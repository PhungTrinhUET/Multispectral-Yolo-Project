import torch
import time
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device

def benchmark_model(weights_path, data_path, imgsz=1280, device='0'):
    # 1. Khởi tạo
    print(f"\n{'='*20} BENCHMARKING STARTED {'='*20}")
    print(f"Model: {weights_path}")
    print(f"Image Size: {imgsz}")
    
    device = select_device(device)
    model = YOLO(weights_path)
    
    # Ép model load sang device (GPU)
    model.to(device)
    
    # 2. Lấy GFLOPs 
    # Lấy từ log có sẵn của model
    print(f"\n--- 1. COMPUTATIONAL COST ---")
    model.info(detailed=True) 

    # 3. Đo VRAM Usage 
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # 4. Chạy Warmup (Khởi động để tránh tính thời gian load model)
    print(f"\n--- 2. WARMUP ---")
    dummy_input = torch.randn(1, 4, imgsz, imgsz).to(device) # Input 4 kênh giả lập
    for _ in range(10):
        _ = model(dummy_input, verbose=False)
    
    # 5. Đo FPS & Inference Time (Batch size = 1)
    print(f"\n--- 3. SPEED TEST (Batch=1) ---")
    num_samples = 100
    t_start = time.time()
    for _ in range(num_samples):
        _ = model(dummy_input, verbose=False)
    t_end = time.time()
    
    total_time = t_end - t_start
    avg_time_per_img = total_time / num_samples * 1000 # ms
    fps = 1000 / avg_time_per_img
    
    print(f"Test samples: {num_samples}")
    print(f"Average Inference Time: {avg_time_per_img:.2f} ms")
    print(f"FPS (Batch 1): {fps:.2f}")

    # 6. Đo VRAM Peak
    max_memory = torch.cuda.max_memory_allocated(device) / (1024 ** 2) # MB
    print(f"\n--- 4. VRAM USAGE ---")
    print(f"Max VRAM Allocated: {max_memory:.2f} MB")

    # 7. Lấy chỉ số chính xác (Validation)
    print(f"\n--- 5. ACCURACY METRICS (mAP) ---")
    # Lưu ý: Chạy val thật trên tập dữ liệu test
    metrics = model.val(data=data_path, split='val', imgsz=imgsz, batch=4, verbose=True)
    
    print("\n" + "="*20 + " SUMMARY " + "="*20)
    print(f"Model: {weights_path}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")
    print(f"Inference: {avg_time_per_img:.2f} ms")
    print(f"FPS:       {fps:.2f}")
    print(f"VRAM:      {max_memory:.2f} MB")
    print("="*50)

if __name__ == '__main__':
    # Cấu hình đường dẫn
    WEIGHTS = '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics/Multispectral_DuaLuoi_Project/Exp11_CrossCBAM_Spatial/weights/best.pt'
    DATA_YAML = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/data.yaml'
    
    # Chạy benchmark
    # Lưu ý: Chỉ chạy khi đã có file best.pt
    try:
        benchmark_model(WEIGHTS, DATA_YAML, imgsz=1280)
    except Exception as e:
        print(f"Chưa chạy được Benchmark vì: {e}")
        print("DONE")