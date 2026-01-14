from ultralytics.data.dataset import YOLODataset
import yaml
from pathlib import Path
import numpy as np

cfg_path = "dataset_melon.yaml"
try:
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    
    print(f"Đang kiểm tra với đường dẫn: {cfg['path']}")
    img_path = Path(cfg['path']) / cfg['train']
    
    print("Đang khởi tạo Dataset...")
    dataset = YOLODataset(img_path=str(img_path), data=cfg, task='detect')
    
    print("Đang load thử ảnh đầu tiên...")
    im_result = dataset.load_image(0)
    
    img_data = im_result[0] if isinstance(im_result, tuple) else im_result

    print("\n" + "="*30)
    print("KẾT QUẢ KIỂM TRA:")
    print(f"File ảnh: {dataset.im_files[0]}") 
    # ----------------------------------------
    print(f"Kích thước (H, W, Channels): {img_data.shape}")
    
    if img_data.shape[2] == 4:
        print(" THÀNH CÔNG: Đã đọc được 4 kênh (R, G, Raw, NDVI).")
        print("Bạn có thể chuyển sang bước Train.")
    else:
        print(f" THẤT BẠI: Chỉ đọc được {img_data.shape[2]} kênh.")
    print("="*30 + "\n")

except Exception as e:
    import traceback
    traceback.print_exc() # In chi tiết lỗi để dễ debug
    print(f"\n CÓ LỖI XẢY RA: {e}")