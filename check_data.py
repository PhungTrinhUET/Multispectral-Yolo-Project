import cv2
import numpy as np
import os

# Đường dẫn file mẫu 
raw_path = "/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/train/2019_0101_000031_012.jpg"
ndvi_path = "/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi/images/train/2019_0101_000031_012.png"

def check_images():
    print("--- KIỂM TRA DỮ LIỆU ĐẦU VÀO ---")
    
    # 1. Kiểm tra ảnh RAW (R-G-NIR)
    if not os.path.exists(raw_path):
        print(f"LỖI: Không tìm thấy file RAW tại {raw_path}")
        return
    
    # Đọc ảnh Raw
    img_raw = cv2.imread(raw_path) # Mặc định OpenCV đọc là BGR, nhưng bản chất dữ liệu là R-G-NIR
    
    print(f"1. Ảnh RAW (Dataset R-G-NIR):")
    print(f"   - Shape: {img_raw.shape} (Cao, Rộng, Kênh)")
    print(f"   - Dtype: {img_raw.dtype}")
    print(f"   - Min/Max giá trị: {img_raw.min()} / {img_raw.max()}")
    if img_raw.shape[2] != 3:
        print("   -> SOS: Ảnh Raw không phải 3 kênh!")
    else:
        print("   -> OK: Ảnh Raw có 3 kênh.")

    # 2. Kiểm tra ảnh NDVI
    if not os.path.exists(ndvi_path):
        print(f"LỖI: Không tìm thấy file NDVI tại {ndvi_path}")
        return
    
    # Đọc chế độ UNCHANGED để biết chính xác số kênh gốc
    img_ndvi = cv2.imread(ndvi_path, cv2.IMREAD_UNCHANGED)
    
    print(f"\n2. Ảnh NDVI (Dataset NDVI):")
    print(f"   - Shape: {img_ndvi.shape}")
    print(f"   - Dtype: {img_ndvi.dtype}")
    print(f"   - Min/Max giá trị: {img_ndvi.min()} / {img_ndvi.max()}")
    
    # 3. Khả năng ghép (Merge)
    print("\n--- KẾT LUẬN ---")
    if img_raw.shape[:2] == img_ndvi.shape[:2]:
        print(" Kích thước (Cao, Rộng) khớp nhau -> Có thể ghép được.")
    else:
        print(f" Kích thước lệch nhau: Raw {img_raw.shape[:2]} vs NDVI {img_ndvi.shape[:2]}")

    if len(img_ndvi.shape) == 2:
        print("ℹ  Ảnh NDVI là 1 kênh -> Ghép thẳng thành kênh thứ 4.")
    elif len(img_ndvi.shape) == 3:
        print(" Ảnh NDVI là 3 kênh -> Cần lấy 1 kênh đại diện để ghép.")

if __name__ == "__main__":
    check_images()