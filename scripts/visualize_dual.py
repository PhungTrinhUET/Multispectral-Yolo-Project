import cv2
import numpy as np
import os
from pathlib import Path
from ultralytics import YOLO

# ==================== CẤU HÌNH (SỬA Ở ĐÂY) ====================
# 1. Đường dẫn Model (File best.pt)
MODEL_PATH = 'Multispectral_DuaLuoi_Project/Exp9_Dual_Rectify_CustomPretrain/weights/best.pt'

# 2. Thư mục chứa ảnh RGB (Dùng tập Test)
RGB_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/test'

# 3. Thư mục chứa ảnh NDVI (Thư mục gốc của NDVI)
# Script sẽ tự tìm file tương ứng trong này (kể cả nằm trong subfolder test nếu có)
NDVI_DIR = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_ndvi/images'

# 4. Nơi lưu kết quả
SAVE_DIR = 'Multispectral_DuaLuoi_Project/Final_Vis_Rectify_Result'
# ==============================================================

def find_ndvi_file(rgb_filename, ndvi_base_dir):
    """
    Tìm file NDVI tương ứng dựa trên tên file RGB (bỏ qua đuôi file).
    Ví dụ: RGB='abc.jpg' -> Tìm 'abc.png', 'abc.tif', 'abc.jpg' trong folder NDVI.
    """
    rgb_stem = Path(rgb_filename).stem # Lấy tên file không đuôi (VD: image_01)
    ndvi_base_path = Path(ndvi_base_dir)

    # 1. Tìm đệ quy (Recursive) trong thư mục NDVI
    # Giúp tìm thấy kể cả khi file nằm trong dataset_ndvi/images/test/
    for file_path in ndvi_base_path.rglob('*'):
        if file_path.is_file() and file_path.stem == rgb_stem:
            return str(file_path)
    
    return None

def run_predict():
    # 1. Load Model
    print(f" Đang tải model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f" Lỗi không tải được model: {e}")
        return

    # 2. Quét file RGB
    rgb_path_obj = Path(RGB_DIR)
    if not rgb_path_obj.exists():
        print(f" Đường dẫn RGB không tồn tại: {RGB_DIR}")
        return

    # Lấy tất cả ảnh jpg, png, tif
    image_files = list(rgb_path_obj.glob('*.jpg')) + \
                  list(rgb_path_obj.glob('*.png')) + \
                  list(rgb_path_obj.glob('*.tif'))
    
    if not image_files:
        print(" Không tìm thấy ảnh nào trong thư mục RGB!")
        return

    print(f" Tìm thấy {len(image_files)} ảnh RGB. Bắt đầu xử lý...")
    
    success_count = 0
    fail_count = 0

    # 3. Vòng lặp xử lý
    for i, rgb_file in enumerate(image_files):
        filename = rgb_file.name
        
        try:
            # --- Bước A: Đọc RGB ---
            im_rgb = cv2.imread(str(rgb_file))
            if im_rgb is None:
                print(f" [{i+1}] Lỗi file ảnh RGB (Corrupt): {filename} -> Bỏ qua.")
                fail_count += 1
                continue
            
            h, w = im_rgb.shape[:2]

            # --- Bước B: Tìm NDVI ---
            ndvi_path = find_ndvi_file(filename, NDVI_DIR)
            
            if ndvi_path is None:
                print(f"⚠️ [{i+1}] Không tìm thấy cặp NDVI cho: {filename} -> Bỏ qua.")
                fail_count += 1
                continue

            # --- Bước C: Đọc NDVI ---
            im_nir = cv2.imread(ndvi_path, cv2.IMREAD_GRAYSCALE)
            if im_nir is None:
                print(f"⚠️ [{i+1}] Lỗi file ảnh NDVI (Corrupt): {ndvi_path} -> Bỏ qua.")
                fail_count += 1
                continue

            # --- Bước D: Resize NDVI (Quan trọng) ---
            # Đảm bảo kích thước khớp 100% với RGB
            if im_nir.shape != (h, w):
                im_nir = cv2.resize(im_nir, (w, h))

            # --- Bước E: Ghép kênh (Fusion) ---
            # Kết quả là ảnh 4 kênh (Blue, Green, Red, NIR)
            im_4ch = np.dstack((im_rgb, im_nir))

            # --- Bước F: Predict & Save ---
            model.predict(
                source=im_4ch, 
                imgsz=1280, 
                conf=0.25, 
                save=True, 
                project=SAVE_DIR, 
                name='predict_run',
                exist_ok=True,
                verbose=False # Tắt log rác
            )
            
            print(f"✅[{i+1}/{len(image_files)}] Xong: {filename}")
            success_count += 1

        except Exception as e:
            print(f"❌ [{i+1}] Lỗi không xác định với {filename}: {e}")
            fail_count += 1

    # 4. Tổng kết
    print("\n" + "="*30)
    print(f" HOÀN TẤT!")
    print(f" Thành công: {success_count} ảnh")
    print(f" Thất bại/Bỏ qua: {fail_count} ảnh")
    print(f" Xem kết quả tại: {os.path.join(SAVE_DIR, 'predict_run')}")
    print("="*30)

if __name__ == "__main__":
    run_predict()