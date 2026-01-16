import torch
import torch.nn as nn
from ultralytics import YOLO
import sys
import os

sys.path.append(os.getcwd())

def perform_surgery():
    print(" BẮT ĐẦU EDIT MODEL (WEIGHT SURGERY)...")

    # 1. Load 2 model gốc (Donor Models)
    print("   -> Loading YOLOv11s (RGB Donor)...")
    model_s = YOLO('yolo11s.pt')
    
    print("   -> Loading YOLOv11n (NDVI Donor)...")
    model_n = YOLO('yolo11n.pt')

    # 2. Khởi tạo model Dual rỗng (Patient Model)
    print("   -> Creating Dual-Backbone Skeleton...")
    # Lưu ý: Cần file config nằm đúng chỗ
    dual_model = YOLO('configs/yolo11_dual_backbone.yaml') 
    
    # Ép tạo weights bằng cách dry-run 1 lần (quan trọng để khởi tạo các layer)
    try:
        # Load model với scale='custom' để dùng đúng thông số trong YAML
        dual_model = YOLO('configs/yolo11_dual_backbone.yaml', task='detect') 
        # Ép scale custom nếu code YOLO không tự nhận 
        if hasattr(dual_model.model, 'yaml'):
             dual_model.model.yaml['scale'] = 'custom'
    except Exception as e:
        print(f" Warning during init: {e}")

    target_model = dual_model.model
    
    # 3. TIẾN HÀNH GHÉP TẠNG (TRANSFER WEIGHTS)
    # Theo file yaml:
    # Layer 0: InputContainer (Không có weights)
    # Layer 1-10: Nhánh RGB (Tương ứng Layer 0-9 của 11s)
    # Layer 11-20: Nhánh NDVI (Tương ứng Layer 0-9 của 11n)
    
    dict_s = model_s.model.state_dict()
    dict_n = model_n.model.state_dict()
    target_dict = target_model.state_dict()

    transferred_count = 0
    
    print("   -> Đang chuyển giao trọng số...")

    # Duyệt qua các layer của model đích
    for key in target_dict.keys():
        # === NHÁNH 1: RGB (Layer 1-10) ===
        if key.startswith('model.1.') or key.startswith('model.2.') or \
           key.startswith('model.3.') or key.startswith('model.4.') or \
           key.startswith('model.5.') or key.startswith('model.6.') or \
           key.startswith('model.7.') or key.startswith('model.8.') or \
           key.startswith('model.9.') or key.startswith('model.10.'):
            
            # Logic map: model.X -> model.(X-1) của 11s
            # Ví dụ: Dual model.1.conv.weight -> 11s model.0.conv.weight
            parts = key.split('.')
            layer_idx = int(parts[1])
            source_layer_idx = layer_idx - 1
            source_key = f"model.{source_layer_idx}.{'.'.join(parts[2:])}"
            
            if source_key in dict_s:
                if target_dict[key].shape == dict_s[source_key].shape:
                    target_dict[key] = dict_s[source_key].clone()
                    transferred_count += 1
                else:
                    print(f"      ⚠️ Lệch size RGB tại {key}: Target {target_dict[key].shape} vs Source {dict_s[source_key].shape}")

        # === NHÁNH 2: NDVI (Layer 11-20) ===
        elif key.startswith('model.11.') or key.startswith('model.12.') or \
             key.startswith('model.13.') or key.startswith('model.14.') or \
             key.startswith('model.15.') or key.startswith('model.16.') or \
             key.startswith('model.17.') or key.startswith('model.18.') or \
             key.startswith('model.19.') or key.startswith('model.20.'):

            # Logic map: model.X -> model.(X-11) của 11n
            parts = key.split('.')
            layer_idx = int(parts[1])
            source_layer_idx = layer_idx - 11
            source_key = f"model.{source_layer_idx}.{'.'.join(parts[2:])}"
            
            if source_key in dict_n:
                # ---LAYER ĐẦU TIÊN CỦA NDVI (NÉN 3 KÊNH THÀNH 1) ---
                if layer_idx == 11 and 'conv.weight' in key:
                    # Shape gốc 11n: [16, 3, 3, 3] -> Target: [16, 1, 3, 3]
                    w_source = dict_n[source_key]
                    # Lấy trung bình cộng theo chiều channels (dim=1)
                    w_compressed = torch.mean(w_source, dim=1, keepdim=True)
                    
                    if target_dict[key].shape == w_compressed.shape:
                        target_dict[key] = w_compressed.clone()
                        transferred_count += 1
                        print(f"       Đã nén weights 3 kênh -> 1 kênh cho Layer 11 (NDVI Input)")
                    else:
                        print(f"       Lỗi shape khi nén Layer 11: {target_dict[key].shape} vs {w_compressed.shape}")
                
                # Các layer khác của nhánh NDVI (Copy nguyên xi)
                elif target_dict[key].shape == dict_n[source_key].shape:
                    target_dict[key] = dict_n[source_key].clone()
                    transferred_count += 1
                else:
                     print(f"       Lệch size NDVI tại {key}: Target {target_dict[key].shape} vs Source {dict_n[source_key].shape}")

    # Load dict đã chỉnh sửa vào model
    target_model.load_state_dict(target_dict)
    
    # Lưu file
    save_path = 'yolo_dual_pretrain.pt'
    torch.save({'model': target_model}, save_path) # Lưu dạng raw dictionary bọc trong 'model' để YOLO load được
    # Hoặc chuẩn hơn theo Ultralytics format:
    ckpt = {'epoch': -1, 'best_fitness': None, 'model': target_model, 'optimizer': None}
    torch.save(ckpt, save_path)
    
    print(f" PHẪU THUẬT THÀNH CÔNG! Đã chuyển {transferred_count} tensors.")
    print(f" File trọng số đã lưu tại: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    perform_surgery()