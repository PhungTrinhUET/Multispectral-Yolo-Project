# scripts/weight_surgery_smart_v2.py
from ultralytics import YOLO
import torch
import torch.nn as nn
import sys

def perform_smart_surgery():
    print("🏥 BẮT ĐẦU PHẪU THUẬT THÔNG MINH V2 (FINAL FIX)...")
    
    # 1. Load các "Người hiến tạng" (Donors)
    print("   -> Loading RGB Donor (YOLOv11s)...")
    try:
        rgb_donor = YOLO('yolo11s.pt')
        rgb_sd = rgb_donor.model.state_dict()
    except Exception as e:
        print(f"❌ Lỗi load yolo11s.pt: {e}")
        return

    print("   -> Loading NIR Donor (YOLOv11n)...")
    try:
        nir_donor = YOLO('yolo11n.pt')
        nir_sd = nir_donor.model.state_dict()
    except Exception as e:
        print(f"❌ Lỗi load yolo11n.pt: {e}")
        return
    
    # 2. Khởi tạo Model mới (Exp 12) từ file Config
    print("   -> Creating Exp12 Skeleton (Dual-Stream Rectified Feedback)...")
    try:
        # Lưu ý: Đảm bảo đường dẫn config đúng
        new_model = YOLO('configs/yolo11_dual_rectify_feedback.yaml', task='detect')
        new_sd = new_model.model.state_dict()
    except Exception as e:
        print(f"❌ Lỗi tạo model từ config: {e}")
        return

    # DEBUG: In key mẫu để kiểm tra
    first_key = next(iter(new_sd.keys()))
    print(f"   ℹ️  Sample key format in new model: '{first_key}'")

    # ==============================================================================
    # 🗺️ BẢN ĐỒ CHỈ ĐƯỜNG (MAPPING CONFIG)
    # Key: Layer index trong Exp 12 (Model Mới)
    # Value: Layer index trong Donor (Model Gốc v11s/v11n)
    # ==============================================================================
    
    # --- NHÁNH RGB (Lấy từ yolo11s) ---
    # Cấu trúc v11s gốc: Stem(0,1,2) -> P3(3,4) -> P4(5,6) -> P5(7,8,9)
    rgb_map = {
        1: 0, 2: 1, 3: 2,       # Stem
        7: 3, 8: 4,             # P3 (Target 256 khớp Source 256)
        15: 5, 16: 6,           # P4 (Target 512 khớp Source 512)
        23: 7, 24: 8, 25: 9     # P5 (Target 512 khớp Source 512)
    }
    
    # --- NHÁNH NIR (Lấy từ yolo11n) ---
    nir_map = {
        4: 0, 5: 1, 6: 2,       # Stem (Layer 4 sẽ được nén kênh 3->1)
        9: 3, 10: 4,            # P3
        17: 5, 18: 6,           # P4
        26: 7, 27: 8, 28: 9     # P5
    }

    # ==============================================================================
    # 💉 TIẾN HÀNH GHÉP TẠNG
    # ==============================================================================
    transferred_count = 0
    
    for key in new_sd.keys():
        parts = key.split('.')
        layer_idx = -1
        
        # --- LOGIC TỰ DÒ TÌM LAYER INDEX ---
        # Xử lý cả 2 trường hợp tên key: "model.4..." hoặc "model.model.4..."
        if len(parts) > 1 and parts[1].isdigit():
            layer_idx = int(parts[1])
            prefix_new = f"model.{layer_idx}."
        elif len(parts) > 2 and parts[2].isdigit():
            layer_idx = int(parts[2])
            prefix_new = f"model.model.{layer_idx}."
        else:
            continue # Bỏ qua các layer không xác định index (như head output)

        # 1. XỬ LÝ NHÁNH RGB
        if layer_idx in rgb_map:
            src_idx = rgb_map[layer_idx]
            # Tạo key nguồn giả định
            src_key_short = key.replace(prefix_new, f'model.{src_idx}.')
            src_key_long = key.replace(prefix_new, f'model.model.{src_idx}.')
            
            # Tìm key chính xác trong donor
            if src_key_short in rgb_sd: src_key = src_key_short
            elif src_key_long in rgb_sd: src_key = src_key_long
            else: src_key = None

            if src_key:
                # Kiểm tra kích thước (Shape)
                if new_sd[key].shape == rgb_sd[src_key].shape:
                    new_sd[key] = rgb_sd[src_key].clone()
                    transferred_count += 1
                else:
                    print(f"   ⚠️ RGB Mismatch at {key}: Target {new_sd[key].shape} != Source {rgb_sd[src_key].shape}")

        # 2. XỬ LÝ NHÁNH NIR
        elif layer_idx in nir_map:
            src_idx = nir_map[layer_idx]
            src_key_short = key.replace(prefix_new, f'model.{src_idx}.')
            src_key_long = key.replace(prefix_new, f'model.model.{src_idx}.')
            
            if src_key_short in nir_sd: src_key = src_key_short
            elif src_key_long in nir_sd: src_key = src_key_long
            else: src_key = None
            
            if src_key:
                # ĐẶC BIỆT: Nén kênh cho Layer 4 (Conv đầu vào NIR)
                if layer_idx == 4 and 'conv.weight' in key:
                    w = nir_sd[src_key]
                    if w.shape[1] == 3: # Nếu nguồn là 3 kênh (RGB)
                        # Tổng hợp 3 kênh thành 1 kênh (Sum Pooling)
                        w_compressed = w.sum(dim=1, keepdim=True) 
                        new_sd[key] = w_compressed.clone()
                        print(f"   ✅ Compressed Layer {layer_idx} (NIR Stem): 3->1 channel")
                        transferred_count += 1
                
                # Các layer khác copy bình thường
                elif new_sd[key].shape == nir_sd[src_key].shape:
                    new_sd[key] = nir_sd[src_key].clone()
                    transferred_count += 1
                else:
                    # In warning nếu lệch size (để debug)
                    pass 

    # 3. Load weights đã chỉnh sửa vào model
    new_model.model.load_state_dict(new_sd)
    
    # ==============================================================================
    # 4. LƯU FILE CHECKPOINT CHUẨN (FIX LỖI KEYERROR 'model')
    # ==============================================================================
    save_path = 'yolo_dual_rectify_feedback_pretrain.pt'
    
    # Quan trọng: YOLO yêu cầu file .pt phải là một dictionary chứa key 'model'
    ckpt = {
        'model': new_model.model,
        'epoch': -1,
        'optimizer': None,
    }
    
    torch.save(ckpt, save_path)
    
    print(f"\n🎉 PHẪU THUẬT HOÀN TẤT! Đã chuyển: {transferred_count} tensors.")
    print(f"💾 File lưu tại: {save_path} (Format: YOLO Checkpoint Dictionary)")
    print("👉 Bây giờ bạn có thể chạy lệnh 'yolo train' mà không bị lỗi KeyError nữa!")

if __name__ == '__main__':
    perform_smart_surgery()