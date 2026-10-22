# scripts/weight_surgery_smart_v3.py
from ultralytics import YOLO
import torch

def perform_smart_surgery():
    print(" SMART SURGERY V3 (GMD-YOLO: R-G-NIR + NDVI)...")
    
    # 1. Load Donors
    print("   -> Loading YOLOv11s (for R-G-NIR Stream)...")
    s_donor = YOLO('yolo11s.pt').model.state_dict()
    
    print("   -> Loading YOLOv11n (for NDVI Stream)...")
    n_donor = YOLO('yolo11n.pt').model.state_dict()
    
    # 2. Init New Model
    print("   -> Creating Exp13 Model...")
    new_model = YOLO('configs/yolo11_dual_modulation.yaml', task='detect')
    new_sd = new_model.model.state_dict()

    # 3. MAPPING CONFIG
    # ---------------------------------------------------------
    # STREAM 1: R-G-NIR (Main) -> Lấy từ YOLOv11s
    # Mapping Layer 1->10 (New) từ Layer 0->9 (Donor S)
    # ---------------------------------------------------------
    stream1_map = {
        1:0, 2:1, 3:2,      # Stem
        4:3, 5:4,           # P3
        6:5, 7:6,           # P4
        8:7, 9:8, 10:9      # P5
    }
    
    # ---------------------------------------------------------
    # STREAM 2: NDVI (Aux) -> Lấy từ YOLOv11n
    # Mapping Layer 11->20 (New) từ Layer 0->9 (Donor N)
    # ---------------------------------------------------------
    stream2_map = {
        11:0, 12:1, 13:2,   # Stem (Layer 11 cần nén kênh)
        14:3, 15:4,         # P3
        16:5, 17:6,         # P4
        18:7, 19:8, 20:9    # P5
    }

    count = 0
    for key in new_sd.keys():
        parts = key.split('.')
        if len(parts) < 2 or not parts[1].isdigit(): continue
        
        layer_idx = int(parts[1])
        prefix_new = f"model.{layer_idx}."
        
        # --- XỬ LÝ STREAM 1 (R-G-NIR) ---
        if layer_idx in stream1_map:
            src_idx = stream1_map[layer_idx]
            src_key = key.replace(prefix_new, f'model.{src_idx}.')
            if src_key not in s_donor: src_key = key.replace(prefix_new, f'model.model.{src_idx}.')
            
            if src_key in s_donor:
                # Copy trực tiếp (3 kênh -> 3 kênh)
                if new_sd[key].shape == s_donor[src_key].shape:
                    new_sd[key] = s_donor[src_key].clone()
                    count += 1

        # --- XỬ LÝ STREAM 2 (NDVI) ---
        elif layer_idx in stream2_map:
            src_idx = stream2_map[layer_idx]
            src_key = key.replace(prefix_new, f'model.{src_idx}.')
            if src_key not in n_donor: src_key = key.replace(prefix_new, f'model.model.{src_idx}.')
            
            if src_key in n_donor:
                # ĐẶC BIỆT: Nén kênh cho Layer 11 (NDVI Input)
                # Source: [16, 3, 3, 3] -> Target: [16, 1, 3, 3]
                if layer_idx == 11 and 'conv.weight' in key and n_donor[src_key].shape[1] == 3:
                    w = n_donor[src_key]
                    w_compressed = w.sum(dim=1, keepdim=True) # Sum RGB lại thành 1 kênh
                    new_sd[key] = w_compressed.clone()
                    print(f"   Compressed NDVI Layer 11: 3->1 channel")
                    count += 1
                
                elif new_sd[key].shape == n_donor[src_key].shape:
                    new_sd[key] = n_donor[src_key].clone()
                    count += 1

    # Load & Save
    new_model.model.load_state_dict(new_sd)
    ckpt = {'model': new_model.model}
    torch.save(ckpt, 'yolo_dual_modulation_pretrain.pt')
    print(f" Done! Transferred {count} tensors. Ready for Exp 13.")

if __name__ == '__main__':
    perform_smart_surgery()