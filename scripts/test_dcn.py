import torch
from torchvision.ops import DeformConv2d
import torch.nn as nn

def test_dcn():
    print("⚡ Đang kiểm tra Deformable Conv trên GPU...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Giả lập input giống hệt mô hình của bạn
    # Batch=4, Channels=256, Size=80x80 (tương tự P3)
    input_tensor = torch.randn(4, 256, 80, 80).to(device)
    
    # Offset: 18 kênh (2 * 3 * 3)
    offset = torch.zeros(4, 18, 80, 80).to(device)
    
    # Layer DCN
    dcn = DeformConv2d(256, 256, kernel_size=3, padding=1).to(device)
    
    try:
        # Chạy thử Forward pass
        output = dcn(input_tensor, offset)
        print(f" DCN hoạt động tốt! Output shape: {output.shape}")
    except Exception as e:
        print(f" Lỗi Python: {e}")

if __name__ == "__main__":
    test_dcn()