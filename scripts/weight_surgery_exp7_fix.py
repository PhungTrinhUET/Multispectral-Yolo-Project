"""
Weight surgery for Exp7 AFF best.pt:
- Layer 11 (NIR branch first Conv): [16,4,3,3] → [16,1,3,3]
  (Exp7 was trained with 4ch in both branches; now NIR should receive 1ch NDVI)
- Averaging across 4 input channels preserves learned feature sensitivity
Output: yolo_dual_aff_best_fixed.pt
"""
import torch
import torch.nn as nn
from copy import deepcopy

print("Loading Exp7 AFF best.pt...")
ckpt = torch.load(
    'Multispectral_DuaLuoi_Project/Exp7_Dual_AFF_CustomPretrain2/weights/best.pt',
    map_location='cpu'
)
model = deepcopy(ckpt['model'])

# --- Fix Layer 1 (RGB branch): [32,4,3,3] → [32,3,3,3] (keep RGB channels, drop 4th) ---
conv1 = model.model[1].conv
w1    = conv1.weight.data               # [32, 4, 3, 3]
print(f"  Layer 1  conv BEFORE: {w1.shape}")
new_conv1 = nn.Conv2d(
    in_channels=3,
    out_channels=conv1.out_channels,
    kernel_size=conv1.kernel_size,
    stride=conv1.stride,
    padding=conv1.padding,
    bias=conv1.bias is not None,
)
new_conv1.weight.data = w1[:, :3, :, :]   # keep first 3 channels (R,G,B)
if conv1.bias is not None:
    new_conv1.bias.data = conv1.bias.data.clone()
model.model[1].conv = new_conv1
print(f"  Layer 1  conv AFTER:  {new_conv1.weight.data.shape}")

# --- Fix Layer 11 (NIR branch): [16,4,3,3] → [16,1,3,3] (average all 4 channels) ---
old_conv = model.model[11].conv
old_w    = old_conv.weight.data          # [16, 4, 3, 3]
print(f"  Layer 11 conv BEFORE: {old_w.shape}")
new_conv = nn.Conv2d(
    in_channels=1,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None,
)
new_conv.weight.data = old_w.mean(dim=1, keepdim=True)   # [16,4,3,3] → [16,1,3,3]
if old_conv.bias is not None:
    new_conv.bias.data = old_conv.bias.data.clone()
model.model[11].conv = new_conv
print(f"  Layer 11 conv AFTER:  {new_conv.weight.data.shape}")

# Save
save_ckpt = {k: v for k, v in ckpt.items()}
save_ckpt['model'] = model
save_ckpt['epoch'] = -1
torch.save(save_ckpt, 'yolo_dual_aff_best_fixed.pt')
print("\nSaved → yolo_dual_aff_best_fixed.pt")

# Verify
ck2 = torch.load('yolo_dual_aff_best_fixed.pt', map_location='cpu')
m2 = ck2['model']
print(f"  Verify Layer 1  (RGB): {m2.model[1].conv.weight.shape}")
print(f"  Verify Layer 11 (NIR): {m2.model[11].conv.weight.shape}")
print(f"  Verify Layer 22 type:  {type(m2.model[22]).__name__}")
print(f"  Total params: {sum(p.numel() for p in m2.parameters())/1e6:.2f}M")

# Quick forward pass via YOLO wrapper
from ultralytics import YOLO
import torch
model_yolo = YOLO('yolo_dual_aff_best_fixed.pt')
dummy = torch.randn(1, 4, 640, 640).cuda()
model_yolo.to('cuda')
with torch.no_grad():
    try:
        out = model_yolo(dummy, verbose=False)
        print("Forward pass OK")
    except Exception as e:
        print(f"Forward pass FAILED: {e}")
