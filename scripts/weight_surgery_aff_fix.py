"""
Weight surgery for AgriFusion (AFF) fix:
- Layer 1  (RGB branch): keep [32, 3, 3, 3] → takes 3ch RGB ✓
- Layer 11 (NIR branch): patch Conv2d from [16,3,3,3] → [16,1,3,3]
  by averaging across RGB channels (1ch NDVI input)
Output: yolo_dual_aff_pretrain_fixed.pt
"""
import torch
import torch.nn as nn
from copy import deepcopy

print("Loading AFF pretrain as full model object...")
ckpt = torch.load('yolo_dual_aff_pretrain.pt', map_location='cpu')
model = deepcopy(ckpt['model'])

# --- Fix Layer 11 Conv: 3ch → 1ch ---
old_conv = model.model[11].conv          # nn.Conv2d(3, 16, 3, 2, 1)
old_w    = old_conv.weight.data          # [16, 3, 3, 3]
old_b    = old_conv.bias                 # None usually

new_conv = nn.Conv2d(
    in_channels=1,
    out_channels=old_conv.out_channels,
    kernel_size=old_conv.kernel_size,
    stride=old_conv.stride,
    padding=old_conv.padding,
    bias=old_conv.bias is not None,
)
# Compress RGB channels → 1ch by averaging
new_conv.weight.data = old_w.mean(dim=1, keepdim=True)
if old_b is not None:
    new_conv.bias.data = old_b.clone()

model.model[11].conv = new_conv

print(f"  Layer 11 conv: {old_w.shape} → {new_conv.weight.data.shape}")

# Verify other key layers unchanged
print(f"  Layer 1  conv: {model.model[1].conv.weight.shape}  (RGB, unchanged)")

# Save
save_ckpt = {'model': model, 'epoch': -1}
torch.save(save_ckpt, 'yolo_dual_aff_pretrain_fixed.pt')
print("\nSaved → yolo_dual_aff_pretrain_fixed.pt")

# Final verification
ckpt2 = torch.load('yolo_dual_aff_pretrain_fixed.pt', map_location='cpu')
m2 = ckpt2['model']
print(f"  Verify Layer 1  conv: {m2.model[1].conv.weight.shape}")
print(f"  Verify Layer 11 conv: {m2.model[11].conv.weight.shape}")
print("Done.")
