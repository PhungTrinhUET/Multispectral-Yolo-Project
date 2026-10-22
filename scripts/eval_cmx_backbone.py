"""
Eval Exp20 (CM-FRM @ Backbone) with canonical protocol:
  iou=0.6, conf=0.001, half=False, imgsz=1280, split=test (383 images)

Run after train_cmx_backbone.py finishes.
"""
from ultralytics import YOLO

CKPT = '/home/tower2080/Documents/YOLO-Knowledge-Distillation-TinyML/ultralytics/runs/detect/runs/ablation_position_backbone/CMX_Backbone/weights/best.pt'
DATA = 'dataset_melon.yaml'

model = YOLO(CKPT)
metrics = model.val(
    data=DATA,
    split='test',
    imgsz=1280,
    batch=4,
    iou=0.6,
    conf=0.001,
    half=False,
    device=0,
    verbose=True,
)

mp  = metrics.box.mp
mr  = metrics.box.mr
map50    = metrics.box.map50
map5095  = metrics.box.map
ap50_per = metrics.box.ap50   # per-class AP@50

cls_names = ['pre-pollination', 'post-pollination', 'net-formation', 'sweetening']

print("\n=== CM-FRM @ Backbone-Injection (Exp20) ===")
print(f"Overall: mAP@50={map50:.4f}  mAP@50:95={map5095:.4f}  P={mp:.4f}  R={mr:.4f}")
print("Per-class AP@50:")
for name, ap in zip(cls_names, ap50_per):
    print(f"  {name:<20} {ap:.4f}")

# Compare vs neck baseline
print("\nBaseline (Exp09 neck, iou=0.6): mAP@50=0.5817")
print(f"Delta:  {map50 - 0.5817:+.4f}")
