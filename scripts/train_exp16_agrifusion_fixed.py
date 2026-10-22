"""
Exp16: AgriFusion Fixed
- Fix for Exp7 architecture bug: both branches received identical 4ch input
- Weight surgery applied to Exp7 best.pt:
    Layer 1  (RGB): [32,4,3,3] → [32,3,3,3]  (drop 4th channel, keep RGB)
    Layer 11 (NIR): [16,4,3,3] → [16,1,3,3]  (average 4ch → 1ch NDVI)
- FusionAFF blocks at layers 22, 24, 26 are preserved from Exp7
- Now InputContainer correctly routes: RGB(3ch) → L1, NDVI(1ch) → L11
"""
from ultralytics import YOLO

DATA_YAML   = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/data.yaml'
PRETRAIN_PT = 'yolo_dual_aff_best_fixed.pt'

model = YOLO(PRETRAIN_PT)

model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=1280,
    batch=4,
    device=0,
    workers=8,
    project='Multispectral_DuaLuoi_Project',
    name='Exp16_AgriFusion_Fixed',
    exist_ok=False,

    # Match Exp7 hyperparams
    optimizer='auto',
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    patience=100,
    seed=0,
    deterministic=True,
    amp=True,
    close_mosaic=10,
    mosaic=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,
    verbose=True,
)
