"""
Exp17: AFF Clean Run
- Uses FusionAFF (AgriFusion Attention Fusion Feature, Eq.2-6)
- Pretrained: yolo_dual_aff_pretrain_fixed.pt (correct AFF architecture weight)
- Hyperparams: identical to Exp09_CMX_Redux (CM-FRM baseline)
- Config: configs/yolo11_dual_aff.yaml
"""
from ultralytics import YOLO

MODEL_CFG  = 'configs/yolo11_dual_aff.yaml'
PRETRAIN   = 'yolo_dual_aff_pretrain_fixed.pt'
DATA_YAML  = 'dataset_melon.yaml'

model = YOLO(MODEL_CFG)
model.load(PRETRAIN)  # weight surgery: backbone layers transfer, fusion/head random-init

model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=1280,
    batch=4,
    device=0,
    workers=8,
    project='Multispectral_DuaLuoi_Project',
    name='Exp17_AFF_Clean',
    exist_ok=False,

    # Identical to Exp09_CMX_Redux
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
