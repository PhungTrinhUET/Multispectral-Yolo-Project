"""
Exp20: CM-FRM @ Backbone-Injection — ablation on fusion position.

Identical to Exp09_CMX_Redux in ALL hyperparameters; only the YAML changes.
FusionRectify moved from neck (layers 21-23) to inside backbone (layers 11, 16, 23).
Rectified-RGB features flow through later backbone convs before reaching the head.

Saves to: runs/ablation_position_backbone/CMX_Backbone/
"""
from ultralytics import YOLO

MODEL_CFG = 'configs/yolo11_dual_cmx_backbone.yaml'
PRETRAIN  = 'yolo_dual_modulation_pretrain.pt'
DATA_YAML = 'dataset_melon.yaml'

model = YOLO(MODEL_CFG)
model.load(PRETRAIN)   # backbone weights matched by name/shape; FusionRectify random-init

model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=1280,
    batch=4,
    device=0,
    workers=8,
    project='runs/ablation_position_backbone',
    name='CMX_Backbone',
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
