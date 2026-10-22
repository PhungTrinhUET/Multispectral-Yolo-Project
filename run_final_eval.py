"""
Final Results Collection for paper tables.
Rules: No source edits. All on TEST-383, fp32, AMP OFF, same harness.
"""
import os, sys, glob
os.chdir('/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')
sys.path.insert(0, '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')

import torch

DATA_3CH = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/data_3ch.yaml'
DATA_4CH = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/data.yaml'
BASE = 'Multispectral_DuaLuoi_Project'

TEST_IMAGES = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/test'
n_test = len(glob.glob(os.path.join(TEST_IMAGES, '*')))
print(f"\n[CONFIRMED] Test images on disk: {n_test}")

MODELS = [
    # (label, weights_path, data_yaml)
    ('YOLOv8s',
     '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolov8s_1280_balanced_b4/weights/best.pt',
     DATA_3CH),
    ('YOLOv10s',
     '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolov10s_1280_balanced_b4/weights/best.pt',
     DATA_3CH),
    ('YOLOv11s',
     '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolo11s_1280_balanced_b4/weights/best.pt',
     DATA_3CH),
    ('EarlyFusion',
     f'{BASE}/Exp1_NoPretrain_1280px/weights/best.pt',
     DATA_4CH),
    ('AgriFusion',
     f'{BASE}/Exp16_AgriFusion_Fixed/weights/best.pt',
     DATA_4CH),
    ('GMD-YOLO',
     f'{BASE}/Exp09_CMX_Redux/weights/best.pt',
     DATA_4CH),
    ('Neck_Add2',
     f'{BASE}/Exp14_Neck_Add2/weights/best.pt',
     DATA_4CH),
    ('Neck_Concat',
     f'{BASE}/Exp15_Neck_Concat/weights/best.pt',
     DATA_4CH),
]

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

collected = {}

for label, pt, data in MODELS:
    print(f"\n{'='*60}")
    print(f"MODEL: {label}")
    print(f"  weights: {pt}")
    print(f"  data:    {data}")
    print('='*60)

    model = YOLO(pt)

    # Params
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"  params: {params_m:.2f}M")

    # GFLOPs at 640 and 1280
    try:
        g640  = get_flops(model.model, imgsz=640)
        g1280 = get_flops(model.model, imgsz=1280)
        print(f"  GFLOPs@640={g640:.1f}  @1280={g1280:.1f}")
    except Exception as e:
        g640 = g1280 = None
        print(f"  GFLOPs ERROR: {e}")

    # Validation on test split, fp32
    try:
        r = model.val(
            data=data,
            split='test',
            imgsz=1280,
            batch=4,
            half=False,
            device='0',
            amp=False,
            save_json=False,
            plots=False,
            verbose=False,
            conf=0.001,
            iou=0.6,
        )
        map50    = r.box.map50
        map5095  = r.box.map
        prec     = r.box.mp
        recall   = r.box.mr
        speed    = r.speed   # dict with 'preprocess','inference','postprocess'
        fps      = round(1000 / (speed['preprocess'] + speed['inference'] + speed['postprocess']), 1)
        print(f"  [RESULT] mAP50={map50:.4f} mAP50:95={map5095:.4f} P={prec:.4f} R={recall:.4f} FPS={fps}")
    except Exception as e:
        map50 = map5095 = prec = recall = fps = None
        print(f"  VAL ERROR: {e}")

    collected[label] = dict(
        params=params_m, g640=g640, g1280=g1280,
        map50=map50, map5095=map5095, prec=prec, recall=recall, fps=fps
    )

# ---- SUMMARY ----
print("\n\n" + "="*70)
print("FINAL SUMMARY (all TEST-383, fp32, AMP OFF, imgsz=1280, batch=4)")
print("="*70)
fmt = "{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}"
print(fmt.format("Model","Params(M)","GF@640","GF@1280","mAP50","mAP5095","Prec","Recall"))
print("-"*70)
for label, v in collected.items():
    def f(x, decimals=4): return f"{x:.{decimals}f}" if x is not None else "ERR"
    print(fmt.format(
        label,
        f(v['params'],2),
        f(v['g640'],1) if v['g640'] else "ERR",
        f(v['g1280'],1) if v['g1280'] else "ERR",
        f(v['map50']),
        f(v['map5095']),
        f(v['prec']),
        f(v['recall']),
    ))
