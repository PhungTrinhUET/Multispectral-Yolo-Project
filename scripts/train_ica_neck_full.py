"""
ICAFusion @ neck — full 100 epochs, NaN fix: amp=False + clip_grad=10.
"""
import sys, os, gc, traceback, torch
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics import YOLO
from ultralytics.utils import RUNS_DIR

DATA_YAML = 'dataset_melon.yaml'
PRETRAIN  = 'yolo_dual_modulation_pretrain.pt'
PROJ      = 'runs/ablation_neck'
CLS_NAMES = ['pre-pollination', 'post-pollination', 'net-formation', 'sweetening']

TRAIN_BASE = dict(
    data=DATA_YAML, epochs=100, imgsz=1280, batch=2, device=0, workers=8,
    optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    patience=100, seed=0, deterministic=True, close_mosaic=10,
    mosaic=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    scale=0.5, fliplr=0.5, flipud=0.0, verbose=True,
    amp=False,
)

print("=" * 65)
print("  ICAFusion @ neck — full 100 epochs (amp=False, clip=10)")
print("=" * 65)

try:
    model = YOLO('configs/yolo11_dual_icafusion.yaml')
    model.load(PRETRAIN)
    model.train(**TRAIN_BASE, project=PROJ, name='ICAFusion_full', exist_ok=False)
    del model; gc.collect(); torch.cuda.empty_cache()

    ckpt = RUNS_DIR / 'detect' / PROJ / 'ICAFusion_full' / 'weights' / 'best.pt'
    if not ckpt.exists():
        import glob
        hits = glob.glob('**/ICAFusion_full/weights/best.pt', recursive=True)
        ckpt = Path(hits[0]) if hits else None

    if ckpt:
        m = YOLO(str(ckpt))
        mx = m.val(data=DATA_YAML, split='test', imgsz=1280, batch=4,
                   iou=0.6, conf=0.001, half=False, device=0, verbose=False)
        params = sum(p.numel() for p in m.model.parameters()) / 1e6
        print(f"\n=== ICAFusion neck (full 100ep) ===")
        print(f"Params={params:.3f}M")
        print(f"mAP@50={mx.box.map50:.4f}  mAP@50:95={mx.box.map:.4f}  P={mx.box.mp:.4f}  R={mx.box.mr:.4f}")
        print("Per-class AP@50:")
        for cn, ap in zip(CLS_NAMES, mx.box.ap50):
            print(f"  {cn:<22} {ap:.4f}")
        print(f"Checkpoint: {ckpt}")
    else:
        print("WARNING: best.pt not found for ICAFusion_full")

except Exception as e:
    print(f"\n[ICAFusion_full FAILED] {e}")
    traceback.print_exc()

print("\nDone.")
