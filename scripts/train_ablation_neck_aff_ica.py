"""
Task ①: AFF @ neck — standard backbone (15.833M, same backbone as CMX/Add/Concat/GAFF).
Task ②: ICAFusion @ neck — full 100 epochs, NaN fixed with amp=False + clip_grad=10.

Run sequentially; failure in one does NOT abort the other.
Saves to: runs/ablation_neck/{AFF_stdbackbone, ICAFusion_full}/
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
    data=DATA_YAML, epochs=100, imgsz=1280, batch=4, device=0, workers=8,
    optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    patience=100, seed=0, deterministic=True, close_mosaic=10,
    mosaic=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    scale=0.5, fliplr=0.5, flipud=0.0, verbose=True,
)

RUNS = [
    dict(
        tag   = 'AFF neck (std backbone)',
        cfg   = 'configs/yolo11_dual_aff_neck_std.yaml',
        name  = 'AFF_stdbackbone',
        notes = 'Standard CMX backbone (15.833M), FusionAFF at neck P3/P4/P5',
        extra = dict(amp=True),   # AFF is CNN-based, AMP safe
    ),
    dict(
        tag   = 'ICAFusion neck (full 100ep)',
        cfg   = 'configs/yolo11_dual_icafusion.yaml',
        name  = 'ICAFusion_full',
        notes = 'NaN fix: amp=False + clip_grad=10.0 (fp16 overflow in MultiheadAttention)',
        extra = dict(amp=False, clip=10.0),   # amp=False prevents fp16 softmax overflow
    ),
]


def run_eval(ckpt, tag):
    m = YOLO(str(ckpt))
    mx = m.val(data=DATA_YAML, split='test', imgsz=1280, batch=4,
               iou=0.6, conf=0.001, half=False, device=0, verbose=False)
    params = sum(p.numel() for p in m.model.parameters()) / 1e6
    gf = m.model.info(imgsz=640, detailed=False)[1] / 1e9  # may return GFLOPs directly
    print(f"\n=== {tag} ===")
    print(f"best.pt: {ckpt}")
    print(f"Params(M)={params:.3f}  |  mAP@50={mx.box.map50:.4f}  mAP@50:95={mx.box.map:.4f}"
          f"  P={mx.box.mp:.4f}  R={mx.box.mr:.4f}")
    print("Per-class AP@50:")
    for cn, ap in zip(CLS_NAMES, mx.box.ap50):
        print(f"  {cn:<22} {ap:.4f}")


for run in RUNS:
    tag  = run['tag']
    cfg  = run['cfg']
    name = run['name']
    notes= run['notes']
    extra= run['extra']

    print(f"\n{'='*65}")
    print(f"  START: {tag}")
    print(f"  Notes: {notes}")
    print(f"{'='*65}\n")

    try:
        model = YOLO(cfg)
        model.load(PRETRAIN)

        kwargs = {**TRAIN_BASE, **extra, 'project': PROJ, 'name': name, 'exist_ok': False}
        model.train(**kwargs)
        del model; gc.collect(); torch.cuda.empty_cache()

        # Locate best.pt
        ckpt = RUNS_DIR / 'detect' / PROJ / name / 'weights' / 'best.pt'
        if not ckpt.exists():
            import glob
            hits = glob.glob(f'**/{name}/weights/best.pt', recursive=True)
            ckpt = Path(hits[0]) if hits else None

        if ckpt:
            run_eval(ckpt, tag)
        else:
            print(f"  WARNING: best.pt not found for {name}")

    except Exception as e:
        print(f"\n  [{tag}] FAILED: {e}")
        traceback.print_exc()
        gc.collect(); torch.cuda.empty_cache()

print("\n\nAll done.")
