"""
Sequential backbone-injection training — position ablation grid.
Order: Add → Concat → GAFF → AFF → ICAFusion (transformer last = OOM risk).
Each operator runs independently; one failure does NOT abort the chain.
ICAFusion fallback: pool_size=8 → pool_size=6 if OOM.
AFF: NOTE backbone (15.833M) differs from existing Exp17 neck (10.041M, different backbone arch).
     Training proceeds; delta is noted in the report.
"""
import sys, os, traceback, gc, torch
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_YAML = 'dataset_melon.yaml'
PRETRAIN  = 'yolo_dual_modulation_pretrain.pt'
PROJ      = 'runs/ablation_position_backbone'
CLS_NAMES = ['pre-pollination', 'post-pollination', 'net-formation', 'sweetening']

TRAIN_COMMON = dict(
    data=DATA_YAML, epochs=100, imgsz=1280, batch=4, device=0, workers=8,
    optimizer='auto', lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005,
    warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1,
    patience=100, seed=0, deterministic=True, amp=True, close_mosaic=10,
    mosaic=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5,
    fliplr=0.5, flipud=0.0, verbose=True,
)

OPERATORS = [
    # (yaml_cfg, run_name, notes)
    ('configs/yolo11_dual_add_backbone.yaml',    'Add_Backbone',  ''),
    ('configs/yolo11_dual_concat_backbone.yaml', 'Concat_Backbone',''),
    ('configs/yolo11_dual_gaff_backbone.yaml',   'GAFF_Backbone', ''),
    ('configs/yolo11_dual_aff_backbone.yaml',    'AFF_Backbone',
        'NOTE: backbone (CMX dims, 15.833M) differs from neck Exp17 (10.041M, diff arch)'),
    ('configs/yolo11_dual_ica_backbone.yaml',    'ICA_Backbone',  'pool_size=8'),
]
ICA_FALLBACK_YAML = 'configs/yolo11_dual_ica_backbone_p6.yaml'


def find_ckpt(run_name):
    from ultralytics.utils import RUNS_DIR
    ckpt = RUNS_DIR / 'detect' / PROJ / run_name / 'weights' / 'best.pt'
    if ckpt.exists(): return ckpt
    import glob
    hits = glob.glob(f'**/{run_name}/weights/best.pt', recursive=True)
    return Path(hits[0]) if hits else None


def run_eval(ckpt, run_name):
    from ultralytics import YOLO
    print(f"\n--- Canonical eval: {run_name} ---")
    ev = YOLO(str(ckpt))
    metrics = ev.val(
        data=DATA_YAML, split='test', imgsz=1280,
        batch=4, iou=0.6, conf=0.001, half=False, device=0, verbose=False,
    )
    mp, mr = metrics.box.mp, metrics.box.mr
    m50, m5095 = metrics.box.map50, metrics.box.map
    ap50 = metrics.box.ap50
    params = sum(p.numel() for p in ev.model.parameters()) / 1e6

    print(f"\n=== {run_name} ===")
    print(f"mAP@50={m50:.4f}  mAP@50:95={m5095:.4f}  P={mp:.4f}  R={mr:.4f}  Params={params:.3f}M")
    print(f"Per-class AP@50:")
    for cn, ap in zip(CLS_NAMES, ap50):
        print(f"  {cn:<22} {ap:.4f}")
    print(f"Checkpoint: {ckpt}")
    return m50


def train_one(cfg, name, notes='', extra_train_kwargs=None):
    from ultralytics import YOLO
    if notes:
        print(f"  [{name}] {notes}")
    model = YOLO(cfg)
    model.load(PRETRAIN)
    kwargs = {**TRAIN_COMMON, 'project': PROJ, 'name': name, 'exist_ok': False}
    if extra_train_kwargs:
        kwargs.update(extra_train_kwargs)
    model.train(**kwargs)
    del model; gc.collect(); torch.cuda.empty_cache()


print("=" * 65)
print("  Backbone Position Ablation — 5 Operators (sequential)")
print("=" * 65)

results = {}

for cfg, name, notes in OPERATORS:
    print(f"\n\n{'='*65}")
    print(f"  START: {name}")
    print(f"{'='*65}\n")

    if name == 'ICA_Backbone':
        # ICA: try pool=8 first, fallback to pool=6 on OOM
        pool_used = 8
        try:
            train_one(cfg, name, notes)
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n  [ICA OOM with pool=8] Retrying with pool_size=6 ...\n")
            torch.cuda.empty_cache()
            pool_used = 6
            try:
                train_one(ICA_FALLBACK_YAML, name + '_p6', 'pool_size=6 fallback')
                name = name + '_p6'   # update name for eval lookup
            except Exception as e2:
                print(f"  [ICA p6 FAILED] {e2}")
                traceback.print_exc()
                results[name] = 'FAILED (OOM even at pool=6)'
                continue
        except Exception as e:
            print(f"  [{name} FAILED] {e}")
            traceback.print_exc()
            results[name] = f'FAILED: {e}'
            continue
        results[name] = f'pool_size={pool_used}'
    else:
        try:
            train_one(cfg, name, notes)
            results[name] = 'OK'
        except Exception as e:
            print(f"  [{name} FAILED] {e}")
            traceback.print_exc()
            results[name] = f'FAILED: {e}'
            continue

    # Eval immediately
    ckpt = find_ckpt(name)
    if ckpt:
        try:
            map50 = run_eval(ckpt, name)
            results[name] += f'  mAP@50={map50:.4f}'
        except Exception as e:
            print(f"  [Eval failed for {name}] {e}")
    else:
        print(f"  WARNING: best.pt not found for {name}")

print("\n\n" + "="*65)
print("  FINAL SUMMARY")
print("="*65)
for name, status in results.items():
    print(f"  {name:<22} {status}")
