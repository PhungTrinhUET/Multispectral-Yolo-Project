"""
Diagnostic script for AgriFusion (Part A) and YOLOv5s (Part B).
No source edits. Measure/inspect only.
"""
import os, sys, glob, torch, numpy as np
os.chdir('/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')
sys.path.insert(0, '/home/tower2080/Documents/DuaLuoi/Multispectral_YOLO_Project/ultralytics')

DATA_4CH  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/data.yaml'
DATA_3CH  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/data_3ch.yaml'
AFF_CKPT  = 'Multispectral_DuaLuoi_Project/Exp16_AgriFusion_Fixed/weights/best.pt'
V5_CKPT   = '/home/tower2080/Documents/DuaLuoi/yolov5/runs_train/yolov5s_1280_balanced_b4/weights/best.pt'
SURGERY   = 'yolo_dual_aff_best_fixed.pt'  # weight surgery source
TEST_DIR  = '/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/test'

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

SEP = '='*60

print(f"\n{SEP}")
print("PART A — AgriFusion Diagnostic")
print(SEP)

# ─── A1: VAL vs TEST back-to-back ────────────────────────────────────────────
print("\n[A1] VAL and TEST back-to-back — same harness, same settings")
print(f"     Checkpoint: {AFF_CKPT}")

aff = YOLO(AFF_CKPT)
params_m = sum(p.numel() for p in aff.model.parameters()) / 1e6
print(f"     Params: {params_m:.2f}M")

common_val_kwargs = dict(
    data=DATA_4CH, imgsz=1280, batch=4,
    half=False, device='0', amp=False,
    save_json=False, plots=False, verbose=False,
    conf=0.001, iou=0.7,   # use training iou=0.7 to match training eval
)

r_val = aff.val(split='val', **common_val_kwargs)
n_val = len(glob.glob(os.path.join('/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/val', '*')))
print(f"\n  VAL  ({n_val} images): mAP50={r_val.box.map50:.4f}  mAP50:95={r_val.box.map:.4f}"
      f"  P={r_val.box.mp:.4f}  R={r_val.box.mr:.4f}")
print(f"     Per-class mAP50: {[round(x,4) for x in r_val.box.maps.tolist()]}")

r_test = aff.val(split='test', **common_val_kwargs)
n_test = len(glob.glob(os.path.join(TEST_DIR, '*')))
print(f"\n  TEST ({n_test} images): mAP50={r_test.box.map50:.4f}  mAP50:95={r_test.box.map:.4f}"
      f"  P={r_test.box.mp:.4f}  R={r_test.box.mr:.4f}")
print(f"     Per-class mAP50: {[round(x,4) for x in r_test.box.maps.tolist()]}")

gap = r_val.box.map50 - r_test.box.map50
print(f"\n  GAP val-test: {gap:+.4f}")

# ─── A2: Input inspection ────────────────────────────────────────────────────
print(f"\n[A2] Input channel / first-layer inspection")

# First layer of backbone (after InputContainer = L0; L1 is first real conv)
m0 = list(aff.model.model)[0]
m1 = list(aff.model.model)[1]
print(f"  Layer 0 type: {type(m0).__name__}")
print(f"  Layer 1 type: {type(m1).__name__}")
if hasattr(m1, 'conv'):
    w = m1.conv.weight
    print(f"  Layer 1 conv.weight shape: {list(w.shape)}  (out_ch, IN_CH, kH, kW)")
    print(f"  => Layer 1 expects {w.shape[1]} input channels")

# NDVI branch (Layer 11)
m11 = list(aff.model.model)[11]
if hasattr(m11, 'conv'):
    w11 = m11.conv.weight
    print(f"  Layer 11 conv.weight shape: {list(w11.shape)}")
    print(f"  => Layer 11 expects {w11.shape[1]} input channel(s)")

# Dataloader output tensor check
print(f"\n  Checking dataloader output shape for 4ch data yaml...")
import yaml as _yaml
with open(DATA_4CH) as f:
    d = _yaml.safe_load(f)
print(f"  data.yaml ch={d.get('ch','ABSENT')}  channels={d.get('channels','ABSENT')}")

# Load one test batch to confirm shape
from ultralytics.data import build_dataloader
from ultralytics.data.dataset import YOLODataset
import torchvision
try:
    ds = YOLODataset(
        img_path='/home/tower2080/Documents/DuaLuoi/yolov5/data_compact/dataset_raw/images/test',
        imgsz=640, batch_size=1,
        augment=False, hyp=None, rect=True,
        cache=False, single_cls=False, stride=32, pad=0.5,
        data=d
    )
    sample = ds[0]
    img_tensor = sample['img']
    print(f"  Dataloader sample img shape: {list(img_tensor.shape)}  dtype={img_tensor.dtype}")
    print(f"  Value range: min={img_tensor.min():.3f}  max={img_tensor.max():.3f}")
    print(f"  Per-channel stats (C0..C3): "
          + "  ".join([f"C{i}=[{img_tensor[i].min():.2f},{img_tensor[i].max():.2f}]"
                       for i in range(img_tensor.shape[0])]))
except Exception as e:
    print(f"  YOLODataset check error: {e}")

# ─── A3: NIR-influence sanity check ─────────────────────────────────────────
print(f"\n[A3] NIR-influence on 10 test images (zero vs random NDVI)")

import cv2, random
test_imgs = sorted(glob.glob(os.path.join(TEST_DIR, '*')))[:10]
aff_model = aff.model.eval()
device = next(aff_model.parameters()).device

diffs_zero, diffs_rand = [], []
for img_path in test_imgs:
    bgr = cv2.imread(img_path)
    if bgr is None:
        continue
    # Build a 4ch tensor in same way as dataloader: [R,G,B,NDVI(=alpha or computed)]
    # Load as BGR, convert to RGB, compute NDVI approx from channel 0 (R) and channel 2 (NIR stored in 4th)
    # Actually just test by loading the image and creating a 4ch tensor
    # Since images are 3ch JPEG, synthesize a 4ch input
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # Resize to 640
    h, w = rgb.shape[:2]
    rgb = cv2.resize(rgb, (640, 640))
    # Normalize to [0,1] (same as ultralytics)
    rgb_t = torch.from_numpy(rgb).permute(2,0,1).float() / 255.0   # [3,H,W]
    # Create a synthetic NDVI channel (same as the dataloader would for these JPEG images)
    # Approx: NDVI = (NIR-R)/(NIR+R); here we just use the raw channel avg as proxy
    ndvi = rgb_t.mean(0, keepdim=True)  # [1,H,W]
    x_normal = torch.cat([rgb_t, ndvi], dim=0).unsqueeze(0).to(device)  # [1,4,H,W]

    with torch.no_grad():
        out_normal = aff_model(x_normal)[0]

        x_zero = x_normal.clone(); x_zero[0, 3] = 0.0
        out_zero = aff_model(x_zero)[0]

        x_rand = x_normal.clone(); x_rand[0, 3] = torch.rand_like(x_rand[0, 3])
        out_rand = aff_model(x_rand)[0]

    diffs_zero.append((out_normal - out_zero).abs().mean().item())
    diffs_rand.append((out_normal - out_rand).abs().mean().item())

print(f"  Mean |out_normal - out_zero_NIR|: {np.mean(diffs_zero):.4f}")
print(f"  Mean |out_normal - out_rand_NIR|: {np.mean(diffs_rand):.4f}")
print(f"  NIR-INFLUENCES-OUTPUT: {'YES' if np.mean(diffs_zero) > 0.01 else 'NO'}")

# ─── A4: Weight surgery check ─────────────────────────────────────────────────
print(f"\n[A4] Weight surgery check — Detect head vs fresh init")

if os.path.exists(SURGERY):
    surg = torch.load(SURGERY, map_location='cpu', weights_only=False)
    surg_m = surg.get('model', surg)
    print(f"  Surgery source ({SURGERY}) loaded: {type(surg_m).__name__}")
    # Count nc in surgery
    surg_layers = list(surg_m.model)
    surg_last = surg_layers[-1]
    print(f"  Surgery last layer: {type(surg_last).__name__}  nc={getattr(surg_last,'nc',None)}")
else:
    print(f"  Surgery source NOT FOUND: {SURGERY}")
    surg_m = None

# Compare Exp16 best.pt vs fresh init from yaml
best_ckpt = torch.load(AFF_CKPT, map_location='cpu', weights_only=False)
best_m = best_ckpt.get('model', best_ckpt)

# Get fresh model from yaml
from ultralytics.nn.tasks import DetectionModel
fresh_m = DetectionModel('configs/yolo11_dual_aff.yaml', ch=4, nc=4, verbose=False)

# Compare module by module
best_sd  = best_m.state_dict()
fresh_sd = fresh_m.state_dict()
if surg_m is not None:
    surg_sd  = surg_m.state_dict()

modules_to_check = {
    'Detect head': [k for k in best_sd if 'model.40.' in k or k.startswith('model.40.')],
    'RGB backbone (L1)': [k for k in best_sd if k.startswith('model.model.1.')],
    'NDVI backbone (L11)': [k for k in best_sd if k.startswith('model.model.11.')],
    'FusionAFF (L22)': [k for k in best_sd if k.startswith('model.model.22.')],
}

print(f"\n  Per-module weight delta from fresh random init:")
print(f"  {'Module':<22} {'|best - fresh|':>18}  {'keys':>5}  Notes")
print(f"  {'-'*65}")

for mod_name, keys in modules_to_check.items():
    if not keys:
        print(f"  {mod_name:<22}  (no keys found)")
        continue
    deltas = []
    for k in keys:
        if k in best_sd and k in fresh_sd:
            b = best_sd[k].float()
            f = fresh_sd[k].float()
            if b.shape == f.shape:
                deltas.append((b - f).abs().mean().item())
    if deltas:
        mean_d = np.mean(deltas)
        note = 'TRAINED' if mean_d > 0.05 else 'NEAR-INIT (suspicious)'
        print(f"  {mod_name:<22}  {mean_d:>18.4f}  {len(keys):>5}  {note}")
    else:
        print(f"  {mod_name:<22}  (shape mismatch or no overlap)")

# Also compare Detect head vs surgery source
if surg_m is not None:
    det_keys_best = [k for k in best_sd if 'model.40.' in k]
    det_keys_surg = [k.replace('model.model.', 'model.') for k in surg_sd if 'model.40.' in k]
    print(f"\n  Surgery source last layer: {type(surg_last).__name__}  nc={getattr(surg_last,'nc',None)}")
    # Try direct key comparison
    best_det_flat = [best_sd[k] for k in det_keys_best if k in best_sd]
    print(f"  Detect keys in best.pt: {len(det_keys_best)}")
    print(f"  Detect keys in surgery: {len([k for k in surg_sd if '40' in k])}")

# ─── A5 helper: per-class breakdown ─────────────────────────────────────────
print(f"\n[A5 helper] Per-class mAP50 comparison (classes: truoc, sau, cay40, tao_ngot)")
classes = ['truoc_thu_phan','sau_thu_phan','cay40','tao_ngot']
print(f"  {'Class':<22} {'VAL':>8} {'TEST':>8} {'DIFF':>8}")
print(f"  {'-'*50}")
for i, cls in enumerate(classes):
    v = r_val.box.maps[i].item()
    t = r_test.box.maps[i].item()
    print(f"  {cls:<22} {v:>8.4f} {t:>8.4f} {t-v:>+8.4f}")

print(f"\n{SEP}")
print("PART B — YOLOv5s baseline")
print(SEP)

# ─── B1: Try loading in ultralytics ──────────────────────────────────────────
print(f"\n[B1] Attempting to load YOLOv5s in ultralytics harness...")
try:
    v5 = YOLO(V5_CKPT)
    print(f"  LOADED OK — running TEST-383...")
    r_v5 = v5.val(data=DATA_3CH, split='test', imgsz=1280, batch=4,
                  half=False, device='0', amp=False, conf=0.001, iou=0.6,
                  save_json=False, plots=False, verbose=False)
    g640  = get_flops(v5.model, imgsz=640)
    g1280 = get_flops(v5.model, imgsz=1280)
    params_v5 = sum(p.numel() for p in v5.model.parameters()) / 1e6
    print(f"\n  YOLOv5s (ultralytics): params={params_v5:.2f}M  GFLOPs@640={g640:.1f}  @1280={g1280:.1f}")
    print(f"  mAP50={r_v5.box.map50:.4f}  mAP50:95={r_v5.box.map:.4f}"
          f"  P={r_v5.box.mp:.4f}  R={r_v5.box.mr:.4f}")
    print(f"  STATUS: VERIFIED from ultralytics harness")
except Exception as e:
    print(f"  LOAD FAILED: {type(e).__name__}: {e}")
    print(f"\n[B2] Falling back to native yolov5 val.py numbers...")
    # Inspect the checkpoint to get architecture info
    try:
        ckpt_v5 = torch.load(V5_CKPT, map_location='cpu', weights_only=False)
        m_v5    = ckpt_v5.get('model', ckpt_v5)
        print(f"  Checkpoint type: {type(m_v5).__name__}")
        # Try to get param count from checkpoint
        if hasattr(m_v5, 'parameters'):
            n_p = sum(p.numel() for p in m_v5.parameters()) / 1e6
            print(f"  Params from checkpoint: {n_p:.2f}M")
        # Get model stride / info
        if hasattr(m_v5, 'stride'):
            print(f"  Stride: {m_v5.stride}")
        if hasattr(m_v5, 'nc'):
            print(f"  nc: {m_v5.nc}")
        # First layer weight
        try:
            first = list(m_v5.model)[0] if hasattr(m_v5, 'model') else None
            if first and hasattr(first, 'conv'):
                print(f"  First layer conv.weight shape: {list(first.conv.weight.shape)}")
        except:
            pass
    except Exception as e2:
        print(f"  Checkpoint inspect error: {e2}")

    print(f"\n  Native yolov5 val.py defaults (from val.py source):")
    print(f"    conf_thres = 0.001  (line 193)")
    print(f"    iou_thres  = 0.6    (line 194, NMS IoU threshold)")
    print(f"    augment    = False")
    print(f"    data       = data_balanced.yaml  (test: dataset/images/test = 383 images)")
    print(f"    imgsz      = 1280 (from opt.yaml)")
    print(f"    batch_size = 4    (from opt.yaml)")
    print(f"\n  Previously measured from native run:")
    print(f"    mAP50  ≈ 0.505")
    print(f"    params ≈ 7.02M")
    print(f"    GFLOPs@640 = 15.8  (from yolov5 model summary)")
    print(f"    GFLOPs@1280 = UNVERIFIED (native harness did not produce this; do not hand-scale)")
    print(f"\n  Note: native yolov5 dataloader loads 4ch for this dataset (R+G+B+NDVI),")
    print(f"    then passes all 4ch to YOLOv5s which has 3ch first conv → only RGB used.")

print(f"\n{SEP}")
print("END OF DIAGNOSTIC")
print(SEP)
