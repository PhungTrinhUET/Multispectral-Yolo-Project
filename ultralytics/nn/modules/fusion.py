# Compatibility shim — re-exports fusion classes from block.py
# Required to load models saved from the yolo11s_dual_V2 ultralytics fork
from .block import (
    FusionAFF,
    FusionRectify,
    FusionDeformRectify,
    FusionCrossCBAM,
    FusionRectifyFeedback,
    FusionModulation,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


# FusionAdd: adaptive add with optional learnable adapter + spatial alignment
# Supports both the simple version (no adapter) and the version with adapter+interpolation
# saved in the yolo11s_dual_V2 checkpoints.
class FusionAdd(nn.Module):
    def __init__(self, ch_main, ch_nir):
        super().__init__()
        if ch_nir != ch_main:
            self.adapter = Conv(ch_nir, ch_main, 1, 1)

    def forward(self, x, x_custom=None):
        if isinstance(x, (list, tuple)):
            x_main, x_aux = x[0], x[1]
        else:
            x_main, x_aux = x, x_custom
        if x_main.shape[2:] != x_aux.shape[2:]:
            x_aux = F.interpolate(x_aux, size=x_main.shape[2:], mode='bilinear', align_corners=False)
        if hasattr(self, 'adapter'):
            x_aux = self.adapter(x_aux)
        return x_main + x_aux


# FusionConcat: concat + 1x1 Conv (or C3k2 blend) to reduce back to ch_main
# Supports checkpoints saved with either join or blend submodule.
class FusionConcat(nn.Module):
    def __init__(self, ch_main, ch_nir):
        super().__init__()
        self.join = Conv(ch_main + ch_nir, ch_main, 1, 1)

    def forward(self, x, x_custom=None):
        if isinstance(x, (list, tuple)):
            x_main, x_aux = x[0], x[1]
        else:
            x_main, x_aux = x, x_custom
        if x_main.shape[2:] != x_aux.shape[2:]:
            x_aux = F.interpolate(x_aux, size=x_main.shape[2:], mode='bilinear', align_corners=False)
        cat = torch.cat([x_main, x_aux], dim=1)
        if hasattr(self, 'blend'):
            return self.blend(cat)
        return self.join(cat)


class GlobalAttention(nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x): return x

class LocalAttention(nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x): return x

class DualFeatureFusion(nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x): return x[0] + x[1] if isinstance(x, (list, tuple)) else x

class TinyBackbone(nn.Module):
    def __init__(self, *a, **kw): super().__init__()
    def forward(self, x): return x
