# Copyright (c) OpenMMLab. All rights reserved.
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .uper_head_paired import UPerHeadPaired
from .uper_head_class_weighted import UPerHeadWiehted
__all__ = [
    'MaskFormerHead',
    'Mask2FormerHead',
    'UPerHeadPaired',
    'UPerHeadWiehted',
]
