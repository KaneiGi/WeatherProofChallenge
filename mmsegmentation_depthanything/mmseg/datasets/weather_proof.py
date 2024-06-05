# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class WeatherProofDataset(BaseSegDataset):
    """Weather Proof dataset.
    """

    METAINFO = dict(
        classes=('building', 'structure', 'road', 'sky', 'stone', 'terrain-vegetation',
               'terrain-other', 'terrain-snow', 'tree'),
        palette=[[36, 179, 83], [250, 250, 55], [112, 128, 0], [51, 204, 179],
               [8, 100, 192], [251, 0, 42], [202, 153, 58], [49, 179, 245],
               [245, 0, 196]])


    def __init__(self,
                 img_suffix='_degraded.png',
                 seg_map_suffix='_gt-intern.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)


@DATASETS.register_module()
class WeatherProofCleanDataset(BaseSegDataset):
    """Weather Proof dataset.
    """

    METAINFO = dict(
        classes=('building', 'structure', 'road', 'sky', 'stone', 'terrain-vegetation',
               'terrain-other', 'terrain-snow', 'tree'),
        palette=[[36, 179, 83], [250, 250, 55], [112, 128, 0], [51, 204, 179],
               [8, 100, 192], [251, 0, 42], [202, 153, 58], [49, 179, 245],
               [245, 0, 196]])

    def __init__(self,
                 img_suffix='_real.png',
                 seg_map_suffix='_gt_seg.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class WeatherProofTestDataset(BaseSegDataset):
    """Weather Proof dataset.
    """

    METAINFO = dict(
        classes=('building', 'structure', 'road', 'sky', 'stone', 'terrain-vegetation',
               'terrain-other', 'terrain-snow', 'tree'),
        palette=[[36, 179, 83], [250, 250, 55], [112, 128, 0], [51, 204, 179],
               [8, 100, 192], [251, 0, 42], [202, 153, 58], [49, 179, 245],
               [245, 0, 196]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

@DATASETS.register_module()
class WeatherProofExtraDataset(BaseSegDataset):
    """Weather Proof dataset.
    """

    METAINFO = dict(
        classes=('building', 'structure', 'road', 'sky', 'stone', 'terrain-vegetation',
               'terrain-other', 'terrain-snow', 'tree'),
        palette=[[36, 179, 83], [250, 250, 55], [112, 128, 0], [51, 204, 179],
               [8, 100, 192], [251, 0, 42], [202, 153, 58], [49, 179, 245],
               [245, 0, 196]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)