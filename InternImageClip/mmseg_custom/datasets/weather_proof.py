# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import random

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class WeatherProofDataset(CustomDataset):
    """Weather Proof dataset.
    """

    # CLASSES = ('tree', 'structure', 'stone', 'terrain-vegetation', 'building',
    #            'terrain-other', 'road', 'sky', 'terrain-snow',)
    CLASSES = ('building', 'structure', 'road', 'sky', 'stone',
               'terrain-vegetation', 'terrain-other', 'terrain-snow', 'tree',)

    
    PALETTE = [[36, 179, 83], [250, 250, 55], [112, 128, 0], [51, 204, 179],
               [8, 100, 192], [251, 0, 42], [202, 153, 58], [49, 179, 245],
               [245, 0, 196],]

    def __init__(self, split,img_suffix='_degraded.png',seg_map_suffix = '_gt-intern.png' ,**kwargs):
        super(WeatherProofDataset, self).__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            split=split,
            reduce_zero_label=True,
            **kwargs)
    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """
        while True:
            random_idx = random.choices(range(len(self.img_infos)), k=1)[0]
            if random_idx != idx:
                break

        random_img_info = self.img_infos[random_idx]
        random_ann_info = self.get_ann_info(random_idx)

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)


        results = dict(img_info=img_info, ann_info=ann_info,random_img_info=random_img_info,random_ann_info=random_ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)


