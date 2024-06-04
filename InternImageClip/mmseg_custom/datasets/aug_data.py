# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import random
import mmcv
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from mmcv.utils import print_log
from mmseg.utils import get_root_logger

@DATASETS.register_module()
class AugDataset(CustomDataset):
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
        super(AugDataset, self).__init__(
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

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix,
                         split):
        """Load annotation from directory.

        Args:
            img_dir (str): Path to image directory
            img_suffix (str): Suffix of images.
            ann_dir (str|None): Path to annotation directory.
            seg_map_suffix (str|None): Suffix of segmentation maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None

        Returns:
            list[dict]: All image info of dataset.
        """

        img_infos = []
        if split is not None:
            lines = mmcv.list_from_file(
                split, file_client_args=self.file_client_args)
            for line in lines:
                img_name = line.strip()
                img_info = dict(filename=img_name + img_suffix)
                if ann_dir is not None:
                    seg_map = img_name[:-11] + seg_map_suffix
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
        else:
            for img in self.file_client.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=img_suffix,
                    recursive=True):
                img_info = dict(filename=img)
                if ann_dir is not None:
                    seg_map = img.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=seg_map)
                img_infos.append(img_info)
            img_infos = sorted(img_infos, key=lambda x: x['filename'])

        print_log(f'Loaded {len(img_infos)} images', logger=get_root_logger())
        return img_infos


