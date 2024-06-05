# ------------------------------------------------------------------------------
# Reference: https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/datasets/builtin.py
# Modified by Jitesh Jain (https://github.com/praeclarumjj3)
# ------------------------------------------------------------------------------


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os
# from detectron2.data.datasets.builtin_meta import  _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances


_PREDEFINED_SPLITS_COCO = {
    "sku110k_2017_train": ("sku110k/train2017", "sku110k/annotations/instances_train2017.json"),
    "sku110k_100_2017_train": ("sku110k/train2017", "sku110k/annotations/instances_100_train2017.json"),
    "sku110k_1k_2017_train": ("sku110k/train2017", "sku110k/annotations/instances_1k_train2017.json"),
    "sku110k_2017_val": ("sku110k/val2017", "sku110k/annotations/panoptic2instances_val2017.json"),
}


metadata = {
    'thing_dataset_id_to_contiguous_id': {1: 0},
    'thing_classes': ['item'],
    'thing_colors': [[220, 100, 100]]
}


def register_instances_sku110k(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            metadata,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_instances_sku110k(_root)
