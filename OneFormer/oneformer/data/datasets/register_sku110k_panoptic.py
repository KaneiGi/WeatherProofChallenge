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
from detectron2.data.datasets import register_coco_panoptic


_PREDEFINED_SPLITS_COCO = {
    "sku110k_2017_train_panoptic": ("sku110k/panoptic_train2017", "sku110k/annotations/panoptic_train2017.json"),
    "sku110k_100_2017_train_panoptic": ("sku110k/panoptic_train2017", "sku110k/annotations/panoptic_100_train2017.json"),
    "sku110k_1k_2017_train_panoptic": ("sku110k/panoptic_train2017", "sku110k/annotations/panoptic_1k_train2017.json"),
    "sku110k_2017_val_panoptic": ("sku110k/panoptic_val2017", "sku110k/annotations/panoptic_val2017.json"),
}


SKU110K_CATEGORIES = [
    {"supercategory": "item", "color": [220, 100, 100], "isthing": 1, "id": 1, "name": "item"},
    {"supercategory": "ceiling", "color": [146, 139, 141], "isthing": 0, "id": 2, "name": "ceiling-merged"},
    {"supercategory": "appliance", "color": [59, 105, 106], "isthing": 0, "id": 3, "name": "refrigerator"},
    {"supercategory": "wall", "color": [137, 54, 74], "isthing": 0, "id": 4, "name": "wall-brick"},
    {"supercategory": "wall", "color": [135, 158, 223], "isthing": 0, "id": 5, "name": "wall-stone"},
    {"supercategory": "wall", "color": [7, 246, 231], "isthing": 0, "id": 6, "name": "wall-tile"},
    {"supercategory": "wall", "color": [107, 255, 200], "isthing": 0, "id": 7, "name": "wall-wood"},
    {"supercategory": "wall", "color": [102, 102, 156], "isthing": 0, "id": 8, "name": "wall-other-merged"},
    {"supercategory": "floor","color": [218, 88, 184], "isthing": 0, "id": 9, "name": "floor-wood"},
    {"supercategory": "floor", "color": [96, 36, 108], "isthing": 0, "id": 10, "name": "floor-other-merged"},
    {"supercategory": "furniture-stuff", "color": [255, 160, 98], "isthing": 0, "id": 11, "name": "shelf"},
    {"supercategory": "other", "color": [96, 96, 96], "isthing": 0, "id": 12, "name": "background-other-merged"}
]


def get_metadata():
    meta = {}
    # The following metadata maps contiguous id from [0, #thing categories +
    # #stuff categories) to their names and colors. We have to replica of the
    # same name and color under "thing_*" and "stuff_*" because the current
    # visualization function in D2 handles thing and class classes differently
    # due to some heuristic used in Panoptic FPN. We keep the same naming to
    # enable reusing existing visualization functions.
    thing_classes = [k["name"] for k in SKU110K_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SKU110K_CATEGORIES if k["isthing"] == 1]
    stuff_classes = [k["name"] for k in SKU110K_CATEGORIES]
    stuff_colors = [k["color"] for k in SKU110K_CATEGORIES]

    meta["thing_classes"] = thing_classes
    meta["thing_colors"] = thing_colors
    meta["stuff_classes"] = stuff_classes
    meta["stuff_colors"] = stuff_colors

    # Convert category id for training:
    #   category id: like semantic segmentation, it is the class id for each
    #   pixel. Since there are some classes not used in evaluation, the category
    #   id is not always contiguous and thus we have two set of category ids:
    #       - original category id: category id in the original dataset, mainly
    #           used for evaluation.
    #       - contiguous category id: [0, #classes), in order to train the linear
    #           softmax classifier.
    thing_dataset_id_to_contiguous_id = {}
    stuff_dataset_id_to_contiguous_id = {}

    for i, cat in enumerate(SKU110K_CATEGORIES):
        if cat["isthing"]:
            thing_dataset_id_to_contiguous_id[cat["id"]] = i
        # else:
        #     stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        # in order to use sem_seg evaluator
        stuff_dataset_id_to_contiguous_id[cat["id"]] = i

    meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
    meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id

    return meta



def register_panoptic_sku110k(root):
    for key, (panoptic_root, json_file) in _PREDEFINED_SPLITS_COCO.items():
        image_root = panoptic_root.replace('panoptic_', '')
        instance_json_file = json_file.replace('panoptic_', 'instances_')
        if 'val' in json_file:
            instance_json_file = json_file.replace('panoptic_', 'panoptic2instances_')
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_panoptic(
            key,
            get_metadata(),
            os.path.join(root, image_root),
            os.path.join(root, panoptic_root),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, instance_json_file)
        )


_root = os.path.expanduser(os.getenv("DETECTRON2_DATASETS", "datasets"))
register_panoptic_sku110k(_root)
