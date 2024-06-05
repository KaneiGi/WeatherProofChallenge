import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.file_io import PathManager


WEATHERPROOF_SPLITS = {
    "weatherproof_sem_seg_train": ("WeatherProofDataset", "WeatherProofDataset"),
    "weatherproof_sem_seg_train_clean": ("WeatherProofDatasetClean", "WeatherProofDatasetClean"),
    "weatherproof_sem_seg_test": ("WeatherProofTest/images", "WeatherProofTest/annotations"),
    "weatherproof_sem_seg_train_extra": ("WeatherProofExtra/images", "WeatherProofExtra/annotations")
}


WEATHERPROOF_CATEGORIES = [
    # {"supercategory": "background", "color": [10, 10, 10], "isthing": 0, "id": 0, "name": "background"},
    {"supercategory": "building", "color": [8, 100, 192], "isthing": 0, "id": 1, "name": "building"},
    {"supercategory": "structure", "color": [250, 250, 55], "isthing": 0, "id": 2, "name": "structure"},
    {"supercategory": "road", "color": [202, 153, 58], "isthing": 0, "id": 3, "name": "road"},
    {"supercategory": "sky", "color": [49, 179, 245], "isthing": 0, "id": 4, "name": "sky"},
    {"supercategory": "stone", "color": [112, 128, 0], "isthing": 0, "id": 5, "name": "stone"},
    {"supercategory": "terrain", "color": [51, 204, 179], "isthing": 0, "id": 6, "name": "vegetation"},
    {"supercategory": "terrain", "color": [251, 0, 42], "isthing": 0, "id": 7, "name": "other"},
    {"supercategory": "terrain", "color": [245, 0, 196], "isthing": 0, "id": 8, "name": "snow"},
    {"supercategory": "tree","color": [36, 179, 83], "isthing": 0, "id": 9, "name": "tree"},
]


NAMES = []
NUMS = []
MAX_NUM = 30
for i in range(300):
    if i % 10 == 0:
        NAMES.append("%06d.png"%i)
        NUMS.append(i)

def _get_weatherproof_files(image_dir):
    files = []
    for root, dirs, names in os.walk(image_dir):
        for name in names:
            # for WeatherProofDataset
            if name.endswith("degraded.png"):
                if int(name[:3]) not in NUMS:
                    continue
                image_file = os.path.join(root, name)
                label_name = name.replace("degraded.png", "gt-intern-reduce.png")
                label_file = os.path.join(root, label_name)
                files.append((image_file, label_file))
                continue
            # for for WeatherProofTest
            if name.endswith("000000.png") and len(name) > 10:
                image_file = os.path.join(root, name)
                label_file = os.path.join(root.replace("images", "annotations-reduce"), name)
                files.append((image_file, label_file))
                continue
            # for for WeatherProofExtra
            if name in NAMES:
                image_file = os.path.join(root, name)
                label_file = root.replace("images", "annotations-reduce") + ".png"
                files.append((image_file, label_file))
                continue
            # for WeatherProofDatasetClean
            if name == "real.png":
                image_file = os.path.join(root, name)
                label_file = os.path.join(root.replace("WeatherProofDatasetClean", "WeatherProofDataset"), "000_gt-intern-reduce.png")
                files.append((image_file, label_file))
                continue
    assert len(files), "No images found in {}".format(image_dir)
    print(f"Found {len(files)} samples in {image_dir}.")
    for f in files[0]:
        assert PathManager.isfile(f), f
    return files


def load_weatherproof_semantic(image_dir):
    ret = []
    for image_file, label_file in _get_weatherproof_files(image_dir):
        ret.append(
            {
                "file_name": image_file,
                "sem_seg_file_name": label_file,
            }
        )
    assert len(ret), f"No images found in {image_dir}!"
    return ret    


def register_all_weatherproof(root):
    for key, (image_dir, gt_dir) in WEATHERPROOF_SPLITS.items():
        meta = {}
        
        thing_classes = [k["name"] for k in WEATHERPROOF_CATEGORIES if k["isthing"] == 1]
        thing_colors = [k["color"] for k in WEATHERPROOF_CATEGORIES if k["isthing"] == 1]
        stuff_classes = [k["name"] for k in WEATHERPROOF_CATEGORIES]
        stuff_colors = [k["color"] for k in WEATHERPROOF_CATEGORIES]
    
        meta["thing_classes"] = thing_classes
        meta["thing_colors"] = thing_colors
        meta["stuff_classes"] = stuff_classes
        meta["stuff_colors"] = stuff_colors
        
        thing_dataset_id_to_contiguous_id = {}
        stuff_dataset_id_to_contiguous_id = {}

        for i, cat in enumerate(WEATHERPROOF_CATEGORIES):
            if cat["isthing"]:
                thing_dataset_id_to_contiguous_id[cat["id"]] = i
            stuff_dataset_id_to_contiguous_id[cat["id"]] = i

        meta["thing_dataset_id_to_contiguous_id"] = thing_dataset_id_to_contiguous_id
        meta["stuff_dataset_id_to_contiguous_id"] = stuff_dataset_id_to_contiguous_id
        # print(meta)
        image_dir = os.path.join(root, image_dir)
        gt_dir = os.path.join(root, gt_dir)
        
        DatasetCatalog.register(
            key, lambda x=image_dir: load_weatherproof_semantic(x)
        )
        MetadataCatalog.get(key).set(
            image_dir=image_dir,
            gt_dir=gt_dir,
            evaluator_type="sem_seg",
            ignore_label=255,
            **meta,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_weatherproof(_root)

