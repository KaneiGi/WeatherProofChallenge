# dataset settings
# __base__ = ['./weatherproof_extra_data.py']
dataset_type = 'WeatherProofDataset'
data_root = '/data0/weijn/WeatherProofDataset'
deweather_data_root = '/data0/weijn/WeatherProofDatasetDw'
ann_data_root = '/data0/weijn/WeatherProofDataset'
clean_data_root = '/data0/weijn/WeatherProofDatasetClean'

img_norm_cfg = dict(
    # mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    mean=[122.77, 116.75, 104.10], std=[68.50, 66.63, 70.323], to_rgb=True)
crop_size = (640, 640)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1280, 640),ratio_range=(0.5,2.0)),
    # dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(640,640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img','gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 640),
        # img_scale=crop_size,
        # img_ratios=[1.0, 1.25, 1.5, 1.75],
        img_ratios=[0.1,0.2,0.3,0.4,0.5, 0.6,0.7,0.8,0.9, 1.0, 1.1, 1.2,1.3,1.4,1.5,],
        # img_ratios=[0.5, 0.6,0.7,0.8,0.9, 1.0, 1.1, 1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0],
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(1280, 640), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline1 = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 640),
        # img_scale=crop_size,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(1280, 640), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
weather_proof_adverse_train = dict(
    type=dataset_type,
    img_dir=data_root + '/train_scenes',
    ann_dir = ann_data_root + '/train_scenes',
    split=data_root + '/train_scenes/split_train.txt',
    # split=data_root + '/train_scenes/split_simple.txt',
    pipeline=train_pipeline
)
weather_proof_adverse_train_step_10 = dict(
    type=dataset_type,
    img_dir=data_root + '/train_scenes',
    ann_dir = ann_data_root + '/train_scenes',
    split=data_root + '/train_scenes/split_step_10.txt',
    # split=data_root + '/train_scenes/split_simple.txt',
    pipeline=train_pipeline
)
weather_proof_adverse_test = dict(
    type=dataset_type,
    img_dir=data_root + '/test_scenes',
    ann_dir = ann_data_root + '/test_scenes',
    split=data_root + '/test_scenes/split_simple.txt',
    # split=data_root + '/test_scenes/split_test.txt',
    pipeline=test_pipeline
)
weather_proof_adverse_val = dict(
    type=dataset_type,
    img_dir=data_root + '/test_scenes',
    ann_dir = ann_data_root + '/test_scenes',
    # split=data_root + '/test_scenes/split_simple.txt',
    split=data_root + '/test_scenes/split_test.txt',
    pipeline=train_pipeline
)
weather_proof_adverse_val_step_10 = dict(
    type=dataset_type,
    img_dir=data_root + '/test_scenes',
    ann_dir = ann_data_root + '/test_scenes',
    # split=data_root + '/test_scenes/split_simple.txt',
    split=data_root + '/test_scenes/split_step_10.txt',
    pipeline=train_pipeline
)
weather_proof_clean_train = dict(
    type=dataset_type,
    img_dir=clean_data_root + '/train_scenes',
    ann_dir = clean_data_root + '/train_scenes',
    split = None,
    # split='/data0/weijn/WeatherProofDataset/train_scenes/split_train.txt',
    img_suffix='real.png',
    seg_map_suffix='gt_seg.png',
    pipeline=train_pipeline
)
weather_proof_clean_val = dict(
    type=dataset_type,
    img_dir=clean_data_root + '/test_scenes',
    ann_dir=clean_data_root + '/test_scenes',
    split=None,
    # split='/data0/weijn/WeatherProofDataset/train_scenes/split_train.txt',
    img_suffix='real.png',
    seg_map_suffix='gt_seg.png',
    pipeline=train_pipeline
)
weather_proof_deweather_train = dict(
    type=dataset_type,
    img_dir=deweather_data_root + '/train_scenes',
    ann_dir = data_root + '/train_scenes',
    split = deweather_data_root + '/train_scenes/split_train.txt',
    # split='/data0/weijn/WeatherProofDataset/train_scenes/split_train.txt',
    pipeline=train_pipeline
)
weather_proof_deweather_test = dict(
    type=dataset_type,
    img_dir=deweather_data_root + '/test_scenes',
    ann_dir = data_root + '/test_scenes',
    split = deweather_data_root + '/test_scenes/split_simple.txt',
    # split='/data0/weijn/WeatherProofDataset/train_scenes/split_train.txt',
    pipeline=test_pipeline
)
test20= dict(
    type=dataset_type,
    img_dir=data_root + '/test-20/images',
    ann_dir = data_root + '/test-20/annotations',
    split = None,
    # split=data_root + '/scenes/split.txt',
    img_suffix='.png',
    seg_map_suffix='.png',
    pipeline=test_pipeline)

scenes= dict(
    type=dataset_type,
    img_dir=data_root + '/scenes',
    ann_dir = [None],
    # split = None,
    split=[data_root + '/scenes/split_18.txt'],
    img_suffix='.png',
    seg_map_suffix=None,
    pipeline=test_pipeline)

weather_proof_extra_1 = dict(
    type=dataset_type,
    img_dir=data_root + '/extra_data/test_search/new_train-0514/imagesX2',
    ann_dir = ann_data_root + '/extra_data/test_search/new_train-0514/annotationsX2',
    split=None,
    img_suffix='.png',
    seg_map_suffix='.png',
    # split=data_root + '/test_scenes/split_test.txt',
    pipeline=train_pipeline
)
weather_proof_extra_2 = dict(
    type=dataset_type,
    img_dir='/data0/weijn/WeatherProofDataset/extra_data/val_based/images',
    ann_dir = '/data0/weijn/WeatherProofDataset/extra_data/val_based/annotations',
    split=None,
    img_suffix='.jpg',
    seg_map_suffix='.png',
    # split=data_root + '/test_scenes/split_test.txt',
    pipeline=train_pipeline
)
weather_proof_extra_3 = dict(
    type=dataset_type,
    img_dir='/data0/weijn/WeatherProofDataset/extra_data/test_search/new_train_0517/images',
    ann_dir = '/data0/weijn/WeatherProofDataset/extra_data/test_search/new_train_0517/annotations',
    split=None,
    img_suffix='.png',
    seg_map_suffix='.png',
    # split=data_root + '/test_scenes/split_test.txt',
    pipeline=train_pipeline
)
aug_step_10 = dict(
    type='AugDataset',
    img_dir=data_root + '/extra_data/aug/images',
    ann_dir = ann_data_root + '/extra_data/aug/annotations',
    # split=data_root + '/test_scenes/split_simple.txt',
    split=data_root + '/extra_data/aug/split_step_10.txt',
    img_suffix='',
    seg_map_suffix='.png',
    pipeline=train_pipeline
)
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=[weather_proof_extra_3,aug_step_10,weather_proof_adverse_train_step_10,weather_proof_adverse_val_step_10,
           weather_proof_clean_train,weather_proof_clean_val],

    val=test20,
    test=test20)


