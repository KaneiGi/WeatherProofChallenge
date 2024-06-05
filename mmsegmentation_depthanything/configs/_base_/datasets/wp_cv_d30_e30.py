# dataset settings
dataset_type = 'WeatherProofDataset'
clean_dataset_type = 'WeatherProofCleanDataset'
test_dataset_type = 'WeatherProofTestDataset'
extra_dataset_type = 'WeatherProofExtraDataset'

data_root = '/data0/zhangxd/dataset/segmentation/WeatherProofDataset'
clean_data_root = '/data0/zhangxd/dataset/segmentation/WeatherProofDatasetClean'
test_data_root = '/data0/zhangxd/dataset/segmentation/WeatherProofDataset'
extra_data_root = '/data0/zhangxd/dataset/segmentation/WeatherProofDatasetExtra'

crop_size = (448, 448)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(2560, 448),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2560, 896), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

## degraded for train
degraded_train = dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline)

degraded_train_30 = dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/training_30', seg_map_path='annotations/training_30'),
        pipeline=train_pipeline)

# test_scence for train
degraded_val_4train = dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation', seg_map_path='annotations/validation'),
        pipeline=train_pipeline)

# clean for train
clean_train = dict(
        type=clean_dataset_type,
        data_root=clean_data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline)

clean_val_4train = dict(
        type=clean_dataset_type,
        data_root=clean_data_root,
        data_prefix=dict(
            img_path='images/validation', seg_map_path='annotations/validation'),
        pipeline=train_pipeline)

# extra for train
extra_train = dict(
        type=extra_dataset_type,
        data_root=extra_data_root,
        data_prefix=dict(
            img_path='images/training', seg_map_path='annotations/training'),
        pipeline=train_pipeline)

extra_aug_train = dict(
        type=extra_dataset_type,
        data_root=extra_data_root,
        data_prefix=dict(
            img_path='images/aug', seg_map_path='annotations/aug'),
        pipeline=train_pipeline)


train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    # dataset=dict(type='ConcatDataset', datasets=[degraded_train, clean_train, degraded_val_4train, clean_val_4train]))
    # dataset=dict(type='ConcatDataset', datasets=[extra_train, extra_aug_train, degraded_train_30, clean_train, degraded_val_4train, clean_val_4train]))   ## for finetune
    dataset = dict(type='ConcatDataset', datasets=[extra_train, clean_train, degraded_val_4train, clean_val_4train]))  ## for finetune

# degraded_val = dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             img_path='images/validation', seg_map_path='annotations/validation'),
#         pipeline=test_pipeline)
#
# clean_val = dict(
#         type=clean_dataset_type,
#         data_root=clean_data_root,
#         data_prefix=dict(
#             img_path='images/validation', seg_map_path='annotations/validation'),
#         pipeline=test_pipeline)
#
# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(type='ConcatDataset', datasets=[degraded_val, clean_val]))

# val_dataloader = dict(
#     batch_size=1,
#     num_workers=4,
#     persistent_workers=True,
#     sampler=dict(type='DefaultSampler', shuffle=False),
#     dataset=dict(
#         type=test_dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             img_path='images/test_20',
#             seg_map_path='annotations/test_20'),
#         pipeline=test_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=test_dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/testing',
            seg_map_path='annotations/testing'),
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
