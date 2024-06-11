# --------------------------------------------------------
# InternImage
# Copyright (c) 2022 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
_base_ = [
    '../_base_/models/upernet_r50.py', '../_base_/datasets/weatherproof.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

#pretrained = 'https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth'
pretrained = 'checkpoint_dir/internimage_xl_22k_192to384.pth'
model = dict(
    backbone=dict(
        _delete_=True,
        type='InternImage',
        core_op='DCNv3',
        channels=192,
        depths=[5, 5, 24, 5],
        groups=[12, 24, 48, 96],
        mlp_ratio=4.,
        drop_path_rate=0.4,
        norm_layer='LN',
        layer_scale=1.0,
        offset_scale=2.0,
        post_norm=True,
        with_cp=False,
        out_indices=(0, 1, 2, 3),
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(num_classes=10, in_channels=[192, 384, 768, 1536]),
    auxiliary_head=dict(num_classes=10, in_channels=768),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(320, 320)),
    # test_cfg=dict(mode='whole')
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (640, 640)
    # crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='Resize', img_scale=(1280, 640), ratio_range=(0.5, 2.0)),
    # dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 640),
        # img_scale=(1280, 640),
        # img_scale=(640, 640),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],

        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1280, 640)),
            # dict(type='Resize', img_scale=(1280, 640), keep_ratio=True),
            # dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # dict(type='Pad', size=(640, 640), pad_val=0, seg_pad_val=255),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.00002, betas=(0.9, 0.999), weight_decay=0.05,
    constructor='CustomLayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=39, layer_decay_rate=0.94,
                       depths=[5, 5, 24, 5], offset_lr_scale=1.0))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
# By default, models are trained on 8 GPUs with 2 images per GPU
data = dict(samples_per_gpu=2,
            train=dict(pipeline=train_pipeline),
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner')
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=1500, metric='mIoU', save_best='mIoU')
# fp16 = dict(loss_scale=dict(init_scale=512))
# runner = dict(type='EpochBasedRunner', max_epochs=100)
# optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=0.1, norm_type=2))
# checkpoint_config = dict(by_epoch=True, interval=1)
# evaluation = dict(interval=1, metric='mIoU', save_best='mIoU')

