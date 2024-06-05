# _base_ = [
#     '../_base_/default_runtime.py', '../_base_/datasets/weather_proof_add_clean_448x448.py'
# ]

# _base_ = [
#     '../_base_/default_runtime.py', '../_base_/datasets/weather_proof_cdv4train_test_896x896.py'
# ]

_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/wp_cv_d30_e30.py'
]

norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (896, 896)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size)
num_classes = 9

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DINOv2',
        version='large',
        freeze=False,
        load_from='/home/zhangxd/project/mmsegmentation_depthanything_9/checkpoints/depth_anything_vitl14.pth'),
    neck=dict(type='Feature2Pyramid', embed_dim=1024, rescales=[4, 2, 1, 0.5]),
    decode_head=dict(
            type='UPerHead',
            # in_channels=[128, 256, 512, 1024],
            in_channels=[1024, 1024, 1024, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=[
                dict(type='CrossEntropyLoss', use_sigmoid=False, loss_name='loss_ce', loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0)]),
        auxiliary_head=dict(
                type='SegmenterMaskTransformerHead',
                in_channels=1024,
                channels=768,
                num_classes=num_classes,
                num_layers=2,
                num_heads=12,
                embed_dims=768,
                dropout_ratio=0.0,
                # loss_decode=dict(
                #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_decode=
                    dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(426, 426)))

# dataset config
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomChoiceResize',
        scales=[int(x * 0.1 * 896) for x in range(5, 21)],
        resize_type='ResizeShortestEdge',
        max_size=3584),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(3584, 896), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
# train_dataloader = dict(batch_size=1, dataset=dict(pipeline=train_pipeline))
# val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

# optim_wrapper = dict(
#     _delete_=True,
#     type='OptimWrapper',
#     optimizer=dict(
#         type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05),
#     constructor='LayerDecayOptimizerConstructor',
#     paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone.dinov2': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.dinov2.norm': backbone_norm_multi,
    'pos_embed': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.dinov2.blocks.{block_id}.norm': backbone_norm_multi
    for block_id in range(24)
})
# optimizer
optimizer = dict(
    type='AdamW', lr=0.00003, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999))
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))

find_unused_parameters=True

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=160000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# training schedule for 160k
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=10, save_best='mIoU', max_keep_ckpts=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)

work_dir = '../work_dirs/'
