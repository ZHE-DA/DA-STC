log_config = dict(
    interval=50,
    img_interval=1000,
    hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='HRDAEncoderDecoder',
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='HRDAHead',
        in_channels=2048,
        in_index=3,
        dilations=(6, 12),
        num_classes=19,
        align_corners=False,
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        single_scale_head='DLV2Head',
        attention_classwise=True,
        hr_loss_weight=0.1),
    train_cfg=dict(
        work_dir=
        'work_dirs/local-exp811/240810_1100_synthiaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_synthia',
        log_config=dict(
            interval=50,
            img_interval=1000,
            hooks=[dict(type='TextLoggerHook', by_epoch=False)])),
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]),
    scales=[1, 0.5],
    hr_crop_size=(512, 512),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True)
dataset_type = 'CityscapesDataset'
data_root = 'data/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (1024, 1024)
synthia_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2560, 1520)),
    dict(type='RandomCrop', crop_size=(1024, 1024), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
cityscapes_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024)),
    dict(type='RandomCrop', crop_size=(1024, 1024)),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        type='UDADataset',
        source=dict(
            type='synthiaDataset',
            data_root='data/synthia/',
            img_dir='RGB',
            ann_dir='GT/LABELS',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(2560, 1520)),
                dict(
                    type='RandomCrop',
                    crop_size=(1024, 1024),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(
                    type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ]),
        target=dict(
            type='CityscapesDataset',
            data_root='data/cityscapes/',
            img_dir='leftImg8bit/train',
            ann_dir='gtFine/train',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations'),
                dict(type='Resize', img_scale=(2048, 1024)),
                dict(type='RandomCrop', crop_size=(1024, 1024)),
                dict(type='RandomFlip', prob=0.5),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(
                    type='Pad', size=(1024, 1024), pad_val=0, seg_pad_val=255),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_semantic_seg'])
            ],
            crop_pseudo_margins=[30, 240, 30, 30]),
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0)),
    val=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        data_root='data/cityscapes/',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
uda = dict(
    type='DACS',
    source_only=False,
    alpha=0.999,
    pseudo_threshold=0.968,
    pseudo_weight_ignore_top=0,
    pseudo_weight_ignore_bottom=0,
    imnet_feature_dist_lambda=0.005,
    imnet_feature_dist_classes=[6, 7, 11, 12, 13, 14, 15, 16, 17, 18],
    imnet_feature_dist_scale_min_ratio=0.75,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.2,
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(type='block', mask_ratio=0.7, mask_block_size=64),
    debug_img_interval=1000,
    print_grad_magnitude=False)
use_ddp_wrapper = True
optimizer = dict(
    type='AdamW',
    lr=6e-05,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
optimizer_config = None
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
gpu_model = 'NVIDIATITANRTX'
n_gpus = 1
seed = 0
# checkpoint_config = dict(by_epoch=False, interval=40000, max_keep_ckpts=1)
# evaluation = dict(interval=4000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=-1)
evaluation = dict(interval=500, metric='mIoU')
name = '240810_1100_synthiaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_synthia'
exp = 811
name_dataset = 'synthiaHR2cityscapesHR_1024x1024'
name_architecture = 'hrda1-512-0.1_dlv2red_sl_r101v1c'
name_encoder = 'r101v1c'
name_decoder = 'hrda1-512-0.1_dlv2red_sl'
name_uda = 'dacs_a999_fdthings_rcs0.01-2.0_cpl2_m64-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
work_dir = 'work_dirs/local-exp811/240810_1100_synthiaHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_synthia'
git_rev = 'unknown'
gpu_ids = range(1, 2)
