_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/deeplabv2red_r50-d8.py',
    '../../_base_/datasets/uda_synthiaHR_to_cityscapesHR_1024x1024.py',
    '../../_base_/uda/dacs_a999_fdthings.py',
    '../../_base_/schedules/adamw.py', '../../_base_/schedules/poly10warm.py'
]
gpu_model = 'NVIDIATITANRTX'
n_gpus = 1
seed = 0
model = dict(
    pretrained='open-mmlab://resnet101_v1c',
    backbone=dict(depth=101),
    decode_head=dict(
        single_scale_head='DLV2Head',
        type='HRDAHead',
        attention_classwise=True,
        hr_loss_weight=0.1),
    type='HRDAEncoderDecoder',
    scales=[1, 0.5],
    hr_crop_size=(512, 512),
    feature_scale=0.5,
    crop_coord_divisible=8,
    hr_slide_inference=True,
    test_cfg=dict(
        mode='slide',
        batched_slide=True,
        stride=[512, 512],
        crop_size=[1024, 1024]))
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=1,
    train=dict(
        target=dict(crop_pseudo_margins=[30, 240, 30, 30]),
        rare_class_sampling=dict(
            min_pixels=3000, class_temp=0.01, min_crop_ratio=2.0)))
uda = dict(
    mask_mode='separatetrgaug',
    mask_alpha='same',
    mask_pseudo_threshold='same',
    mask_lambda=1,
    mask_generator=dict(
        type='block', mask_ratio=0.7, mask_block_size=64, _delete_=True))
optimizer_config = None
optimizer = dict(
    lr=6e-05, paramwise_cfg=dict(custom_keys=dict(head=dict(lr_mult=10.0))))
runner = dict(type='IterBasedRunner', max_iters=40000)
checkpoint_config = dict(by_epoch=False, interval=500, max_keep_ckpts=-1)
evaluation = dict(interval=500, metric='mIoU')
name = '240810_1333_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_b6485'
exp = 821
name_dataset = 'synthiaHR2cityscapesHR_1024x1024'
name_architecture = 'hrda1-512-0.1_dlv2red_sl_r101v1c'
name_encoder = 'r101v1c'
name_decoder = 'hrda1-512-0.1_dlv2red_sl'
name_uda = 'dacs_a999_fdthings_rcs0.01-2.0_cpl2_m64-0.7-spta'
name_opt = 'adamw_6e-05_pmTrue_poly10warm_1x2_40k'
work_dir = 'work_dirs/local-exp821/240810_1333_synHR2csHR_1024x1024_dacs_a999_fdthings_rcs001-20_cpl2_m64-07-spta_hrda1-512-01_dlv2red_sl_r101v1c_poly10warm_s0_b6485'
git_rev = 'unknown'
