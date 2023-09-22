_base_ = ['../_base_/default_runtime.py']

# runtime
train_cfg = dict(max_epochs=20, val_interval=1)

# optimizer
optim_wrapper = dict(optimizer=dict(
    type='Adam',
    lr=1e-3,
))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', begin=0, end=500, start_factor=0.001,
        by_epoch=False),  # warm-up
    dict(
        type='MultiStepLR',
        begin=0,
        end=train_cfg['max_epochs'],
        milestones=[15, 18],
        gamma=0.1,
        by_epoch=True)
]

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# codec settings
codec = dict(type='RegressionLabel', input_size=(192, 256))

# model settings
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        type='ResNet',
        depth=50,
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
    ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='RLEHead',
        in_channels=2048,
        num_joints=17,
        loss=dict(type='RLELoss', use_target_weight=True),
        decoder=codec),
    test_cfg=dict(
        flip_test=True,
        shift_coords=True,
    ))

# base dataset settings
dataset_type = 'CocoDataset'
data_mode = 'topdown'
data_root = 'data/'

# pipelines
train_pipeline = [
    dict(type='LoadImage', ignore_empty=True),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(type='RandomBBoxTransform'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]
val_pipeline = [
    dict(type='LoadImage', ignore_empty=True),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]


# train datasets
dataset_coco = dict(
    type='RepeatDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_train2017_0.05.json',
        data_prefix=dict(img='coco/train2017/'),
        pipeline=[],
    ),
    times=3)

dataset_controlpose = dict(
    type='RepeatDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='controlpose/annotations/controlpose_pseudo.json',
        data_prefix=dict(img='controlpose/images/'),
        pipeline=[],
    ),
    times=1)

dataset_bedlam = dict(
    type='RepeatDataset',
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='bedlam/training_labels/bedlam_train_refine.json',
        data_prefix=dict(img='bedlam/training_images/'),
        pipeline=[],
    ),
    times=1)

# data loaders
train_dataloader = dict(
    batch_size=256,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CombinedDataset',
        metainfo=dict(from_file='configs/_base_/datasets/coco.py'),
        # datasets=[dataset_coco, dataset_bedlam],
        datasets=[dataset_controlpose, dataset_coco],
        pipeline=train_pipeline,
        test_mode=False,
    )
    )
val_dataloader = dict(
    batch_size=128,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file='coco/annotations/person_keypoints_val2017.json',
        # bbox_file=f'{data_root}person_detection_results/'
        # 'COCO_val2017_detections_AP_H_56_person.json',
        data_prefix=dict(img='coco/val2017/'),
        test_mode=True,
        pipeline=val_pipeline,
    ))
test_dataloader = val_dataloader

# hooks
default_hooks = dict(checkpoint=dict(save_best='coco/AP', rule='greater'))

# evaluators
val_evaluator = dict(
    type='CocoMetric',
    ann_file=f'{data_root}coco/annotations/person_keypoints_val2017.json',
    score_mode='bbox_rle')
test_evaluator = val_evaluator
