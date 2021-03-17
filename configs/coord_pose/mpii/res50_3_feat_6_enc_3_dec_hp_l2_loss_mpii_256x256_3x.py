log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
# evaluation = dict(interval=25, metric='mAP', key_indicator='AP')
evaluation = dict(interval=25, metric='PCKh', key_indicator='PCKh')
optimizer = dict(
    type='AdamW',
    lr=4e-3,
    weight_decay=1e-5,
    # type='Adam',
    # lr=4e-,
    # weight_decay=1e-4,
    paramwise_cfg = dict(
        custom_keys={
            'transformer': dict(lr_mult=0.1, decay_mult=1.0),
            # 'query_embed': dict(lr_mult=0.5, decay_mult=1.0),
        },
        # bypass_duplicate=True
    )
)

optimizer_config = dict(grad_clip=None,
                        # grad_clip=dict(max_norm=5, norm_type=2),
                        # paramwise_cfg=dict(
                        #     custom_keys={
                        #         'transformer': dict(grad_clip=dict(max_norm=0.1, norm_type=2)),
                        #         # 'query_embed': dict(lr_mult=0.1, decay_mult=1.0),
                        #     },
                        # )
                    )

# optimizer
# optimizer = dict(
#     type='AdamW',
#     lr=0.0001,
#     weight_decay=0.0001,
#     paramwise_cfg=dict(
#         custom_keys={'transformer': dict(lr_mult=0.1, decay_mult=1.0)})
# )
# optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))

# learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=2500,
#     warmup_ratio=0.001,
#     step=[170, 190, 200])
# total_epochs = 210

# lr_config = dict(
#     policy='Linear',
#     warmup='linear',
#     warmup_iters=2400,
#     warmup_ratio=0.1,
#     by_epoch=False
# )

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=2400,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)

total_epochs = 325

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

channel_cfg = dict(
    num_output_channels=16,
    dataset_joints=16,
    dataset_channel=list(range(16)),
    inference_channel=list(range(16)))

# model settings
model = dict(
    type='CoordAndHeatmapTopDown',
    pretrained='torchvision://resnet50',
    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(1, 2, 3)),
    # neck=dict(type='FPN', in_channels=[64, 128, 256, 512], out_channels=256, num_outs=4),
    neck=dict(type='InputProj', in_channels=(512, 1024, 2048), out_channel=256),
    keypoint_head=dict(
        type='HybridTransHead',
        # num_joints=channel_cfg['num_output_channels'],
        num_joints=channel_cfg['dataset_joints'],
        # loss_keypoint=dict(type='SmoothL1Loss', use_target_weight=True, loss_weight=1000),
        # loss_keypoint=dict(type='L1Loss', use_target_weight=True, loss_weight=40),
        loss_hp_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=50),
        loss_coord_keypoint=dict(type='WingLoss', use_target_weight=True, loss_weight=1),
        # loss_coord_keypoint=dict(type='L2Loss', use_target_weight=True, loss_weight=1),
        # in_channels=2048,
        # out_indices=(1, 2, 3),
        num_levels=3,
        num_encoder_layers=6,
        num_decoder_layers=3,
        decoder_layer_type="deformable",
        # decoder_layer_type="standard",
        with_box_refine=True,
        num_stages=1,
        neck_type='InputProj',
        use_heatmap_loss=True,

    ),
    train_cfg=dict(),
    test_cfg = dict(
        flip_test=True,
        post_process='default',
        shift_heatmap=True,
        modulate_kernel=11)
)


data_cfg = dict(
    image_size=[256, 256],
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    use_gt_bbox=True,
    bbox_file=None,
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownRandomFlip', flip_prob=0.5),
    dict(
        type='TopDownHalfBodyTransform',
        num_joints_half_body=8,
        prob_half_body=0.3),
    dict(
        type='TopDownGetRandomScaleRotation', rot_factor=40, scale_factor=0.25),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    # dict(
    #     type='TopDownGenerateTarget',
    #     kernel=[(11, 11), (9, 9), (7, 7), (5, 5)],
    #     encoding='Megvii'),
    dict(
        type='TopDownGenerateCoordAndHeatMapTarget',
        encoding='UDP',
        sigma=2),
    dict(
        type='Collect',
        keys=['img', 'coord_target', 'coord_target_weight', 'hp_target', 'hp_target_weight'],
        meta_keys=[
            'image_file', 'joints_3d', 'joints_3d_visible', 'center', 'scale',
            'rotation', 'bbox_score', 'flip_pairs'
        ]),
]

val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='TopDownAffine'),
    dict(type='ToTensor'),
    dict(
        type='NormalizeTensor',
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
    dict(
        type='Collect',
        keys=[
            'img',
        ],
        meta_keys=[
            'image_file', 'center', 'scale', 'rotation', 'bbox_score',
            'flip_pairs'
        ]),
]

test_pipeline = val_pipeline


data_root = 'data/mpii'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_train.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownMpiiDataset',
        ann_file=f'{data_root}/annotations/mpii_val.json',
        img_prefix=f'{data_root}/images/',
        data_cfg=data_cfg,
        pipeline=test_pipeline),
)

