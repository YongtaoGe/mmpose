log_level = 'INFO'
load_from = None
resume_from = None
dist_params = dict(backend='nccl')
workflow = [('train', 1)]
checkpoint_config = dict(interval=10)
evaluation = dict(interval=25, metric='mAP', key_indicator='AP')

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
    num_output_channels=17,
    dataset_joints=17,
    dataset_channel=[
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    ],
    inference_channel=[
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
    ])

# model settings
model = dict(
    type='CoordAndHeatmapTopDown',
    pretrained='torchvision://resnet18',
    backbone=dict(type='ResNet', depth=18, num_stages=4, out_indices=(1, 2, 3)),
    # neck=dict(type='FPN', in_channels=[64, 128, 256, 512], out_channels=256, num_outs=4),
    neck_old=dict(type='InputProj', in_channels=(128, 256, 512), out_channel=256),
    neck=dict(
            type='ChannelMapper',
            in_channels=[128, 256, 512],
            kernel_size=1,
            out_channels=256,
            act_cfg=None,
            norm_cfg=dict(type='GN', num_groups=32),
            num_outs=4),
    keypoint_head=dict(
            type='TFHead',
            num_query=300,
            num_classes=80,
            in_channels=2048,
            embed_dims=256,
            reg_num_fcs=2,
            as_two_stage=False,
            transformer=dict(
                type='DeformableDetrTransformer',
                as_two_stage=False,
                encodelayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'),
                ),
                decodelayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(type='MultiScaleDeformableAttention', embed_dims=256),
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
                num_encoder_layers=3,
                num_decoder_layers=1,
                return_intermediate=True),

            positional_encoding=dict(
                type='SinePositionalEncoding',
                num_feats=128,
                normalize=True,
                offset=-0.5),

            loss_hp_keypoint=dict(type='JointsMSELoss', use_target_weight=True, loss_weight=10),
            loss_coord_keypoint=dict(type='L1Loss', use_target_weight=True, loss_weight=1),
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
    image_size=[192, 256],
    heatmap_size=[48, 64],
    num_output_channels=channel_cfg['num_output_channels'],
    num_joints=channel_cfg['dataset_joints'],
    dataset_channel=channel_cfg['dataset_channel'],
    inference_channel=channel_cfg['inference_channel'],
    soft_nms=False,
    # use_nms=False,
    nms_thr=1.0,
    oks_thr=0.9,
    vis_thr=0.2,
    use_gt_bbox=False,
    det_bbox_thr=0.0,
    # bbox_file='',
    bbox_file='data/coco/person_detection_results/'
    'COCO_val2017_detections_AP_H_56_person.json',
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

data_root = 'data/coco'
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    val_dataloader=dict(samples_per_gpu=32),
    test_dataloader=dict(samples_per_gpu=32),
    train=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_train2017.json',
        img_prefix=f'{data_root}/train2017/',
        data_cfg=data_cfg,
        pipeline=train_pipeline),
    val=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
    test=dict(
        type='TopDownCocoDataset',
        ann_file=f'{data_root}/annotations/person_keypoints_val2017.json',
        img_prefix=f'{data_root}/val2017/',
        data_cfg=data_cfg,
        pipeline=val_pipeline),
)
