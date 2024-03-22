# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    # workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=['data/line2_annotation/loader1_1.json', 'data/line2_annotation/loader2.json', 'data/line2_annotation/human1_1.json', 'data/line2_annotation/human2.json', 'data/line2_annotation/human3.json', 'data/line2_annotation/1_2.json', 'data/line2_annotation/10_1.json'],
        img_prefix=['data/line2_annotation/loader1_1/', 'data/line2_annotation/loader2/', 'data/line2_annotation/human1_1/', 'data/line2_annotation/human2/', 'data/line2_annotation/human3/', 'data/line2_annotation/1_2/', 'data/line2_annotation/10_1'],
        separate_eval=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=['data/line2_loader/COCO_line2_loader_train.json', 'data/sw_loader/sw_loader_train/COCO_sw_loader_train.json', 'data/sw_loader/sw_loader_test/COCO_sw_loader_test.json'],
        img_prefix=['data/line2_loader/images/', 'data/sw_loader/sw_loader_train/images/', 'data/sw_loader/sw_loader_test/images/'],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=['data/line2_loader/COCO_line2_loader_train.json', 'data/sw_loader/sw_loader_train/COCO_sw_loader_train.json', 'data/sw_loader/sw_loader_test/COCO_sw_loader_test.json'],
        img_prefix=['data/line2_loader/images/', 'data/sw_loader/sw_loader_train/images/', 'data/sw_loader/sw_loader_test/images/'],
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')