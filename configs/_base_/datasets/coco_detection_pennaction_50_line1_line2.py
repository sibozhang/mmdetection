# dataset settings
dataset_type = 'CocoDataset'
data_root = '/mnt/data/sibo/Penn_Action'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    # dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', img_scale=(480, 360), keep_ratio=True),
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
        # img_scale=(1333, 800),
        img_scale=[(480, 360),(1333, 800)],
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
    train=dict(
        type=dataset_type,
        ann_file=['/mnt/data/sibo/Penn_Action/pennaction_train_50.json', 'data/line1_annotation/2022-01-10_1_1/2022-01-10_1_1.json', 'data/line1_annotation/2022-01-10_1_2/2022-01-10_1_2.json', 'data/line1_annotation/2022-01-13_1_1/2022-01-13_1_1.json', 'data/line1_annotation/2022-01-13_1_2/2022-01-13_1_2.json', 'data/line1_annotation/2022-01-13_2_1/2022-01-13_2_1.json', 'data/line2_annotation/line2_4class_label/human1_1.json', 'data/line2_annotation/line2_4class_label/human2.json', 'data/line2_annotation/line2_4class_label/human3.json', 'data/line2_annotation/line2_4class_label/loader1_1.json', 'data/line2_annotation/line2_4class_label/loader2.json', 'data/line2_annotation/line2_4class_label/1_2.json', 'data/line2_annotation/line2_4class_label/10_1.json'],
        img_prefix=['/mnt/data/sibo/Penn_Action/frames', 'data/line1_images/2022-01-10_1_1/', 'data/line1_images/2022-01-10_1_2/', 'data/line1_images/2022-01-13_1_1/', 'data/line1_images/2022-01-13_1_2/', 'data/line1_images/2022-01-13_2_1/', 'data/line2_annotation/human1_1/', 'data/line2_annotation/human2/', 'data/line2_annotation/human3/', 'data/line2_annotation/loader1_1/', 'data/line2_annotation/loader2/', 'data/line2_annotation/1_2/', 'data/line2_annotation/10_1'],
        separate_eval=True,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=['data/line1_annotation/2022-01-13_2_1/2022-01-13_2_1.json', 'data/line2_annotation/line2_4class_label/human1_1.json', 'data/line2_annotation/line2_4class_label/human3.json', 'data/line2_annotation/line2_4class_label/loader2.json'],
        img_prefix=['data/line1_images/2022-01-13_2_1/', 'data/line2_annotation/human1_1/', 'data/line2_annotation/human3/', 'data/line2_annotation/loader2/'],
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=['data/line1_annotation/2022-01-13_2_1/2022-01-13_2_1.json', 'data/line2_annotation/line2_4class_label/human1_1.json', 'data/line2_annotation/line2_4class_label/human3.json', 'data/line2_annotation/line2_4class_label/loader2.json'],
        img_prefix=['data/line1_images/2022-01-13_2_1/', 'data/line2_annotation/human1_1/', 'data/line2_annotation/human3/', 'data/line2_annotation/loader2/'],
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
