_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn2.py',
    '../_base_/datasets/coco_detection_sw_loader.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=1)))
