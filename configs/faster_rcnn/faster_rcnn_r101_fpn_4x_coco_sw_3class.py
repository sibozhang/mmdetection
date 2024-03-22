_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn3.py',
    '../_base_/datasets/coco_detection_sw_3class.py',
    '../_base_/schedules/schedule_4x.py', '../_base_/default_runtime.py'
]
model = dict(
    pretrained='torchvision://resnet101', 
    backbone=dict(depth=101),
    
    test_cfg=dict(
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=3))
    
    )
 