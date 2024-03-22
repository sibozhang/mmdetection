_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn3.py',
    '../_base_/datasets/coco_detection_line2_loader.py',
    '../_base_/schedules/schedule_16x.py', '../_base_/default_runtime.py'
]
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
