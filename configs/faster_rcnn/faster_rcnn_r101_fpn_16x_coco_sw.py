_base_ = './faster_rcnn_r50_fpn_4x_coco_sw.py'
model = dict(pretrained='torchvision://resnet101', backbone=dict(depth=101))
