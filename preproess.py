# import warnings

import mmcv
import numpy as np
import torch


# import torchvision.transforms.functional as F
# from mmcv.ops import RoIPool
# from mmcv.parallel import collate, scatter
# from mmcv.runner import load_checkpoint
#
# from mmdet.core import get_classes
# from mmdet.datasets import replace_ImageToTensor
# from mmdet.datasets.pipelines import Compose


# def resize(img, size, keep_retio=True):
#     """
#     resize image in Tensor
#     :param img: torch.Tensor
#     :param size: sequence[h, w]
#     :param keep_retio: bool
#     :return: torch.Tensor
#     """
#     if keep_retio:
#         min_edge = np.min(size)
#         return F.resize(img, min_edge)
#     else:
#         return F.resize(img, size)
#
# def normalize(img, mean=[123.675, 116.28, 103.53], std=[58.375, 57.12, 57.375]):
#
#
# def imagePreprocess(img, keep_ratio=True, device=torch.cuda):
#     if isinstance(img, np.ndarray):
#         img = torch.from_numpy(img.astype(np.float32)).to(device)
#     elif isinstance(img, str):
#         # load image
#     else:
#         # not support yet
#         raise TypeError('img must be np.ndarray or a str')


class preprocess:
    def __init__(self, cfg, device):
        super(preprocess, self).__init__()
        # self.img = None
        self.cfg = cfg
        self.device = device

        # self.img_rescale = None
        self.img_scale = cfg.test_pipeline[1]['img_scale']
        self.keep_ratio = cfg.test_pipeline[1]['transforms'][0]['keep_ratio']
        self.backend = 'cv2'

        # normalization parameters
        self.mean = torch.tensor(cfg.img_norm_cfg['mean'], dtype=torch.float32).reshape(1, -1).to(device)
        self.std_inv = 1.0 / torch.tensor(cfg.img_norm_cfg['std'], dtype=torch.float32).reshape(1, -1).to(device)

        self.size_divisor = cfg.test_pipeline[1]['transforms'][3]['size_divisor']

    def run(self, img):
        # self.img = img
        ori_shape = img.shape
        # Resize
        if self.keep_ratio:
            img_rescale = mmcv.imrescale(img,
                                         self.img_scale,
                                         return_scale=False,
                                         backend=self.backend)
        else:
            img_rescale = mmcv.imresize(img,
                                        self.img_scale,
                                        return_scale=False,
                                        backend=self.backend)
        rescale_shape = img_rescale.shape
        # To tensor & normalization
        img_tensor_norm = torch.from_numpy(img_rescale.astype(np.float32)).to(self.device).sub_(self.mean).mul_(
            self.std_inv)
        # Pad
        pad_h = int(np.ceil(img_tensor_norm.shape[0] / self.size_divisor)) * self.size_divisor
        pad_w = int(np.ceil(img_tensor_norm.shape[1] / self.size_divisor)) * self.size_divisor

        img_pad = img_tensor_norm.unsqueeze_(0).permute((0, 3, 1, 2))
        img_pad = torch.nn.functional.pad(img_pad,
                                          [0, pad_w - rescale_shape[1], 0, pad_h - rescale_shape[0]])

        pad_shape = img_pad.shape

        w_scale = int(rescale_shape[1]) / ori_shape[1]
        h_scale = int(rescale_shape[0]) / ori_shape[0]
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        # To mmcv input format
        meta_datas = {"filename": None,
                      "ori_filename": None,
                      "ori_shape": ori_shape,
                      "keep_ratio": True,
                      "img_shape": rescale_shape,
                      "scale_factor": scale_factor,
                      "flip": False,
                      "flip_direction": "horizontal",
                      "img_norm_cfg": None,
                      "pad_shape": pad_shape,
                      "pad_size_divisor": self.size_divisor}

        return {"img_metas": [[meta_datas]], "img": [img_pad.contiguous()]}
