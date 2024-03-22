import asyncio
import cv2
import mmcv
import numpy as np
import torch
import rospy

from PIL import Image
from sensor_msgs.msg import Image as ROSImage
from mmdet.apis import inference_detector, init_detector

from preproess import preprocess

model = None

name_to_dtypes = {
    "rgb8":    (np.uint8,  3),
    "rgba8":   (np.uint8,  4),
    "rgb16":   (np.uint16, 3),
    "rgba16":  (np.uint16, 4),
    "bgr8":    (np.uint8,  3),
    "bgra8":   (np.uint8,  4),
    "bgr16":   (np.uint16, 3),
    "bgra16":  (np.uint16, 4),
    "mono8":   (np.uint8,  1),
    "mono16":  (np.uint16, 1),
    
    # for bayer image (based on cv_bridge.cpp)
    "bayer_rggb8":	(np.uint8,  1),
    "bayer_bggr8":	(np.uint8,  1),
    "bayer_gbrg8":	(np.uint8,  1),
    "bayer_grbg8":	(np.uint8,  1),
    "bayer_rggb16":	(np.uint16, 1),
    "bayer_bggr16":	(np.uint16, 1),
    "bayer_gbrg16":	(np.uint16, 1),
    "bayer_grbg16":	(np.uint16, 1),

    # OpenCV CvMat types
    "8UC1":    (np.uint8,   1),
    "8UC2":    (np.uint8,   2),
    "8UC3":    (np.uint8,   3),
    "8UC4":    (np.uint8,   4),
    "8SC1":    (np.int8,    1),
    "8SC2":    (np.int8,    2),
    "8SC3":    (np.int8,    3),
    "8SC4":    (np.int8,    4),
    "16UC1":   (np.uint16,   1),
    "16UC2":   (np.uint16,   2),
    "16UC3":   (np.uint16,   3),
    "16UC4":   (np.uint16,   4),
    "16SC1":   (np.int16,  1),
    "16SC2":   (np.int16,  2),
    "16SC3":   (np.int16,  3),
    "16SC4":   (np.int16,  4),
    "32SC1":   (np.int32,   1),
    "32SC2":   (np.int32,   2),
    "32SC3":   (np.int32,   3),
    "32SC4":   (np.int32,   4),
    "32FC1":   (np.float32, 1),
    "32FC2":   (np.float32, 2),
    "32FC3":   (np.float32, 3),
    "32FC4":   (np.float32, 4),
    "64FC1":   (np.float64, 1),
    "64FC2":   (np.float64, 2),
    "64FC3":   (np.float64, 3),
    "64FC4":   (np.float64, 4)
}

class SafetyDetection:
    def __init__(self):
        self.model = None
        self.device = "cuda:0"
        self.config = rospy.get_param("~config", "")
        self.checkpoint = rospy.get_param("~checkpoint", "")
        global model
        model = init_detector(self.config, self.checkpoint, self.device)
        self.preprocess = preprocess(model.cfg, self.device)
    
    def detect(self, img):
        data = self.preprocess.run(img)
        # result = inference_detector(model, img)
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **data)
        result = results[0]

        excavator = None
        loader = None
        person = None
        truck = None
        if 0 < len(result):
            excavator = result[0]
        if 1 < len(result):
            loader = result[1]
        if 2 < len(result):
            person = result[2]
        if 3 < len(result):
            truck = result[3]
        return excavator, loader, person, truck

    def msgToImg(self, msg):
        if not msg.encoding in name_to_dtypes:
            raise TypeError("Unrecognized encoding {}".format(msg.encoding))
        dtype_class, channels = name_to_dtypes[msg.encoding]
        dtype = np.dtype(dtype_class)
        dtype = dtype.newbyteorder(">" if msg.is_bigendian else "<")
        shape = (msg.height, msg.width, channels)
        data = np.fromstring(msg.data, dtype=dtype).reshape(shape)
        data.strides = (msg.step, dtype.itemsize * channels, dtype.itemsize)
        if channels == 1:
            data = data[...,0]
        return np.nan_to_num(data)