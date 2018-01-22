import os
import sys
import random
import math
import numpy as np
import scipy.misc
import matplotlib
import matplotlib.pyplot as plt
import cv2

import models.Mask_RCNN.coco
import models.Mask_RCNN.utils
import models.Mask_RCNN.model as modellib
import models.Mask_RCNN.visualize

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "models/Mask_RCNN/", "mask_rcnn_coco.h5")

class InferenceConfig(models.Mask_RCNN.coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def mask_rcnn_detect(image_url):
    image_url = image_url
    print(image_url)

    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    config = InferenceConfig()
    #config.print()

    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    image = scipy.misc.imread("." + image_url)
    results = model.detect([image], verbose=1)
    r = results[0]

    mask_cnn_url = models.Mask_RCNN.visualize.display_instances(image_url,image, r['rois'], r['masks'], r['class_ids'],
                                                                class_names, r['scores'], title="", figsize=(16, 16) )
    return mask_cnn_url
