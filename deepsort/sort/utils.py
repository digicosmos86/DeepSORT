### There are many bounding box representations
# This module provides a few functions to convert bounding boxes to different formats.
#   - tlbr: top-left and bottom-right format (yxyx format)
#   - xywh: center and size format
#   - ctrwh: center and size format
#   - ctrxy: center and size format
#   - xyxy: top-left and bottom-right format
#
# The following functions are provided:
#
#     * **tlbr_to_xywh**: convert bounding box in tlbr format to xywh format
#     * **xywh_to_tlbr**: convert bounding box in xywh format to tlbr format
#     * **xywh_to_center**: convert bounding box in xywh format to center format
#     * **center_to_xywh**: convert bounding box in center format to xywh format
#     * **tlbr_to_center**: convert bounding box in tlbr format to center format
#     * **center_to_tlbr**: convert bounding box in center format to tlbr format

import numpy as np


def tlbr_to_xywh(tlbr):
    """
    convert tlbr format to xywh format
    """
    xy_min = tlbr[:, [1, 0]]
    xy_max = tlbr[:, [3, 2]]
    wh = xy_max - xy_min
    xy = (xy_min + xy_max) / 2
    xywh = np.concatenate([xy, wh], axis=1)
    return xywh


def xywh_to_tlbr(xywh):
    """
    convert xywh format to tlbr format
    """
    xy_min = xywh[:, [0, 1]] - xywh[:, [2, 3]] / 2
    xy_max = xywh[:, [0, 1]] + xywh[:, [2, 3]] / 2
    tlbr = np.concatenate([xy_min, xy_max], axis=1)

    return tlbr


def xywh_to_center(xywh):
    """
    convert xywh format to center format
    """
    xy_min = xywh[:, [0, 1]] - xywh[:, [2, 3]] / 2
    xy_max = xywh[:, [0, 1]] + xywh[:, [2, 3]] / 2
    xy = (xy_min + xy_max) / 2
    wh = xy_max - xy_min
    center = np.concatenate([xy, wh], axis=1)

    return center


def tlbr_to_xyxy(tlbr):
    """
    convert tlbr format to xyxy format
    """
    xy_min = tlbr[:, [1, 0]]
    xy_max = tlbr[:, [3, 2]]
    xyxy = np.concatenate([xy_min, xy_max], axis=1)

    return xyxy


def scale_bbox_to_original(bboxes, original_hw):
    """
    scale bounding box to original size
    """
    original_hw = np.array(original_hw)

    box_xy_min = bboxes[:, :2] * original_hw
    box_xy_max = bboxes[:, 2:] * original_hw

    bboxes = np.concatenate([box_xy_min[:, [1, 0]], box_xy_max[:, [1, 0]]], axis=1)

    return np.round(bboxes).astype(np.int32)


coco_names = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
