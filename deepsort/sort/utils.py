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

import cv2
import numpy as np
import tensorflow as tf

from .detect import Detection


def scale_bbox_to_original(bboxes, original_hw):
    """
    scale bounding box to original size
    """
    original_hw = np.array(original_hw)

    box_xy_min = bboxes[:, :2] * original_hw
    box_xy_max = bboxes[:, 2:] * original_hw

    bboxes = np.concatenate([box_xy_min[:, [1, 0]], box_xy_max[:, [1, 0]]], axis=1)

    return np.round(bboxes).astype(np.int32)


def to_detections(image, bboxes, model):
    """Convert bboxes to detections.

    Args:
        image (numpy.ndarray): Image.
        bboxes (numpy.ndarray): nx4 array of bboxes.
        model (tf.keras.models.Model): the deep appearance descriptor model
    """

    # Convert bboxes to detections.
    if bboxes.shape[0] == 0:
        return []
    detections = []

    patches = []

    for bbox in bboxes:

        x1, y1, x2, y2 = bbox.astype(np.int32)
        # Get the patch.
        patch = image[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        # Resize the patch to the model input size.
        patch = cv2.resize(patch, (64, 128))
        # Add the patch to the list of patches.
        patches.append(patch)


    if len(patches) == 0:
        return patches

    # Get the deep appearance descriptor for the patches.
    patches = tf.convert_to_tensor(patches, dtype=tf.float32) / 255.0
    descriptors = model(patches, train=False).numpy()

    for bbox, descriptor in zip(bboxes, descriptors):
        # Create a detection.
        detection = Detection(bbox, descriptor)
        # Add the detection to the list of detections.
        detections.append(detection)

    return detections


def draw_track(image, track):
    """Draw bboxes on image.

    Args:
        image (numpy.ndarray): Image.
        bboxes (numpy.ndarray): length-4 array of bbox.
        name (str): name of the bbox.
    """

    # Draw bboxes on image.
    x1, y1, x2, y2 = track.to_tlbr().astype(np.int32)

    image = cv2.rectangle(image, (x1, y1), (x2, y2), track.color, 2)
    image = cv2.rectangle(
        image,
        (x1, y1 - 30),
        (x2 + (len(track.name) + len(str(track.id))) * 17, y1),
        track.color,
        -1,
    )
    cv2.putText(image, track.name, (int(x1), int(y1 - 10)), 0, 0.75, (255, 255, 255), 2)

    return image
