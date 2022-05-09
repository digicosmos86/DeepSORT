import numpy as np
from . import matching

def iou(bbox_true, bbox_pred):
    bbox_true_min, bbox_true_max = bbox_true[:2], bbox_true[:2]+bbox_true[2:]
    bbox_pred_min = bbox_pred[:, :2]
    bbox_pred_max = bbox_pred[:, :2] +bbox_pred[:, 2:]

    min_vals = np.c_[np.maximum(bbox_true_min[0], bbox_pred_min[:, 0])[:, np.newaxis],
    np.maximum(bbox_true_min[1], bbox_pred_min[:, 1])[:, np.newaxis]
    ]
    max_vals = np.c_[np.minimum(bbox_true_max[0], bbox_pred_max[:, 0])[:, np.newaxis],
    np.minimum(bbox_pred_max[1], bbox_pred_max[:, 1])[:, np.newaxis]
    ]
    wh = np.maximum(0, max_vals - min_vals)

    intersection = wh.prod(axis=1)
    union = bbox_true[2:].prod(axis=1) + bbox_pred[:, 2:].prod(axis=1)
    return intersection / (union - intersection)

def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = matching.MAX_COST
            continue

        bbox_true = tracks[track_idx].to_tlwh()
        bbox_pred = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1 - iou(bbox_true, bbox_pred)
    return cost_matrix