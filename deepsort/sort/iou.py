import numpy as np
from .hungarian import MAX_COST


# def iou(bbox_true, bbox_pred):
#     """
#     Compute the Intersection over Union (IoU) between two bboxes
#     BBox format: [x1, y1, w, h]
#     """
#     bbox_true_min, bbox_true_max = bbox_true[:2], bbox_true[:2] + bbox_true[2:]
#     bbox_pred_min = bbox_pred[:, :2]
#     bbox_pred_max = bbox_pred[:, :2] + bbox_pred[:, 2:]

#     min_vals = np.c_[
#         np.maximum(bbox_true_min[0], bbox_pred_min[:, 0])[:, np.newaxis],
#         np.maximum(bbox_true_min[1], bbox_pred_min[:, 1])[:, np.newaxis],
#     ]
#     max_vals = np.c_[
#         np.minimum(bbox_true_max[0], bbox_pred_max[:, 0])[:, np.newaxis],
#         np.minimum(bbox_pred_max[1], bbox_pred_max[:, 1])[:, np.newaxis],
#     ]
#     wh = np.maximum(0, max_vals - min_vals)

#     intersection = wh.prod(axis=1)
#     union = bbox_true[2:].prod(axis=1) + bbox_pred[:, 2:].prod(axis=1)
#     return intersection / (union - intersection)


# def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
#     if track_indices is None:
#         track_indices = np.arange(len(tracks))
#     if detection_indices is None:
#         detection_indices = np.arange(len(detections))

#     cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
#     for row, track_idx in enumerate(track_indices):
#         if tracks[track_idx].time_since_update > 1:
#             cost_matrix[row, :] = MAX_COST
#             continue

#         bbox_true = tracks[track_idx].to_tlwh()
#         if bbox_true.ndim == 1:
#             bbox_true = bbox_true[np.newaxis, :]
#         bbox_pred = np.asarray([detections[i].to_tlwh() for i in detection_indices])
#         if bbox_pred.ndim == 1:
#             bbox_pred = bbox_pred[np.newaxis, :]
#         cost_matrix[row, :] = 1 - iou(bbox_true, bbox_pred)
#     return cost_matrix


def iou(bbox, candidates):
    """Computer intersection over union.

    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.

    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.

    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices=None,
             detection_indices=None):
    """An intersection over union distance metric.

    Parameters
    ----------
    tracks : List[deep_sort.track.Track]
        A list of tracks.
    detections : List[deep_sort.detection.Detection]
        A list of detections.
    track_indices : Optional[List[int]]
        A list of indices to tracks that should be matched. Defaults to
        all `tracks`.
    detection_indices : Optional[List[int]]
        A list of indices to detections that should be matched. Defaults
        to all `detections`.

    Returns
    -------
    ndarray
        Returns a cost matrix of shape
        len(track_indices), len(detection_indices) where entry (i, j) is
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = MAX_COST
            continue

        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].to_tlwh() for i in detection_indices])
        cost_matrix[row, :] = 1. - iou(bbox, candidates)
    return cost_matrix