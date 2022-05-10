import numpy as np

from .kalman import KalmanFilter
from .metrics import Metric
from .hungarian import matching_cascade, hungarian_algorithm_matching, gate_cost_matrix
from .iou import iou_cost
from .track import Track


class Tracker(object):
    def __init__(self, metric=Metric(), n_init=3, max_age=30, max_iou_distance=0.7):

        self.metric = metric
        self.n_init = n_init
        self.max_age = max_age
        self.max_iou_distance = max_iou_distance
        self.tracks = []

        self.kf = KalmanFilter()
        self._next_id = 1

    def update(self, detections):
        """Update tracks with detections.
        Args:
            detections (list): List of detections.

        Returns:
            list: List of tracks with updated tracks.
        """

        # Use a closure to get a gated metric function to compute cost
        def gated_metric(tracks, detections, track_indices, detection_indices):
            features = np.array([detections[i].descriptor for i in detection_indices])
            targets = np.array([tracks[i].id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = gate_cost_matrix(
                self.kf, cost_matrix, tracks, detections, track_indices, detection_indices
            )

            return cost_matrix

        tracking = [i for i, t in enumerate(self.tracks) if t.is_tracked()]
        untracked = [i for i, t in enumerate(self.tracks) if not t.is_tracked()]

        # First, perform matching cascade
        matches_1, unmatched_1, unmatched_detections = matching_cascade(
            distance_metric=gated_metric,
            max_distance=self.max_iou_distance,
            cascade_depth=1,
            tracks=self.tracks,
            detections=detections,
            track_indices=tracking,
        )

        # Then, perform IoU matching on the untracked and inactive tracks
        iou_track_candidates = untracked + [
            k for k in unmatched_1 if self.tracks[k].time_since_update == 1
        ]

        unmatched_1 = [k for k in unmatched_1 if self.tracks[k].time_since_update != 1]

        matches_2, unmatched_2, unmatched_detections = hungarian_algorithm_matching(
            iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        matches = matches_1 + matches_2
        unmatched = list(set(unmatched_1 + unmatched_2))

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        for track_idx in unmatched:
            self.tracks[track_idx].mark_for_deletion()

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            mean, cov = self.kf.initial_state(detection.to_xyah())
            self.tracks.append(
                Track(
                    mean,
                    cov,
                    self._next_id,
                    self.n_init,
                    self.max_age,
                    detection.descriptor,
                )
            )
            self._next_id += 1

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        tracking = [t.id for t in self.tracks if t.is_tracked()]

        features, targets = [], []

        for track in self.tracks:
            if not track.is_tracked():
                continue
            features += features
            targets += [track.id] * len(track.descriptors)

        self.metric.partial_fit(np.array(features), np.array(targets), tracking)

    def predict(self):

        for track in self.tracks:
            track.predict(self.kf)
