from kalman import KalmanFilter


class Tracker(object):
    def __init__(self, n_init=3, max_age=30, max_iou_distance=0.7):

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


