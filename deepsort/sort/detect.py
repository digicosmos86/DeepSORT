import numpy as np


class Detection(object):
    """
    Stores bounding box and its descriptor.
    The bounding box is represented as an array of coordinates in the following order:
    (x_min, y_min, x_max, y_max)
    """

    def __init__(self, bbox, descriptor=None):
        self.bbox = bbox
        self.descriptor = descriptor

    def to_xyah(self):
        """
        convert to xyah format

        Returns:
            xyah: bounding box in xyah format
        """

        xy_min = self.bbox[:2]
        xy_max = self.bbox[2:]
        wh = xy_max - xy_min
        xy = (xy_min + xy_max) / 2
        wh[0] = wh[0] / wh[1]
        return np.concatenate([xy, wh])

    def to_tlwh(self):
        """
        convert to tlwh format

        Returns:
            tlwh: bounding box in tlwh format
        """

        xy_min = self.bbox[:2]
        xy_max = self.bbox[2:]
        wh = xy_max - xy_min
        tlwh = np.concatenate([xy_min, wh])

        return tlwh
