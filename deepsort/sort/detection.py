class Detection(object):
    """
    stores bounding box in xywh format.

    Args:
        bbox: bounding box in xywh format
        score: confidence score
        class_id: class id
        mask: mask
    """

    def __init__(self, bbox, score, class_id, mask=None):
        self.bbox = bbox
        self.score = score
        self.class_id = class_id
        self.mask = mask

    def to_tlbr(self):
        """
        convert to tlbr format
    
        Returns:
            tlbr: bounding box in tlbr format
        """

        tlbr = self.bbox.copy()
        tlbr[2:] += tlbr[:2]
        return tlbr