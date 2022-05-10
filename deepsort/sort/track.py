from enum import Enum

import numpy as np

from unique_names_generator import get_random_name
from unique_names_generator.data import ADJECTIVES, STAR_WARS


class State(Enum):
    PURPOSED = 1
    TRACKING = 2
    DELETED = 3


ocean = [
    "#245c81",
    "#ff7062",
    "#614d80",
    "#338fb8",
    "#ec205b",
    "#00b4b5",
    "#fd6f01",
    "#a00058",
    "#fee40b",
    "#12082d",
]


def convert_colors(color):
    """
    Converts a color in CSS format into (R, G, B) format
    """
    return (
        int(f"0x{color[1:3]}", 16),
        int(f"0x{color[3:5]}", 16),
        int(f"0x{color[5:7]}", 16),
    )


colors = [convert_colors(color) for color in ocean]


class Track(object):
    def __init__(self, mean, cov, id, n_init, max_age, descriptor=None):
        self.mean = mean
        self.cov = cov
        self.id = id
        self.name = get_random_name(combo=[ADJECTIVES, STAR_WARS])
        self.color = colors[id % len(colors)]
        self.n_init = n_init
        self.max_age = max_age

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = State.PURPOSED
        self.max_descriptors = 100
        self.descriptors = []
        if descriptor is not None:
            self.descriptors.append(descriptor)

    def predict(self, kf):
        self.time_since_update += 1
        self.age += 1
        self.mean, self.cov = kf.predict(self.mean, self.cov)

    def update(self, kf, detection):
        self.mean, self.cov = kf.update(self.mean, self.cov, detection.to_xyah())

        self.hits += 1
        self.age += 1
        self.time_since_update = 0

        self.descriptors.append(detection.descriptor)
        if len(self.descriptors) > 100:
            self.descriptors.pop(0)

        if self.state == State.PURPOSED and self.hits >= self.n_init:
            self.state = State.TRACKING

    def mark_for_deletion(self):
        if self.state == State.PURPOSED:
            self.state = State.DELETED
        elif self.time_since_update > self.max_age:
            self.state = State.DELETED

    def to_tlwh(self):
        """
        Converts to (top left x, top left y, width, height) format
        """
        result = self.mean[:4].copy()

        wh = result[2:]
        wh[0] = wh[0] * wh[1]

        xy = result[:2]
        xy -= wh / 2

        return np.concatenate((xy, wh))

    def to_tlbr(self):
        """
        Converts to (top left x, top left y, bottom right x, bottom right y) format
        """
        tlwh = self.to_tlwh()
        tlbr = np.concatenate((tlwh[:2], tlwh[:2] + tlwh[2:4]))

        return tlbr

    def is_proposed(self):
        return self.state == State.PURPOSED

    def is_tracked(self):
        return self.state == State.TRACKING

    def is_deleted(self):
        return self.state == State.DELETED
