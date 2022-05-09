from enum import IntEnum

State = IntEnum("PURPOSED", "TRACKING", "DELETED")

colors = ocean = [
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


class Track(object):
    def __init__(self, mean, cov, id, n_init, max_age, descriptor=None):
        self.mean = mean
        self.cov = cov
        self.id = id
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
        self.mean, self.cov = kf.predict(self.mean, self.cov)

    def update(self, mean, cov, descriptor):
        self.mean = mean
        self.cov = cov
        self.descriptors.append(descriptor)
        self.age += 1
        self.time_since_update = 0
        self.state = State.TRACKING
        self.hits += 1

        if descriptor is not None:
            self.descriptors.append(descriptor)
            if len(self.descriptors) > 100:
                self.descriptors.pop(0)

    def is_proposed(self):
        return self.state == State.PURPOSED

    def is_tracked(self):
        return self.state == State.TRACKING

    def is_deleted(self):
        return self.state == State.DELETED
