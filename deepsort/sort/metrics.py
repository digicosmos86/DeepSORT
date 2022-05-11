import numpy as np
from scipy.spatial.distance import cosine


class Metric(object):
    """
    Computes distance metrics and keeps track of features
    """

    def __init__(self, matching_threshold=0.5, budget=100):

        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples = {}

    def partial_fit(self, features, targets, active_targets):
        """
        Updates features, removes features if the number of features
        exceeds the budget.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets if k in self.samples}

    def distance(self, features, targets):
        """Compute distance between features and targets.

        Parameters
        ----------
        features : ndarray
            An NxM matrix of N features of dimensionality M.
        targets : List[int]
            A list of targets to match the given `features` against.

        Returns
        -------
        ndarray
            Returns a cost matrix of shape len(targets), len(features), where
            element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.

            Because we are using cosine

        """
        cost_matrix = np.zeros((len(targets), len(features)))
        for i, target in enumerate(targets):
            if target in self.samples:
                cost_matrix[i, :] = (1.0 - np.matmul(self.samples[target], features.T)).min(
                    axis=0
                )
        return cost_matrix
