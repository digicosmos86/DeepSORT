import numpy as np
import scipy.linalg


class KalmanFilter(object):
    """
    Simple implementation of a Kalman filter
    input dimensions:
    x, y, a, h, vx, vy, va, vh
    """

    def __init__(self):
        self.dim = 4
        self.dt = 1

        # initialize state transition matrix F
        self.F = np.eye(8, 8)
        for i in range(self.dim):
            self.F[i, i + self.dim] = self.dt

        # initialize output update matrix H
        self.H = np.eye(4, 8)

        # this is taken directly here as a hack
        # https://github.com/Rishita32/Yolov5_Deepsort_KalmanFilter/blob/161f1bf4e040b095b0e9de8bca5719d299089c4b/deep_sort_pytorch/deep_sort/sort/kalman_filter.py#L52

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initial_state(self, init_measurement):
        """
        Initialize the state from a new, untracked measurement.
        Args:
            measurement: a 1d ndarray of length 4, each element indicates:
                x, y, a, h
        """
        state = np.array(init_measurement)
        velocity = np.zeros(4)
        x_0 = np.concatenate([state, velocity])

        # Use relative location as to determine the standard deviations.
        P_stds = np.array(
            [
                2 * self._std_weight_position * init_measurement[3],  # x
                2 * self._std_weight_position * init_measurement[3],  # y
                1e-2,  # aspect ratio
                2 * self._std_weight_position * init_measurement[3],  # height
                10 * self._std_weight_velocity * init_measurement[3],  # x velocity
                10 * self._std_weight_velocity * init_measurement[3],  # y velocity
                1e-5,  # aspect ratio velocity
                10 * self._std_weight_velocity * init_measurement[3],  # height velocity
            ]
        )

        P = np.diag(P_stds**2)

        return x_0, P

    def predict(self, x, P):
        """
        Perform the prediction step of the Kalman filter.
        """
        Q_stds = np.array(
            [
                self._std_weight_position * x[3],  # x
                self._std_weight_position * x[3],  # y
                1e-2,  # aspect ratio
                self._std_weight_position * x[3],  # height
                self._std_weight_velocity * x[3],  # x velocity
                self._std_weight_velocity * x[3],  # y velocity
                1e-5,  # aspect ratio velocity
                self._std_weight_velocity * x[3],  # height velocity
            ]
        )

        Q = np.diag(Q_stds**2)
        x_pred = np.dot(self.F, x)

        P_pred = np.dot(np.dot(self.F, P), self.F.T) + Q

        return x_pred, P_pred


    def project(self, x_pred, P_pred):
        """
        Performs the innovation step of the Kalman filter
        """
        nu = np.dot(self.H, x_pred)

        R_stds = np.array(
            [
                self._std_weight_position * x_pred[0],  # x
                self._std_weight_position * x_pred[1],  # y
                1e-1,  # aspect ratio
                self._std_weight_position * x_pred[3],  # height
            ]
        )
        R = np.diag(R_stds**2)

        S = np.dot(self.H, P_pred).dot(self.H.T) + R

        return nu, S

    def update(self, x_pred, P_pred, z):
        """
        Perform the update step of the Kalman filter.
        """
        projected_mean, projected_cov = self.project(x_pred, P_pred)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(P_pred, self.H.T).T,
            check_finite=False).T
        innovation = z - projected_mean

        x_pred_new = x_pred + np.dot(innovation, kalman_gain.T)
        P_pred_new = P_pred - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return x_pred_new, P_pred_new

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`.
        If `only_position` is False, the chi-square distribution has 4 degrees of freedom, otherwise 2.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cho_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cho_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_mahalanobis = np.sum(z * z, axis=0)
        return squared_mahalanobis


# inclusion of standard gating thresholds.
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}
