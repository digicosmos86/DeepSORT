import numpy as np

class KalmanFilter(object):
    """
    Simple implementation of a Kalman filter

    input dimensions:
    u, v, h, r, x, y, vh, vr

    u: x coordinate of the center of the bounding box
    v: y coordinate of the center of the bounding box
    h: height of the bounding box
    r: aspect ration (w/h) of the bounding box
    x: x coordinate of the center of the bounding box
    y: y coordinate of the center of the bounding box
    vh: velocity of the height of the bounding box
    vr: velocity of the aspect ratio of the bounding box
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
                x, y, h, r
        """
        state = np.array(init_measurement)
        velocity = np.zeros(4)
        x_0 = np.concatenate([state, velocity])

        # Use relative location as to determine the standard deviations.
        P_stds = np.array(
            [
                2 * self._std_weight_position * init_measurement[0],  # x
                2 * self._std_weight_position * init_measurement[1],  # y
                2 * self._std_weight_position * init_measurement[2],  # height
                1 * init_measurement[3],  # aspect ratio
                10 * self._std_weight_velocity * init_measurement[0],  # x velocity
                10 * self._std_weight_velocity * init_measurement[1],  # y velocity
                10 * self._std_weight_velocity * init_measurement[2],  # height velocity
                10 * init_measurement[3],  # aspect ratio velocity
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
                self._std_weight_position * x[0],  # x
                self._std_weight_position * x[1],  # y
                self._std_weight_position * x[2],  # height
                1 * x[3],  # aspect ratio
                self._std_weight_velocity * x[0],  # x velocity
                self._std_weight_velocity * x[1],  # y velocity
                self._std_weight_velocity * x[2],  # height velocity
                1 * x[3],  # aspect ratio velocity
            ]
        )

        Q = np.diag(Q_stds**2)
        x_pred = np.dot(self.F, x)

        P_pred = np.dot(np.dot(self.F, P), self.F.T) + Q

        return x_pred, P_pred

    def innovation(self, x_pred, P_pred, z):
        """
        Performs the innovation step of the Kalman filter
        """
        nu = z - np.dot(self.H, x_pred)

        R_stds = np.array(
            [
                self._std_weight_position * x_pred[0],  # x
                self._std_weight_position * x_pred[1],  # y
                self._std_weight_position * x_pred[2],  # height
                0.1 * x_pred[3],  # aspect ratio
            ]
        )
        R = np.diag(R_stds**2)

        S = np.dot(self.H, P_pred).dot(self.H.T) + R

        return nu, S

    def update(self, x_pred, P_pred, z):
        """
        Perform the update step of the Kalman filter.
        """
        nu, S = self.innovation(x_pred, P_pred, z)
        K = np.dot(P_pred, self.H.T).dot(np.linalg.inv(S))
        x_new = x_pred + np.dot(K, nu)
        I_KH = np.eye(self.dim) - np.dot(K, self.H)
        P_new = np.dot(np.dot(I_KH, P_pred), I_KH.T) + np.dot(np.dot(K, self.R), K.T)

        return x_new, P_new
