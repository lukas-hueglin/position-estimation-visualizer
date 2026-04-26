import numpy as np
from filters.observer import Observer

class Kalman_v1(Observer):
    def __init__(self, init_state, init_var, dt):
        super().__init__(init_state, init_var)

        self.f = lambda x: np.array([
            dt*x[4]*x[0] - dt*x[5]*x[1] - dt*x[6]*x[2] - dt*x[7]*x[3] + x[0],
            dt*x[4]*x[1] + dt*x[5]*x[0] + dt*x[6]*x[3] - dt*x[7]*x[2] + x[1],
            dt*x[4]*x[2] - dt*x[5]*x[3] + dt*x[6]*x[0] + dt*x[7]*x[1] + x[2],
            dt*x[4]*x[3] + dt*x[5]*x[2] - dt*x[6]*x[1] + dt*x[7]*x[0] + x[3],
            x[4], x[5], x[6], x[7]
        ])


        self.F = lambda x: np.array([
            [dt*x[4]+1, -dt*x[5], -dt*x[6], -dt*x[7], dt*x[0], -dt*x[1], -dt*x[2], -dt*x[3]],
            [dt*x[5], dt*x[4]+1, dt*x[7], -dt*x[6], dt*x[1], dt*x[0], dt*x[3], -dt*x[2]],
            [dt*x[6], -dt*x[7], dt*x[4]+1, dt*x[5], dt*x[2], -dt*x[3], dt*x[0], dt*x[1]],
            [dt*x[7], dt*x[6], -dt*x[5], dt*x[4]+1, dt*x[3], dt*x[2], -dt*x[1], dt*x[0]],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])

        #self.F = lambda x: np.array([
        #    [1, 0, 0, 0, dt, 0, 0, 0],
        #    [0, 1, 0, 0, 0, dt, 0, 0],
        #    [0, 0, 1, 0, 0, 0, dt, 0],
        #    [0, 0, 0, 1, 0, 0, 0, dt],
        #    [0, 0, 0, 0, 1, 0, 0, 0],
        #    [0, 0, 0, 0, 0, 1, 0, 0],
        #    [0, 0, 0, 0, 0, 0, 1, 0],
        #    [0, 0, 0, 0, 0, 0, 0, 1],
        #])

        #self.f = lambda x: self.F(x) @ x

        self.h = lambda x: 2 * np.array([
            x[0]*x[4] + x[1]*x[5] + x[2]*x[6] + x[3]*x[7],
            -x[1]*x[4] + x[0]*x[5] - x[3]*x[6] + x[2]*x[7],
            -x[2]*x[4] + x[3]*x[5] + x[0]*x[6] - x[1]*x[7],
            -x[3]*x[4] - x[2]*x[5] + x[1]*x[6] + x[0]*x[7],
        ])

        self.H = lambda x: 2 * np.array([
            [x[4], x[5], x[6], x[7], x[0], x[1], x[2], x[3]],
            [x[5], -x[4], x[7], -x[6], -x[1], x[0], -x[3], x[2]],
            [x[6], -x[7], -x[4], x[5], -x[2], x[3], x[0], -x[1]],
            [x[7], x[6], -x[5], -x[4], -x[3], -x[2], x[1], x[0]]
        ])

        self.R = np.identity(4) * 0.2
        self.Q = np.identity(8) * 0.00001

    def normalize(self):
        self.X[:4] /= np.linalg.norm(self.X[:4])
        self.X[4:] /= np.linalg.norm(self.X[4:])