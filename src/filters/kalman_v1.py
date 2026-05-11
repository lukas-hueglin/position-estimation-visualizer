import numpy as np
from filters.observer import Observer

class Kalman_v1(Observer):
    def __init__(self, data_handler, init_state, init_var):
        super().__init__(data_handler, init_state, init_var)

        dt = data_handler.deltatime

        self.f = lambda x, u: np.array([
            x[0] + dt/2*(-x[1]*u[0] - x[2]*u[1] - x[3]*u[2]),
            x[1] + dt/2*( x[0]*u[0] - x[3]*u[1] + x[2]*u[2]),
            x[2] + dt/2*( x[3]*u[0] + x[0]*u[1] - x[1]*u[2]),
            x[3] + dt/2*(-x[2]*u[0] + x[1]*u[1] + x[0]*u[2]),
        ])

        self.F = lambda x, u: np.array([
            [1,         -dt/2 * u[0], -dt/2 * u[1], -dt/2 * u[2]],
            [dt/2*u[0],  1,            dt/2 * u[2], -dt/2 * u[1]],
            [dt/2*u[1], -dt/2 * u[2],  1,            dt/2 * u[0]],
            [dt/2*u[2],  dt/2 * u[1], -dt/2 * u[0],  1]
        ])

        g = [0, 0, 9.81]

        self.h = lambda x: 2*np.array([
             g[0]*(x[0]**2 + x[1]**2 - 0.5) + g[1]*(x[0]*x[3] + x[1]*x[2])   - g[2]*(x[0]*x[2] - x[1]*x[3]),
            -g[0]*(x[0]*x[3] - x[1]*x[2])   + g[1]*(x[0]**2 + x[2]**2 - 0.5) + g[2]*(x[0]*x[1] + x[2]*x[3]),
             g[0]*(x[0]*x[2] + x[1]*x[3])   - g[1]*(x[0]*x[1] - x[2]*x[3])   + g[2]*(x[0]**2 + x[3]**2 - 0.5)
            ])

        self.H = lambda x: 2*np.array([
            [ 2*g[0]*x[0] +   g[1]*x[3] -   g[2]*x[2], 2*g[0]*x[1] + g[1]*x[2] + g[2]*x[3],               g[1]*x[1] - g[2]*x[0],              g[1]*x[0] +   g[2]*x[1]],
            [ - g[0]*x[3] + 2*g[1]*x[0] +   g[2]*x[1],   g[0]*x[2]             + g[2]*x[0], g[0]*x[1] + 2*g[1]*x[2] + g[2]*x[3], -g[0]*x[0]             +   g[2]*x[2]],
            [   g[0]*x[2] -   g[1]*x[1] + 2*g[2]*x[0],   g[0]*x[3] - g[1]*x[0],             g[0]*x[0] +   g[1]*x[3],              g[0]*x[1] + g[1]*x[2] + 2*g[2]*x[3]]
            ])

        self.W = lambda x, u: dt/2 * np.array([
            [-x[1], -x[2], -x[3]],
            [x[0], -x[3], x[2]],
            [x[3], x[0], -x[1]],
            [-x[2], x[1], x[0]],
        ])

        self.Q = lambda x, u: self.W(x, u) @ np.array([
            [0.001745**2, 0, 0],
            [0, 0.001571**2, 0],
            [0, 0, 0.002094**2],
        ]) @ self.W(x, u).T

        self.R = lambda x: np.array([
            [0.044**2, 0, 0],
            [0, 0.05**2, 0],
            [0, 0, 0.074**2]
        ])

    def normalize(self):
        self.X /= np.linalg.norm(self.X)
