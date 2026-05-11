import numpy as np

class Observer:
    def __init__(self, data_handler, init_state, init_var):
        self.data_handler = data_handler

        self.X = self.X_priori = init_state
        self.P = self.P_priori = init_var

        self.data_handler.add_result('x', self.X)
        self.data_handler.add_result('p', self.P)

        # init variables
        self.f = lambda x, u: 0
        self.F = lambda x, u: 0
        self.h = lambda x: 0
        self.H = lambda x: 0

        self.Q = lambda x: 0
        self.R = lambda x: 0

    def predict(self, u):
        C = np.array([[self.X[0]**2 + self.X[1]**2 - self.X[2]**2 - self.X[3]**2, -2*self.X[0]*self.X[3] + 2*self.X[1]*self.X[2], 2*self.X[0]*self.X[2] + 2*self.X[1]*self.X[3]], [2*self.X[0]*self.X[3] + 2*self.X[1]*self.X[2], self.X[0]**2 - self.X[1]**2 + self.X[2]**2 - self.X[3]**2, -2*self.X[0]*self.X[1] + 2*self.X[2]*self.X[3]], [-2*self.X[0]*self.X[2] + 2*self.X[1]*self.X[3], 2*self.X[0]*self.X[1] + 2*self.X[2]*self.X[3], self.X[0]**2 - self.X[1]**2 - self.X[2]**2 + self.X[3]**2]])
        g = np.array([0, 0, 9.81])
        t = C @ u[3:6] - g

        print(np.linalg.norm(t))
        if (np.linalg.norm(t) < 0):
            u[3:6] = g
            u[6:9] = g
        self.X_priori = self.f(self.X, u)
        self.P_priori = self.F(self.X, u) @ self.P @ self.F(self.X, u).T + self.Q(self.X, u)

    def update(self, Z):
        V = Z - self.h(self.X_priori)
        S = (self.H(self.X_priori) @ self.P_priori @ self.H(self.X_priori).T) + self.R(self.X_priori)
        K = self.P_priori @ self.H(self.X_priori).T @ np.linalg.inv(S)

        self.X = self.X_priori + K @ V
        self.P = (np.identity(len(self.P_priori)) - K @ self.H(self.X_priori)) @ self.P_priori
        self.data_handler.add_result('x', self.X)
        self.data_handler.add_result('z', Z)
        self.data_handler.add_result('h(x)', self.h(self.X_priori))
        self.data_handler.add_result('p', self.P)

        self.normalize()


    def normalize(self):
        pass
    
    def get_X(self):
        return (self.X_priori, self.X)
    

    def get_P(self):
        return (self.P_priori, self.P)