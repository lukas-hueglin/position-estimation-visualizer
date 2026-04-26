import numpy as np

class Observer:
    def __init__(self, data_handler, init_state, init_var, use_acc=True):
        self.use_acc = use_acc
        self.data_handler = data_handler

        self.X = self.X_priori = init_state
        self.P = self.P_priori = init_var

        if use_acc:
            self.data_handler.add_result('x', self.X)
            self.data_handler.add_result('p', self.P)
        else:
            self.data_handler.add_result('x_wo_acc', self.X)


        # init variables
        self.f = lambda x, u: 0
        self.F = lambda x, u: 0
        self.h = lambda x: 0
        self.H = lambda x: 0

        self.Q = lambda x: 0
        self.R = lambda x: 0

    def predict(self, u):
        self.X_priori = self.f(self.X, u)
        self.P_priori = self.F(self.X, u) @ self.P @ self.F(self.X, u).T + self.Q(self.X)

    def update(self, Z):
        V = Z - self.h(self.X_priori)
        S = (self.H(self.X_priori) @ self.P_priori @ self.H(self.X_priori).T) + self.R(self.X_priori)
        K = self.P_priori @ self.H(self.X_priori).T @ np.linalg.inv(S)

        if self.use_acc:
            self.X = self.X_priori + K @ V
            self.P = (np.identity(len(self.P_priori)) - K @ self.H(self.X_priori)) @ self.P_priori
            self.data_handler.add_result('x', self.X)
            self.data_handler.add_result('z', Z)
            self.data_handler.add_result('h(x)', self.h(self.X_priori))
            self.data_handler.add_result('p', self.P)
        else:
            self.X = self.X_priori
            self.P = self.P_priori

            self.data_handler.add_result('x_wo_acc', self.X)

        self.normalize()


    def normalize(self):
        pass

    
    def get_X(self):
        return (self.X_priori, self.X)
    

    def get_P(self):
        return (self.P_priori, self.P)