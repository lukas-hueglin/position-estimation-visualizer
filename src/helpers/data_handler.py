import numpy as np

class DataHandler:
    def __init__(self):
        self.deltatime = 1
        self.n_samples = 1
    
        self.acc = np.array([])
        self.gyro = np.array([])
        self.mag = np.array([])

        self.results = {}

    def get_measurement(self, time):
        return self.gyro[time], self.acc[time]#np.hstack([self.acc[time], self.mag[time]])
    
    def add_result(self, key, value):
        if key not in self.results.keys():
            self.results[key] = []
        self.results[key].append(value)

    def get_result(self, key, time):
        return self.results[key][time]

    def plot(self):
        pass