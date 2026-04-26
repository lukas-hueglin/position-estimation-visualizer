import numpy as np
import h5py as hdf

import matplotlib.pyplot as plt

from helpers.quat_math import *
from helpers.data_handler import DataHandler

class BroadDataHandler(DataHandler):
    def __init__(self, path):
        super().__init__()

        f = hdf.File(path, 'r')
        sampling_rate = f.attrs['sampling_rate']
        duration = 15

        self.deltatime = 1 / sampling_rate
        self.n_samples = int(duration * sampling_rate)
        real_n_samples = len(f['imu_acc'])

        self.acc = np.array(f['imu_acc'])[(real_n_samples-self.n_samples)//2: (real_n_samples+self.n_samples)//2]
        self.gyro = np.array(f['imu_gyr'])[(real_n_samples-self.n_samples)//2: (real_n_samples+self.n_samples)//2]
        self.mag = np.array(f['imu_mag'])[(real_n_samples-self.n_samples)//2: (real_n_samples+self.n_samples)//2]
        self.valid = np.array(f['opt_quat'])[(real_n_samples-self.n_samples)//2: (real_n_samples+self.n_samples)//2]

    def get_measurement(self, time):
        return self.gyro[time], self.acc[time]

    def get_validation(self, time):
        return self.valid[time]

    def plot(self):
        quat = [['quatW'], ['quatX'], ['quatY'], ['quatZ']]
        layout = [['error', 'error'], [quat, 'variance']]
        kw = {
            'error': {
                'title': 'Error',
                'xlabel': 'time (s)',
                'ylabel': 'angle (rad)'
            },
            'quatW': {
                'title': 'Quaternion',
                'xticklabels': [],
                'yticklabels': [],
                'ylim':(-1, 1)

            },
            ('quatX', 'quatY'): {
                'xticklabels': [],
                'yticklabels': [],
                'ylim':(-1, 1)
            },
            'quatZ': {
                'xlabel': 'time (s)',
                'yticklabels': [],
                'ylim':(-1, 1)
            },
            'variance': {
                'title': 'Covariance',
                'xlabel': 'time (s)',
                'ylabel': 'variance'
            },
        }

        colors = [['gray', 'black'],
                  ['lightcoral', 'firebrick'],
                  ['forestgreen', 'darkgreen'],
                  ['cornflowerblue', 'mediumblue']]
        
        fig, axd = plt.subplot_mosaic(layout, per_subplot_kw=kw, layout='constrained', gridspec_kw={
            'bottom': 0.5,
            'top': 0.9,
            'left': 0.1,
            'right': 0.9,
            'wspace': 0.1,
            'hspace': 0.1,
        },)

        # get timeline
        time = np.linspace(0, self.n_samples*self.deltatime, self.n_samples)
        
        # plot error
        error = np.array(list(map(angle_between_quat, self.valid, self.results['x'])))
        axd['error'].plot(time, error)
        ax_deg = axd['error'].twinx()
        ax_deg.plot(time, 180*error/np.pi)
        ax_deg.set_ylabel('angle (°)')

        # plot quaternion of validation and real
        for i, s in enumerate(['W', 'X', 'Y', 'Z']):
            axd['quat'+s].plot(time, self.valid[:, i], label='Validation', color=colors[i][0])
            axd['quat'+s].plot(time, np.array(self.results['x'])[:, i], label='Kalman', color=colors[i][1])

        # plot covariances
        for i in range(4):
            axd['variance'].plot(time, np.array(self.results['p'])[:, i, i])
            axd['variance'].set_yscale('log')

        plt.show()