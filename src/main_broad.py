import wx
import numpy as np

from filters import Kalman_v1
from filters import Kalman_v2
from filters import Kalman_v3
from rendering import BroadRenderer
from helpers import BroadDataHandler

def main():
    data_handler = BroadDataHandler('.\\broad\\data_hdf5\\09_undisturbed_fast_rotation_with_breaks_B.hdf5')

    init_quat, init_pos = data_handler.get_validation(0)

    init_X = np.hstack([init_quat, init_pos, np.zeros(3)])
    init_P = np.identity(10) * 1e-12 # best for few samples

    observer = Kalman_v2(data_handler, init_X, init_P)

    for t in range(1, data_handler.n_samples):
        # get measurements
        gyro, acc = data_handler.get_measurement(t)

        # predict step
        #observer.predict(gyro)
        observer.predict(np.hstack([gyro, acc]))

        # update step
        observer.update(acc)

    data_handler.plot()

    # create app
    app = wx.App()
    renderer = BroadRenderer(data_handler)
    renderer.Show()
    app.MainLoop()

if __name__ == '__main__':
    main()