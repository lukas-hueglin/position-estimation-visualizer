import wx
import threading
import numpy as np

from filters import Kalman_v1
from filters import Kalman_v2
from rendering import LiveRenderer
from helpers import LiveDataHandler


def main():
    # create data handler
    data_handler = LiveDataHandler(port='COM3', deltatime=1/104) # 52 Hz

    # intitial state
    theta = np.deg2rad(0)
    #init_X = np.array([np.cos(theta/2), 0*np.sin(theta/2), 0*np.sin(theta/2), 1*np.sin(theta/2)])
    #init_P = np.identity(4) * 1e-12 # best for few samples
    init_X = np.array([np.cos(theta/2), 0*np.sin(theta/2), 0*np.sin(theta/2), 1*np.sin(theta/2), 0, 0, 0, 0, 0, 0])
    init_P = np.identity(10) * 1e-12 # best for few samples

    # create kalman filter
    observer = Kalman_v2(data_handler, init_X, init_P)

    # create app
    def create_app():
        app = wx.App()
        renderer = LiveRenderer(data_handler)
        renderer.Show()
        app.MainLoop()

    t = threading.Thread(target=create_app)
    t.start()

    try:
        while t.is_alive():
            # get measurements
            gyro, acc = data_handler.get_measurement()
            
            print(f'gx: {gyro[0]:.2f}\tgy: {gyro[1]:.2f}\tgz: {gyro[2]:.2f}\tax: {acc[0]:.2f}\tay: {acc[1]:.2f}\taz: {acc[2]:.2f}')

            # predict step
            #observer.predict(gyro)
            observer.predict(np.hstack([gyro, acc]))

            # update step
            observer.update(acc)
    except KeyboardInterrupt:
        pass

    t.join()


if __name__ == '__main__':
    main()