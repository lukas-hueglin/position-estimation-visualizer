import wx
import threading
import numpy as np

from filters import Kalman_v1
from filters import Kalman_v2
from filters import Kalman_v3
from filters import Kalman_v4
from rendering import LiveRenderer
from helpers import LiveDataHandler

ROTATION_ONLY = False

def main():
    # create data handler
    data_handler = LiveDataHandler(port='COM3', deltatime=1/52) # 52 Hz

    theta = np.deg2rad(0)

    if ROTATION_ONLY:
        # intitial state
        init_X = np.array([np.cos(theta/2), 0*np.sin(theta/2), 0*np.sin(theta/2), 1*np.sin(theta/2)])
        init_P = np.identity(4) * 1e-12 # best for few samples
        
        # create kalman filter
        observer = Kalman_v1(data_handler, init_X, init_P)
    else:
        # intitial state
        init_X = np.array([np.cos(theta/2), 0*np.sin(theta/2), 0*np.sin(theta/2), 1*np.sin(theta/2), 0, 0, 0, 0, 0, 0])
        init_P = np.identity(10) * 1e-12 # best for few samples

        acc_old = np.zeros(3)
        q_old = np.array([np.cos(theta/2), 0*np.sin(theta/2), 0*np.sin(theta/2), 1*np.sin(theta/2)])

        # create kalman filter
        observer = Kalman_v3(data_handler, init_X, init_P)

    # create app
    def create_app():
        app = wx.App()
        renderer = LiveRenderer(data_handler, ROTATION_ONLY)
        renderer.Show()
        app.MainLoop()

    t = threading.Thread(target=create_app)
    t.start()
    try:
        while t.is_alive():
            q_temp = data_handler.get_result('x')[:4]

            # get measurements
            gyro, acc = data_handler.get_measurement()
            
            print(f'gx: {gyro[0]:.2f}\tgy: {gyro[1]:.2f}\tgz: {gyro[2]:.2f}\tax: {acc[0]:.2f}\tay: {acc[1]:.2f}\taz: {acc[2]:.2f}')

            # predict step
            if ROTATION_ONLY:
                observer.predict(gyro)
            else:
                observer.predict(np.hstack([gyro, acc, acc_old, q_old]))

            # update step
            observer.update(acc)
            
            acc_old = acc
            q_old = q_temp
    except KeyboardInterrupt:
        pass

    t.join() 


if __name__ == '__main__':
    main()