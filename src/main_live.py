import wx
import threading
import numpy as np

from filters import Kalman_v2
from rendering import LiveRenderer
from helpers import LiveDataHandler


def main():
    data_handler = LiveDataHandler(port='COM102')

    theta = np.deg2rad(0)
    init_X = np.array([np.cos(theta/2), 0*np.sin(theta/2), 0*np.sin(theta/2), 1*np.sin(theta/2)])

    init_P = np.identity(4) * 1e-12 # best for few samples

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
        while True:
            # get measurements
            u, Z = data_handler.get_measurement()

            print(f'{u[0]:.2f}  {u[1]:.2f}  {u[2]:.2f}')

            # predict step
            observer.predict(u)

            # update step
            observer.update(Z)
    except KeyboardInterrupt:
        pass

    t.join()


if __name__ == '__main__':
    main()