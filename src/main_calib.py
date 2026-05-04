from helpers import LiveDataHandler
import numpy as np

def main():
    # create data handler
    data_handler = LiveDataHandler(port='COM3', deltatime=1/52) # 52 Hz

    gyro_noise = np.zeros((1000, 3))
    acc_noise = np.zeros((1000, 3))

    for i in range(0, 1000):
        gyro_noise[i], acc_noise[i] = data_handler.get_measurement()

    print("Gyro noise: ", np.std(gyro_noise, axis = 0, ddof=1))
    print("Acc noise: ", np.std(acc_noise, axis = 0, ddof=1))

if __name__ == '__main__':
    main()