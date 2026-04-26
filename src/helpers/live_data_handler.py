import time
import serial
import numpy as np

from helpers.data_handler import DataHandler


def twos_comp_two_bytes(msb, lsb):
    a = (msb<<8) + lsb
    if a >= (256*256)//2:
        a = a - (256*256)
    return a

class LiveDataHandler(DataHandler):
    def __init__(self, port):
        super().__init__()
        self.n_samples = 3
        self.deltatime = 0.01
        self.has_started = False
        self.init = False

        radps_fullscale = 1000/180 * np.pi
        self.radps_per_LSB = radps_fullscale / 32768.0

        mps2_fullscale = 9.81 * 2
        self.mps2_per_LSB = mps2_fullscale / 32768.0

        self.serial = serial.Serial(port, baudrate=115200)
        self.init = True
        
        self.serial.write(b'uG')
        time.sleep(5)
        self.serial.reset_input_buffer()
        time.sleep(1)

    def __del__(self):
        if self.init:
            self.serial.close()
    
    def start(self):
        self.serial.write(b'l')
        self.has_started = True

    def get_measurement(self):
        if not self.has_started:
            self.start()

        bytes = self.serial.read(13)

        gyro_x = twos_comp_two_bytes(int(bytes[0]), int(bytes[1]))
        gyro_y = twos_comp_two_bytes(int(bytes[2]), int(bytes[3]))
        gyro_z = twos_comp_two_bytes(int(bytes[4]), int(bytes[5]))

        acc_x = twos_comp_two_bytes(int(bytes[6]), int(bytes[7]))
        acc_y = twos_comp_two_bytes(int(bytes[8]), int(bytes[9]))
        acc_z = twos_comp_two_bytes(int(bytes[10]), int(bytes[11]))
        
        return np.array([gyro_y, -gyro_x, gyro_z]) * self.radps_per_LSB, np.array([acc_y, -acc_x, acc_z]) * self.mps2_per_LSB
    
    def get_result(self, key):
        return self.results[key][-1]