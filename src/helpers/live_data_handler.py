import time
import serial
import numpy as np

from helpers.data_handler import DataHandler


def ieee_754_conversion(n, sgn_len=1, exp_len=8, mant_len=23):
    """
    Converts an arbitrary precision Floating Point number.
    Note: Since the calculations made by python inherently use floats, the accuracy is poor at high precision.
    :param n: An unsigned integer of length `sgn_len` + `exp_len` + `mant_len` to be decoded as a float
    :param sgn_len: number of sign bits
    :param exp_len: number of exponent bits
    :param mant_len: number of mantissa bits
    :return: IEEE 754 Floating Point representation of the number `n`
    """
    if n >= 2 ** (sgn_len + exp_len + mant_len):
        raise ValueError("Number n is longer than prescribed parameters allows")

    sign = (n & (2 ** sgn_len - 1) * (2 ** (exp_len + mant_len))) >> (exp_len + mant_len)
    exponent_raw = (n & ((2 ** exp_len - 1) * (2 ** mant_len))) >> mant_len
    mantissa = n & (2 ** mant_len - 1)

    sign_mult = 1
    if sign == 1:
        sign_mult = -1

    if exponent_raw == 2 ** exp_len - 1:  # Could be Inf or NaN
        if mantissa == 2 ** mant_len - 1:
            return np.nan  # NaN

        return sign_mult * np.inf  # Inf

    exponent = exponent_raw - (2 ** (exp_len - 1) - 1)

    if exponent_raw == 0:
        mant_mult = 0  # Gradual Underflow
    else:
        mant_mult = 1

    for b in range(mant_len - 1, -1, -1):
        if mantissa & (2 ** b):
            mant_mult += 1 / (2 ** (mant_len - b))

    return sign_mult * (2 ** exponent) * mant_mult

class LiveDataHandler(DataHandler):
    def __init__(self, port):
        super().__init__()
        self.n_samples = 3
        self.deltatime = 0.01
        self.has_started = True # always started
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

        # wait on gyro header
        self.serial.read_until(expected=b'g')

        # read gyro data and convert
        bytes = self.serial.read(12)

        gyro_x = ieee_754_conversion(bytes[3] << 24 | bytes[2] << 16 | bytes[1] << 8 | bytes[0])
        gyro_y = ieee_754_conversion(bytes[7] << 24 | bytes[6] << 16 | bytes[5] << 8 | bytes[4])
        gyro_z = ieee_754_conversion(bytes[11] << 24 | bytes[10] << 16 | bytes[9] << 8 | bytes[8])

        # wait on acc header
        self.serial.read_until(expected=b'a')

        # read acc data and convert
        bytes = self.serial.read(12)

        acc_x = ieee_754_conversion(bytes[3] << 24 | bytes[2] << 16 | bytes[1] << 8 | bytes[0])
        acc_y = ieee_754_conversion(bytes[7] << 24 | bytes[6] << 16 | bytes[5] << 8 | bytes[4])
        acc_z = ieee_754_conversion(bytes[11] << 24 | bytes[10] << 16 | bytes[9] << 8 | bytes[8])
        
        return 2*np.array([gyro_y, -gyro_x, gyro_z]), np.array([acc_y, acc_x, acc_z])
    
    def get_result(self, key):
        return self.results[key][-1]