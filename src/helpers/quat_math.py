import numpy as np

def angle_between_quat(quatA, quatB):
    return np.arccos(quat_mul(quatA, quat_conj(quatB))[0])

def quat_mul(quatA, quatB):
    w0, x0, y0, z0 = quatA
    w1, x1, y1, z1 = quatB
    return np.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
        ])

def quat_conj(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

def quat_transform_point(quat, point):
    return quat_mul(quat_mul(quat, [0, *point]), quat_conj(quat))[1:]