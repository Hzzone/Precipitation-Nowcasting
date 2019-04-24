import ctypes
import os
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor, wait

_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__)))
_VALID_DLL_PATH = [os.path.join(_BASE_PATH, '..', 'build', 'Release', 'varflow.dll'),
                   os.path.join(_BASE_PATH, '..', 'build', 'libvarflow.so')]

_VARFLOW_DLL_PATH = None
for p in _VALID_DLL_PATH:
    if os.path.exists(p):
        _VARFLOW_DLL_PATH = p
        break
if _VARFLOW_DLL_PATH is None:
    raise RuntimeError("DLL not found! Valid PATH=%s" %(_VALID_DLL_PATH))
_CDLL = ctypes.cdll.LoadLibrary(_VARFLOW_DLL_PATH)

class VarFlowFactory(object):
    def __init__(self, max_level, start_level, n1, n2, rho, alpha, sigma):
        self._max_level = max_level
        self._start_level = start_level
        self._n1 = n1
        self._n2 = n2
        self._rho = rho
        self._alpha = alpha
        self._sigma = sigma
        self._varflow_executor_pool = ThreadPoolExecutor(max_workers=16)

    def calc_flow(self, I1, I2):
        """

        Parameters
        ----------
        I1 : np.ndarray
            Shape: (H, W)
        I2 : np.ndarray
            Shape: (H, W)
        Returns
        -------
        velocity : np.ndarray
            Shape: (2, H, W)
            The channel dimension will be flow_x, flow_y
        """
        if I1.dtype == np.float32:
            I1 = (I1 * 255).astype(np.uint8)
        else:
            I1 = I1.astype(np.uint8)
        if I2.dtype == np.float32:
            I2 = (I2 * 255).astype(np.uint8)
        else:
            I2 = I2.astype(np.uint8)
        assert I1.ndim == 2 and I2.ndim == 2
        assert I1.shape == I2.shape
        np.ascontiguousarray(I1)
        np.ascontiguousarray(I2)
        height, width = I1.shape
        velocity = np.zeros((2,) + I1.shape, dtype=np.float32)
        self._base_varflow_call(velocity=velocity, I1=I1, I2=I2, width=width, height=height)
        return velocity

    def _base_varflow_call(self, velocity, I1, I2, width, height):
        _CDLL.varflow(ctypes.c_int32(width),
                      ctypes.c_int32(height),
                      ctypes.c_int32(self._max_level),
                      ctypes.c_int32(self._start_level),
                      ctypes.c_int32(self._n1),
                      ctypes.c_int32(self._n2),
                      ctypes.c_float(self._rho),
                      ctypes.c_float(self._alpha),
                      ctypes.c_float(self._sigma),
                      velocity[0].ctypes.data_as(ctypes.c_void_p),
                      velocity[1].ctypes.data_as(ctypes.c_void_p),
                      I1.ctypes.data_as(ctypes.c_void_p),
                      I2.ctypes.data_as(ctypes.c_void_p))

    def batch_calc_flow(self, I1, I2):
        """Calculate the optical flow from two

        Parameters
        ----------
        I1 : np.ndarray
            Shape: (batch_size, H, W)
        I2 : np.ndarray
            Shape: (batch_size, H, W)
        Returns
        -------
        velocity : np.ndarray
            Shape: (batch_size, 2, H, W)
            The channel dimension will be flow_x, flow_y
        """
        if I1.dtype == np.float32:
            I1 = (I1 * 255).astype(np.uint8)
        else:
            I1 = I1.astype(np.uint8)
        if I2.dtype == np.float32:
            I2 = (I2 * 255).astype(np.uint8)
        else:
            I2 = I2.astype(np.uint8)
        np.ascontiguousarray(I1)
        np.ascontiguousarray(I2)
        assert I1.ndim == 3 and I2.ndim == 3
        assert I1.shape == I2.shape
        batch_size, height, width = I1.shape
        velocity = np.zeros((batch_size, 2, height, width), dtype=np.float32)
        future_objs = []
        for i in range(batch_size):
            obj = self._varflow_executor_pool.submit(
                self._base_varflow_call, velocity[i], I1[i], I2[i], width, height)
            future_objs.append(obj)
        wait(future_objs)
        return velocity


if __name__ == '__main__':
    I1 = cv2.imread('../Data/yos_img_08.jpg', 0)
    I2 = cv2.imread('../Data/yos_img_09.jpg', 0)
    varflow_factory = VarFlowFactory(max_level=4, start_level=0, n1=2, n2=2, rho=2.8, alpha=1400,
                                     sigma=1.5)
    velocity = varflow_factory.calc_flow(I1=I1, I2=I2)
    batch_I1 = np.concatenate([I1.reshape((1,) + I1.shape), I2.reshape((1,) + I2.shape)], axis=0)
    batch_I2 = np.concatenate([I2.reshape((1,) + I2.shape), I1.reshape((1,) + I1.shape)], axis=0)
    batch_velocity = varflow_factory.batch_calc_flow(I1=batch_I1, I2=batch_I2)
    velocity = batch_velocity[0]
    import matplotlib.pyplot as plt
    Q = plt.quiver(velocity[0, ::5, ::5], velocity[1, ::5, ::5])
    qk = plt.quiverkey(Q, 0.5, 0.98, 2, r'$2 \frac{m}{s}$', labelpos='W',
                       fontproperties={'weight': 'bold'})
    plt.gca().invert_yaxis()
    l, r, b, t = plt.axis()
    dx, dy = r - l, t - b
    plt.axis([l - 0.05*dx, r + 0.05*dx, b - 0.05*dy, t + 0.05*dy])
    plt.title('Minimal arguments, no kwargs')
    plt.show()