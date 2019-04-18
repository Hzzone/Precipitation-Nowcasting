# File to deal read and write the .mask extensions
import zlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, wait

_executor_pool = ThreadPoolExecutor(max_workers=16)

def read_mask_file(filepath, out=None):
    """Load mask file to numpy array

    Parameters
    ----------
    filepath : str
    out : np.ndarray

    Returns
    -------

    """
    f = open(filepath, 'rb')
    dat = zlib.decompress(f.read())
    if out is None:
        return np.frombuffer(dat, dtype=bool).reshape((480, 480))
    out[:] = np.frombuffer(dat, dtype=bool).reshape((480, 480))
    f.close()


def save_mask_file(npy_mask, filepath):
    compressed_data = zlib.compress(npy_mask.tobytes(), 2)
    f = open(filepath, "wb")
    f.write(compressed_data)
    f.close()


def quick_read_masks(path_list):
    num = len(path_list)
    read_storage = np.empty((num, 480, 480), dtype=np.bool)
    # for i in range(num):
    #     read_storage[i] = read_mask_file(path_list[i])
    future_objs = []
    for i in range(num):
        obj = _executor_pool.submit(read_mask_file, path_list[i], read_storage[i])
        future_objs.append(obj)
    wait(future_objs)
    ret = read_storage.reshape((num, 1, 480, 480))
    return ret
