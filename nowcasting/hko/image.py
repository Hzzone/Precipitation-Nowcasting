# Python plugin that supports loading batch of images in parallel
import cv2
import numpy
import threading
import os
import struct
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, wait

_imread_executor_pool = ThreadPoolExecutor(max_workers=16)

class UnknownImageFormat(Exception):
    pass


def quick_imsize(file_path):
    """Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core

    Parameters
    ----------
    file_path

    Returns
    -------

    """
    size = os.path.getsize(file_path)
    with open(file_path, 'rb') as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0]) - 2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height


def cv2_read_img_resize(path, read_storage, resize_storage, frame_size, grayscale):
    if grayscale:
        read_storage[:] = cv2.imread(path, 0)
    else:
        read_storage[:] = cv2.imread(path)
    resize_storage[:] = cv2.resize(read_storage, frame_size, interpolation=cv2.INTER_LINEAR)


def cv2_read_img(path, read_storage, grayscale):
    if grayscale:
        read_storage[:] = cv2.imread(path, 0)
    else:
        read_storage[:] = cv2.imread(path)


def quick_read_frames(path_list, im_w=None, im_h=None, resize=False, frame_size=None, grayscale=True):
    """Multi-thread Frame Loader

    Parameters
    ----------
    path_list : list
    resize : bool, optional
    frame_size : None or tuple

    Returns
    -------

    """
    img_num = len(path_list)
    for i in range(img_num):
        if not os.path.exists(path_list[i]):
            raise IOError
    if im_w is None or im_h is None:
        im_w, im_h = quick_imsize(path_list[0])
    if grayscale:
        read_storage = numpy.empty((img_num, im_h, im_w), dtype=numpy.uint8)
    else:
        read_storage = numpy.empty((img_num, im_h, im_w, 3), dtype=numpy.uint8)
    if resize:
        if grayscale:
            resize_storage = numpy.empty((img_num, frame_size[0], frame_size[1]), dtype=numpy.uint8)
        else:
            resize_storage = numpy.empty((img_num, frame_size[0], frame_size[1], 3), dtype=numpy.uint8)
        if img_num == 1:
            cv2_read_img_resize(path=path_list[0], read_storage=read_storage[0],
                                resize_storage=resize_storage[0],
                                frame_size=frame_size, grayscale=grayscale)
        else:
            future_objs = []
            for i in range(img_num):
                obj = _imread_executor_pool.submit(cv2_read_img_resize,
                                                   path_list[i],
                                                   read_storage[i],
                                                   resize_storage[i], frame_size, grayscale)
                future_objs.append(obj)
            wait(future_objs)
        if grayscale:
            resize_storage = resize_storage.reshape((img_num, 1, frame_size[0], frame_size[1]))
        else:
            resize_storage = resize_storage.transpose((0, 3, 1, 2))
        return resize_storage[:, ::-1, ...]
    else:
        if img_num == 1:
            cv2_read_img(path=path_list[0], read_storage=read_storage[0], grayscale=grayscale)
        else:
            future_objs = []
            for i in range(img_num):
                obj = _imread_executor_pool.submit(cv2_read_img, path_list[i], read_storage[i], grayscale)
                future_objs.append(obj)
            wait(future_objs)
        if grayscale:
            read_storage = read_storage.reshape((img_num, 1, im_h, im_w))
        else:
            read_storage = read_storage.transpose((0, 3, 1, 2))
        return read_storage[:, ::-1, ...]