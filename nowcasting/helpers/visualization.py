import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import hsv_to_rgb
import cv2
import moviepy.editor as mpy
import numpy as np
from nowcasting.helpers.gifmaker import save_gif

def flow_to_img(flow_dat, max_displacement=None):
    """Convert optical flow data to HSV images

    Parameters
    ----------
    flow_dat : np.ndarray
        Shape: (seq_len, 2, H, W)
    max_displacement : float or None

    Returns
    -------
    rgb_dat : np.ndarray
        Shape: (seq_len, 3, H, W)
    """
    assert flow_dat.ndim == 4
    flow_scale = np.square(flow_dat).sum(axis=1, keepdims=True)
    flow_x = flow_dat[:, :1, :, :]
    flow_y = flow_dat[:, 1:, :, :]
    flow_angle = np.arctan2(flow_y, flow_x)
    flow_angle[flow_angle < 0] += np.pi * 2
    v = np.ones((flow_dat.shape[0], 1, flow_dat.shape[2], flow_dat.shape[3]),
                dtype=np.float32)
    if max_displacement is None:
        flow_scale_max = np.sqrt(flow_scale.max())
    else:
        flow_scale_max = max_displacement
    h = flow_angle / (2 * np.pi)
    s = np.sqrt(flow_scale) / flow_scale_max

    hsv_dat = np.concatenate((h, s, v), axis=1)
    rgb_dat = hsv_to_rgb(hsv_dat.transpose((0, 2, 3, 1))).transpose((0, 3, 1, 2))
    return rgb_dat


def _ax_imshow(ax, im, **kwargs):
    assert im.ndim == 3 or im.ndim == 2
    if im.ndim == 2:
        ax.imshow(im, **kwargs)
        ax.set_axis_off()
    else:
        if im.shape[0] == 1:
            ax.imshow(im[0, :, :], **kwargs)
            ax.set_axis_off()
        elif im.shape[0] == 3:
            ax.imshow(im.transpose((1, 2, 0)), **kwargs)
            ax.set_axis_off()
        else:
            raise NotImplementedError
    ax.set_adjustable('box-forced')
    ax.autoscale(False)


def get_color_flow_legend_image(size=50):
    U, V = np.meshgrid(np.arange(-size, size + 1, dtype=np.float32),
                        np.arange(-size, size + 1, dtype=np.float32))
    flow_scale = np.sqrt(U**2 + V**2)
    flow_angle = np.arctan2(V, U)
    flow_angle[flow_angle < 0] += np.pi * 2
    max_flow_scale = float(size) * np.sqrt(2)
    h = flow_angle / (2 * np.pi)
    s = flow_scale / max_flow_scale
    v = np.ones((size * 2 + 1, size * 2 + 1),
                dtype=np.float32)
    hsv_dat = np.concatenate((h.reshape((1, size * 2 + 1, size * 2 + 1)),
                              s.reshape((1, size * 2 + 1, size * 2 + 1)),
                              v.reshape((1, size * 2 + 1, size * 2 + 1))), axis=0)
    rgb_dat = hsv_to_rgb(hsv_dat.transpose((1, 2, 0))).transpose((2, 0, 1))
    a = np.ones((1, size * 2 + 1, size * 2 + 1), dtype=np.float32)
    rgb_dat[:, flow_scale > max_flow_scale] = 0
    a[:, flow_scale > max_flow_scale] = 0
    rgba_dat = np.concatenate((rgb_dat, a), axis=0)
    return rgba_dat


def save_hko_gif(im_dat, save_path):
    """Save the HKO images to gif

    Parameters
    ----------
    im_dat : np.ndarray
        Shape: (seqlen, H, W)
    save_path : str
    Returns
    -------
    """
    assert im_dat.ndim == 3
    save_gif(im_dat, fname=save_path)
    return


def merge_rgba_cv2(front_img, back_img):
    """Merge the front image with the background image using the `Painter's algorithm`

    Parameters
    ----------
    front_img : np.ndarray
    back_img : np.ndarray

    Returns
    -------
    result_img : np.ndarray
    """
    assert front_img.shape == back_img.shape
    if front_img.dtype == np.uint8:
        front_img = front_img.astype(np.float32) / 255.0
    if back_img.dtype == np.uint8:
        back_img =  back_img.astype(np.float32) / 255.0
    result_img = np.zeros(front_img.shape, dtype=np.float32)
    result_img[:, :, 3] = front_img[:, :, 3] + back_img[:, :, 3] * (1 - front_img[:, :, 3])
    result_img[:, :, :3] = (front_img[:, :, :3] * front_img[:, :, 3:] +
                            back_img[:, :, :3] * back_img[:, :, 3:] * (1 - front_img[:, :, 3:])) /\
                           result_img[:, :, 3:]
    result_img = (result_img * 255.0).astype(np.uint8)
    return result_img


def save_hko_movie(im_dat, datetime_list, mask_dat=None, save_path="hko.mp4", masked=False,
                   fps=5, prediction_start=None):
    """Save the HKO images to a video file
    
    Parameters
    ----------
    im_dat : np.ndarray
        Shape : (seq_len, H, W)
    datetime_list : list
        list of datetimes
    mask_dat : np.ndarray or None
        Shape : (seq_len, H, W)
    save_path : str
    masked : bool
        whether the mask the inputs when saving the image
    fps : float
        the fps of the saved movie
    prediction_start : int or None
        The starting point of the prediction
    """
    from nowcasting.config import cfg
    central_region = cfg.HKO.EVALUATION.CENTRAL_REGION
    seq_len, height, width = im_dat.shape
    display_im_dat = []
    mask_color = np.array((0, 170, 160, 150), dtype=np.float32) / 255.0
    if im_dat.dtype == np.float32:
        im_dat = (im_dat * 255).astype(np.uint8)
    assert im_dat.dtype==np.uint8
    for i in range(im_dat.shape[0]):
        if not masked:
            color_im_dat = cv2.cvtColor(im_dat[i], cv2.COLOR_GRAY2RGBA)
            im = color_im_dat
        else:
            im = im_dat[i] * mask_dat[i]
            assert im.dtype==np.uint8
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGBA)
            # Uncomment the following code to add transparency to the masks
            # color_im_dat = cv2.cvtColor(im_dat[i], cv2.COLOR_GRAY2RGBA)
            # mask_im_dat = mask_color.reshape((1, 1, 4)) * np.expand_dims(1 - mask_dat[i], axis=2)
            # im = merge_rgba_cv2(front_img=mask_im_dat, back_img=color_im_dat)
        if prediction_start is not None and i >= prediction_start:
            cv2.putText(im, text=datetime_list[i].strftime('%Y/%m/%d %H:%M'),
                        org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                        color=(255, 0, 0, 0))
        else:
            cv2.putText(im, text=datetime_list[i].strftime('%Y/%m/%d %H:%M'),
                        org=(0, 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4,
                        color=(255, 255, 255, 0))
        cv2.rectangle(im,
                      pt1=(central_region[0], central_region[1]),
                      pt2=(central_region[2], central_region[3]),
                      color=(0, 255, 0, 0))
        display_im_dat.append(im)
    clip = mpy.ImageSequenceClip(display_im_dat, with_mask=False, fps=fps)
    clip.write_videofile(save_path, audio=False, verbose=False, threads=4)
