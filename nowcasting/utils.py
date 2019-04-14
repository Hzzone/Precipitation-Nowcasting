import numpy as np
from torch import nn
from collections import OrderedDict
from nowcasting.config import cfg

def plot_result(writer, itera, train_result, valid_result):
    train_csi, train_hss, train_mse, train_mae, train_balanced_mse, train_balanced_mae = \
        train_result
    train_csi, train_hss, train_mse, train_mae, train_balanced_mse, train_balanced_mae = \
        np.nan_to_num(train_csi), \
        np.nan_to_num(train_hss), \
        np.nan_to_num(train_mse), \
        np.nan_to_num(train_mae), \
        np.nan_to_num(train_balanced_mse), \
        np.nan_to_num(train_balanced_mae)

    valid_csi, valid_hss, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae = \
        valid_result
    valid_csi, valid_hss, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae = \
        np.nan_to_num(valid_csi), \
        np.nan_to_num(valid_hss), \
        np.nan_to_num(valid_mse), \
        np.nan_to_num(valid_mae), \
        np.nan_to_num(valid_balanced_mse), \
        np.nan_to_num(valid_balanced_mae)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):

        writer.add_scalars("csi/{}".format(thresh), {
            "train": train_csi[:, i].mean(),
            "valid": valid_csi[:, i].mean(),
            "train_last_frame": train_csi[-1, i],
            "valid_last_frame": valid_csi[-1, i]
        }, itera)

    for i, thresh in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):

        writer.add_scalars("hss/{}".format(thresh), {
            "train": train_hss[:, i].mean(),
            "valid": valid_hss[:, i].mean(),
            "train_last_frame": train_hss[-1, i],
            "valid_last_frame": valid_hss[-1, i]
        }, itera)

    writer.add_scalars("mse", {
        "train": train_mse.mean(),
        "valid": valid_mse.mean(),
        "train_last_frame": train_mse[-1],
        "valid_last_frame": valid_mse[-1],
    }, itera)

    writer.add_scalars("mae", {
        "train": train_mae.mean(),
        "valid": valid_mae.mean(),
        "train_last_frame": train_mae[-1],
        "valid_last_frame": valid_mae[-1],
    }, itera)

    writer.add_scalars("balanced_mse", {
        "train": train_balanced_mse.mean(),
        "valid": valid_balanced_mse.mean(),
        "train_last_frame": train_balanced_mse[-1],
        "valid_last_frame": valid_balanced_mse[-1],
    }, itera)

    writer.add_scalars("balanced_mae", {
        "train": train_balanced_mae.mean(),
        "valid": valid_balanced_mae.mean(),
        "train_last_frame": train_balanced_mae[-1],
        "valid_last_frame": valid_balanced_mae[-1],
    }, itera)

def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if 'pool' in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                    padding=v[2])
            layers.append((layer_name, layer))
        elif 'deconv' in layer_name:
            transposeConv2d = nn.ConvTranspose2d(in_channels=v[0], out_channels=v[1],
                                                 kernel_size=v[2], stride=v[3],
                                                 padding=v[4])
            layers.append((layer_name, transposeConv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        elif 'conv' in layer_name:
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                               kernel_size=v[2], stride=v[3],
                               padding=v[4])
            layers.append((layer_name, conv2d))
            if 'relu' in layer_name:
                layers.append(('relu_' + layer_name, nn.ReLU(inplace=True)))
            elif 'leaky' in layer_name:
                layers.append(('leaky_' + layer_name, nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        else:
            raise NotImplementedError

    return nn.Sequential(OrderedDict(layers))

def pixel_to_dBZ(img):
    """

    Parameters
    ----------
    img : np.ndarray or float

    Returns
    -------

    """
    return img * 70.0 - 10.0

def dBZ_to_pixel(dBZ_img):
    """

    Parameters
    ----------
    dBZ_img : np.ndarray

    Returns
    -------

    """
    return np.clip((dBZ_img + 10.0) / 70.0, a_min=0.0, a_max=1.0)


def pixel_to_rainfall(img, a=58.53, b=1.56):
    """Convert the pixel values to real rainfall intensity

    Parameters
    ----------
    img : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    rainfall_intensity : np.ndarray
    """
    dBZ = pixel_to_dBZ(img)
    dBR = (dBZ - 10.0 * np.log10(a)) / b
    rainfall_intensity = np.power(10, dBR / 10.0)
    return rainfall_intensity


def rainfall_to_pixel(rainfall_intensity, a=58.53, b=1.56):
    """Convert the rainfall intensity to pixel values

    Parameters
    ----------
    rainfall_intensity : np.ndarray
    a : float32, optional
    b : float32, optional

    Returns
    -------
    pixel_vals : np.ndarray
    """
    dBR = np.log10(rainfall_intensity) * 10.0
    dBZ = dBR * b + 10.0 * np.log10(a)
    pixel_vals = (dBZ + 10.0) / 70.0
    return pixel_vals