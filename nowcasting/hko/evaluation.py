try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
import logging
import os
from collections import namedtuple
from nowcasting.config import cfg
from nowcasting.hko.dataloader import get_exclude_mask
from nowcasting.helpers.msssim import _SSIMForMultiScale
from nowcasting.utils import *



def get_hit_miss_counts(prediction, truth, mask=None, thresholds=None, sum_batch=False):
    """This function calculates the overall hits and misses for the prediction, which could be used
    to get the skill scores and threat scores:


    This function assumes the input, i.e, prediction and truth are 3-dim tensors, (timestep, row, col)
    and all inputs should be between 0~1

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use
    thresholds : list or tuple

    Returns
    -------
    hits : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        TP
    misses : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        FN
    false_alarms : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        FP
    correct_negatives : np.ndarray
        (seq_len, len(thresholds)) or (seq_len, batch_size, len(thresholds))
        TN
    """
    if thresholds is None:
        thresholds = cfg.HKO.EVALUATION.THRESHOLDS
    assert 5 == prediction.ndim
    assert 5 == truth.ndim
    assert prediction.shape == truth.shape
    assert prediction.shape[2] == 1
    thresholds = rainfall_to_pixel(np.array(thresholds,
                                            dtype=np.float32)
                                   .reshape((1, 1, len(thresholds), 1, 1)))
    bpred = (prediction >= thresholds)
    btruth = (truth >= thresholds)
    bpred_n = np.logical_not(bpred)
    btruth_n = np.logical_not(btruth)
    if sum_batch:
        summation_axis = (1, 3, 4)
    else:
        summation_axis = (3, 4)
    if mask is None:
        hits = np.logical_and(bpred, btruth).sum(axis=summation_axis)
        misses = np.logical_and(bpred_n, btruth).sum(axis=summation_axis)
        false_alarms = np.logical_and(bpred, btruth_n).sum(axis=summation_axis)
        correct_negatives = np.logical_and(bpred_n, btruth_n).sum(axis=summation_axis)
    else:
        hits = np.logical_and(np.logical_and(bpred, btruth), mask)\
            .sum(axis=summation_axis)
        misses = np.logical_and(np.logical_and(bpred_n, btruth), mask)\
            .sum(axis=summation_axis)
        false_alarms = np.logical_and(np.logical_and(bpred, btruth_n), mask)\
            .sum(axis=summation_axis)
        correct_negatives = np.logical_and(np.logical_and(bpred_n, btruth_n), mask)\
            .sum(axis=summation_axis)
    return hits, misses, false_alarms, correct_negatives


def get_correlation(prediction, truth):
    """

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------

    """
    assert truth.shape == prediction.shape
    assert 5 == prediction.ndim
    assert prediction.shape[2] == 1
    eps = 1E-12
    ret = (prediction * truth).sum(axis=(3, 4)) / (
        np.sqrt(np.square(prediction).sum(axis=(3, 4))) * np.sqrt(np.square(truth).sum(axis=(3, 4))) + eps)
    ret = ret.sum(axis=(1, 2))
    return ret


def get_rainfall_mse(prediction, truth):
    ret = np.square(pixel_to_rainfall(prediction) - pixel_to_rainfall(truth)).mean(axis=(2, 3))
    ret = ret.sum(axis=1)
    return ret


def get_PSNR(prediction, truth):
    """Peak Signal Noise Ratio

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    mse = np.square(prediction - truth).mean(axis=(2, 3, 4))
    ret = 10.0 * np.log10(1.0 / mse)
    ret = ret.sum(axis=1)
    return ret


def get_SSIM(prediction, truth):
    """Calculate the SSIM score following
    [TIP2004] Image Quality Assessment: From Error Visibility to Structural Similarity

    Same functionality as
    https://github.com/coupriec/VideoPredictionICLR2016/blob/master/image_error_measures.lua#L50-L75

    We use nowcasting.helpers.msssim, which is borrowed from Tensorflow to do the evaluation

    Parameters
    ----------
    prediction : np.ndarray
    truth : np.ndarray

    Returns
    -------
    ret : np.ndarray
    """
    assert truth.shape == prediction.shape
    assert 5 == prediction.ndim
    assert prediction.shape[2] == 1
    seq_len = prediction.shape[0]
    batch_size = prediction.shape[1]
    prediction = prediction.reshape((prediction.shape[0] * prediction.shape[1],
                                     prediction.shape[3], prediction.shape[4], 1))
    truth = truth.reshape((truth.shape[0] * truth.shape[1],
                           truth.shape[3], truth.shape[4], 1))
    ssim, cs = _SSIMForMultiScale(img1=prediction, img2=truth, max_val=1.0)
    print(ssim.shape)
    ret = ssim.reshape((seq_len, batch_size)).sum(axis=1)
    return ret


def get_GDL(prediction, truth, mask, sum_batch=False):
    """Calculate the masked gradient difference loss

    Parameters
    ----------
    prediction : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    truth : np.ndarray
        Shape: (seq_len, batch_size, 1, height, width)
    mask : np.ndarray or None
        Shape: (seq_len, batch_size, 1, height, width)
        0 --> not use
        1 --> use

    Returns
    -------
    gdl : np.ndarray
        Shape: (seq_len,) or (seq_len, batch_size)
    """
    prediction_diff_h = np.abs(np.diff(prediction, axis=3))
    prediction_diff_w = np.abs(np.diff(prediction, axis=4))
    gt_diff_h = np.abs(np.diff(truth, axis=3))
    gt_diff_w = np.abs(np.diff(truth, axis=4))
    mask_h = mask[:, :, :, :-1, :] * mask[:, :, :, 1:, :]
    mask_w = mask[:, :, :, :, :-1] * mask[:, :, :, :, 1:]
    gd_h = np.abs(prediction_diff_h - gt_diff_h)
    gd_w = np.abs(prediction_diff_w - gt_diff_w)
    gd_h[:] *= mask_h
    gd_w[:] *= mask_w
    summation_axis = (1, 2, 3, 4) if sum_batch else (2, 3, 4)
    gdl = np.sum(gd_h, axis=summation_axis) + np.sum(gd_w, axis=summation_axis)
    return gdl


def get_balancing_weights(data, mask, base_balancing_weights=None, thresholds=None):
    if thresholds is None:
        thresholds = cfg.HKO.EVALUATION.THRESHOLDS
    if base_balancing_weights is None:
        base_balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
    thresholds = rainfall_to_pixel(np.array(thresholds, dtype=np.float32)
                                   .reshape((1, 1, 1, 1, 1, len(thresholds))))
    weights = np.ones_like(data) * base_balancing_weights[0]
    threshold_mask = np.expand_dims(data, axis=5) >= thresholds
    base_weights = np.diff(np.array(base_balancing_weights, dtype=np.float32))\
        .reshape((1, 1, 1, 1, 1, len(base_balancing_weights) - 1))
    weights += (threshold_mask * base_weights).sum(axis=-1)
    weights *= mask
    return weights


try:
    from nowcasting.hko.numba_accelerated import get_GDL_numba, get_hit_miss_counts_numba,\
        get_balancing_weights_numba
except:
    # get_GDL_numba = get_GDL
    # get_hit_miss_counts_numba = get_hit_miss_counts
    # get_balancing_weights_numba = get_balancing_weights
    # print("Numba has not been installed correctly!")
    raise ImportError("Numba has not been installed correctly!")

class HKOEvaluation(object):
    def __init__(self, seq_len, use_central, no_ssim=True, threholds=None,
                 central_region=None):
        if central_region is None:
            central_region = cfg.HKO.EVALUATION.CENTRAL_REGION
        self._thresholds = cfg.HKO.EVALUATION.THRESHOLDS if threholds is None else threholds
        self._seq_len = seq_len
        self._no_ssim = no_ssim
        self._use_central = use_central
        self._central_region = central_region
        self._exclude_mask = get_exclude_mask()
        self.begin()

    def begin(self):
        self._total_hits = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_misses = np.zeros((self._seq_len, len(self._thresholds)),  dtype=np.int)
        self._total_false_alarms = np.zeros((self._seq_len, len(self._thresholds)), dtype=np.int)
        self._total_correct_negatives = np.zeros((self._seq_len, len(self._thresholds)),
                                                 dtype=np.int)
        self._mse = np.zeros((self._seq_len, ), dtype=np.float32)
        self._mae = np.zeros((self._seq_len, ), dtype=np.float32)
        self._balanced_mse = np.zeros((self._seq_len, ), dtype=np.float32)
        self._balanced_mae = np.zeros((self._seq_len,), dtype=np.float32)
        self._gdl = np.zeros((self._seq_len,), dtype=np.float32)
        self._ssim = np.zeros((self._seq_len,), dtype=np.float32)
        self._datetime_dict = {}
        self._total_batch_num = 0

    def clear_all(self):
        self._total_hits[:] = 0
        self._total_misses[:] = 0
        self._total_false_alarms[:] = 0
        self._total_correct_negatives[:] = 0
        self._mse[:] = 0
        self._mae[:] = 0
        self._gdl[:] = 0
        self._ssim[:] = 0
        self._total_batch_num = 0
        self._balanced_mse[:] = 0
        self._balanced_mae[:] = 0

    def update(self, gt, pred, mask, start_datetimes=None):
        """

        Parameters
        ----------
        gt : np.ndarray
        pred : np.ndarray
        mask : np.ndarray
            0 indicates not use and 1 indicates that the location will be taken into account
        start_datetimes : list
            The starting datetimes of all the testing instances

        Returns
        -------

        """
        if start_datetimes is not None:
            batch_size = len(start_datetimes)
            assert gt.shape[1] == batch_size
        else:
            batch_size = gt.shape[1]
        assert gt.shape[0] == self._seq_len
        assert gt.shape == pred.shape
        assert gt.shape == mask.shape

        if self._use_central:
            # Crop the central regions for evaluation
            pred = pred[:, :, :,
                        self._central_region[1]:self._central_region[3],
                        self._central_region[0]:self._central_region[2]]
            gt = gt[:, :, :,
                    self._central_region[1]:self._central_region[3],
                    self._central_region[0]:self._central_region[2]]
            mask = mask[:, :, :,
                        self._central_region[1]:self._central_region[3],
                        self._central_region[0]:self._central_region[2]]
        self._total_batch_num += batch_size
        #TODO Save all the mse, mae, gdl, hits, misses, false_alarms and correct_negatives
        mse = (mask * np.square(pred - gt)).sum(axis=(2, 3, 4))
        mae = (mask * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        weights = get_balancing_weights_numba(data=gt, mask=mask,
                                              base_balancing_weights=cfg.HKO.EVALUATION.BALANCING_WEIGHTS,
                                              thresholds=self._thresholds)
        ## <2, <5, ... 不同权值的 MSE.
        # S*B*1*H*W
        balanced_mse = (weights * np.square(pred - gt)).sum(axis=(2, 3, 4))
        balanced_mae = (weights * np.abs(pred - gt)).sum(axis=(2, 3, 4))
        gdl = get_GDL_numba(prediction=pred, truth=gt, mask=mask)
        self._mse += mse.sum(axis=1)
        self._mae += mae.sum(axis=1)
        self._balanced_mse += balanced_mse.sum(axis=1)
        self._balanced_mae += balanced_mae.sum(axis=1)
        self._gdl += gdl.sum(axis=1)
        if not self._no_ssim:
            raise NotImplementedError
            # self._ssim += get_SSIM(prediction=pred, truth=gt)
        hits, misses, false_alarms, correct_negatives = \
            get_hit_miss_counts_numba(prediction=pred, truth=gt, mask=mask,
                                      thresholds=self._thresholds)
        self._total_hits += hits.sum(axis=1)
        self._total_misses += misses.sum(axis=1)
        self._total_false_alarms += false_alarms.sum(axis=1)
        self._total_correct_negatives += correct_negatives.sum(axis=1)

    def calculate_f1_score(self):
        '''
        计算每个分段的 precision, recall 和 f1 score.
        :return:
        '''
        # a: TP, b: TN, c: FP, d: FN
        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        precision = a / (a + c)
        recall = a / (a + d)
        return precision, recall, (2*precision*recall)/(precision+recall)

    def calculate_stat(self):
        """The following measurements will be used to measure the score of the forecaster

        See Also
        [Weather and Forecasting 2010] Equitability Revisited: Why the "Equitable Threat Score" Is Not Equitable
        http://www.wxonline.info/topics/verif2.html

        We will denote
        (a b    (hits       false alarms
         c d) =  misses   correct negatives)

        We will report the
        POD = a / (a + c)
        FAR = b / (a + b)
        CSI = a / (a + b + c)
        Heidke Skill Score (HSS) = 2(ad - bc) / ((a+c) (c+d) + (a+b)(b+d))
        Gilbert Skill Score (GSS) = HSS / (2 - HSS), also known as the Equitable Threat Score
            HSS = 2 * GSS / (GSS + 1)
        MSE = mask * (pred - gt) **2
        MAE = mask * abs(pred - gt)
        GDL = valid_mask_h * abs(gd_h(pred) - gd_h(gt)) + valid_mask_w * abs(gd_w(pred) - gd_w(gt))
        Returns
        -------

        """
        a = self._total_hits.astype(np.float64)
        b = self._total_false_alarms.astype(np.float64)
        c = self._total_misses.astype(np.float64)
        d = self._total_correct_negatives.astype(np.float64)
        pod = a / (a + c)
        far = b / (a + b)
        csi = a / (a + b + c)
        n = a + b + c + d
        aref = (a + b) / n * (a + c)
        gss = (a - aref) / (a + b + c - aref)
        hss = 2 * gss / (gss + 1)
        mse = self._mse / self._total_batch_num
        mae = self._mae / self._total_batch_num
        balanced_mse = self._balanced_mse / self._total_batch_num
        balanced_mae = self._balanced_mae / self._total_batch_num
        gdl = self._gdl / self._total_batch_num
        if not self._no_ssim:
            raise NotImplementedError
            # ssim = self._ssim / self._total_batch_num
        # return pod, far, csi, hss, gss, mse, mae, gdl
        return pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae, gdl

    def print_stat_readable(self, prefix=""):
        logging.getLogger().setLevel(logging.INFO)
        logging.info("%sTotal Sequence Number: %d, Use Central: %d"
                     %(prefix, self._total_batch_num, self._use_central))
        pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae, gdl = self.calculate_stat()
        # pod, far, csi, hss, gss, mse, mae, gdl = self.calculate_stat()
        logging.info("   Hits: " + ', '.join([">%g:%g/%g" % (threshold,
                                                             self._total_hits[:, i].mean(),
                                                             self._total_hits[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   POD: " + ', '.join([">%g:%g/%g" % (threshold, pod[:, i].mean(), pod[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   FAR: " + ', '.join([">%g:%g/%g" % (threshold, far[:, i].mean(), far[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   CSI: " + ', '.join([">%g:%g/%g" % (threshold, csi[:, i].mean(), csi[-1, i])
                                  for i, threshold in enumerate(self._thresholds)]))
        logging.info("   GSS: " + ', '.join([">%g:%g/%g" % (threshold, gss[:, i].mean(), gss[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   HSS: " + ', '.join([">%g:%g/%g" % (threshold, hss[:, i].mean(), hss[-1, i])
                                             for i, threshold in enumerate(self._thresholds)]))
        logging.info("   MSE: %g/%g" % (mse.mean(), mse[-1]))
        logging.info("   MAE: %g/%g" % (mae.mean(), mae[-1]))
        logging.info("   Balanced MSE: %g/%g" % (balanced_mse.mean(), balanced_mse[-1]))
        logging.info("   Balanced MAE: %g/%g" % (balanced_mae.mean(), balanced_mae[-1]))
        logging.info("   GDL: %g/%g" % (gdl.mean(), gdl[-1]))
        if not self._no_ssim:
            raise NotImplementedError

    def save_pkl(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        f = open(path, 'wb')
        logging.info("Saving HKOEvaluation to %s" %path)
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    def save_txt_readable(self, path):
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        pod, far, csi, hss, gss, mse, mae, balanced_mse, balanced_mae, gdl = self.calculate_stat()
        # pod, far, csi, hss, gss, mse, mae, gdl = self.calculate_stat()
        f = open(path, 'w')
        logging.info("Saving readable txt of HKOEvaluation to %s" % path)
        f.write("Total Sequence Num: %d, Out Seq Len: %d, Use Central: %d\n"
                %(self._total_batch_num,
                  self._seq_len,
                  self._use_central))
        for (i, threshold) in enumerate(self._thresholds):
            f.write("Threshold = %g:\n" %threshold)
            f.write("   POD: %s\n" %str(list(pod[:, i])))
            f.write("   FAR: %s\n" % str(list(far[:, i])))
            f.write("   CSI: %s\n" % str(list(csi[:, i])))
            f.write("   GSS: %s\n" % str(list(gss[:, i])))
            f.write("   HSS: %s\n" % str(list(hss[:, i])))
            f.write("   POD stat: avg %g/final %g\n" %(pod[:, i].mean(), pod[-1, i]))
            f.write("   FAR stat: avg %g/final %g\n" %(far[:, i].mean(), far[-1, i]))
            f.write("   CSI stat: avg %g/final %g\n" %(csi[:, i].mean(), csi[-1, i]))
            f.write("   GSS stat: avg %g/final %g\n" %(gss[:, i].mean(), gss[-1, i]))
            f.write("   HSS stat: avg %g/final %g\n" % (hss[:, i].mean(), hss[-1, i]))
        f.write("MSE: %s\n" % str(list(mse)))
        f.write("MAE: %s\n" % str(list(mae)))
        f.write("Balanced MSE: %s\n" % str(list(balanced_mse)))
        f.write("Balanced MAE: %s\n" % str(list(balanced_mae)))
        f.write("GDL: %s\n" % str(list(gdl)))
        f.write("MSE stat: avg %g/final %g\n" % (mse.mean(), mse[-1]))
        f.write("MAE stat: avg %g/final %g\n" % (mae.mean(), mae[-1]))
        f.write("Balanced MSE stat: avg %g/final %g\n" % (balanced_mse.mean(), balanced_mse[-1]))
        f.write("Balanced MAE stat: avg %g/final %g\n" % (balanced_mae.mean(), balanced_mae[-1]))
        f.write("GDL stat: avg %g/final %g\n" % (gdl.mean(), gdl[-1]))
        f.close()

    def save(self, prefix):
        self.save_txt_readable(prefix + ".txt")
        self.save_pkl(prefix + ".pkl")
