from nowcasting.helpers.ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch

__C = edict()
cfg = __C
__C.GLOBAL = edict()
__C.GLOBAL.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
__C.GLOBAL.BATCH_SZIE = 2
for dirs in ['/home/hzzone/save', '/Users/hzzone/Downloads']:
    if os.path.exists(dirs):
        __C.GLOBAL.MODEL_SAVE_DIR = dirs
assert __C.GLOBAL.MODEL_SAVE_DIR is not None


__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
__C.HKO_DATA_BASE_PATH = os.path.join(__C.ROOT_DIR, 'hko_data')

for dirs in ['/Users/hzzone/Downloads/HKO-7_data/radarPNG', '/home/hzzone/HKO-7/radarPNG']:
    if os.path.exists(dirs):
        __C.HKO_PNG_PATH = dirs
for dirs in ['/Users/hzzone/Downloads/HKO-7_data/radarPNG_mask', '/home/hzzone/HKO-7/radarPNG_mask']:
    if os.path.exists(dirs):
        __C.HKO_MASK_PATH = dirs

__C.HKO = edict()


__C.HKO.EVALUATION = edict()
__C.HKO.EVALUATION.THRESHOLDS = np.array([0.5, 2, 5, 10, 30])
__C.HKO.EVALUATION.CENTRAL_REGION = (120, 120, 360, 360)
__C.HKO.EVALUATION.BALANCING_WEIGHTS = (1, 1, 2, 5, 10, 30)

__C.HKO.EVALUATION.VALID_DATA_USE_UP = True
# __C.HKO.EVALUATION.VALID_TIME = 100
__C.HKO.EVALUATION.VALID_TIME = 20


__C.HKO.BENCHMARK = edict()

__C.HKO.BENCHMARK.STAT_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'benchmark_stat')
if not os.path.exists(__C.HKO.BENCHMARK.STAT_PATH):
    os.makedirs(__C.HKO.BENCHMARK.STAT_PATH)

__C.HKO.BENCHMARK.VISUALIZE_SEQ_NUM = 10  # Number of sequences that will be plotted and saved to the benchmark directory
__C.HKO.BENCHMARK.IN_LEN = 5   # The maximum input length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.OUT_LEN = 20  # The maximum output length to ensure that all models are tested on the same set of input data
__C.HKO.BENCHMARK.STRIDE = 5   # The stride


# pandas data
__C.HKO_PD_BASE_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'pd')
if not os.path.exists(__C.HKO_PD_BASE_PATH):
    os.makedirs(__C.HKO_PD_BASE_PATH)
__C.HKO_VALID_DATETIME_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'valid_datetime.pkl')
__C.HKO_SORTED_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'sorted_day.pkl')
__C.HKO_RAINY_TRAIN_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_train_days.txt')
__C.HKO_RAINY_VALID_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_valid_days.txt')
__C.HKO_RAINY_TEST_DAYS_PATH = os.path.join(__C.HKO_DATA_BASE_PATH, 'hko7_rainy_test_days.txt')

__C.HKO_PD = edict()
__C.HKO_PD.ALL = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all.pkl')
__C.HKO_PD.ALL_09_14 = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all_09_14.pkl')
__C.HKO_PD.ALL_15 = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_all_15.pkl')
__C.HKO_PD.RAINY_TRAIN = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_train.pkl')
__C.HKO_PD.RAINY_VALID = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_valid.pkl')
__C.HKO_PD.RAINY_TEST = os.path.join(__C.HKO_PD_BASE_PATH, 'hko7_rainy_test.pkl')


__C.HKO.ITERATOR = edict()
__C.HKO.ITERATOR.WIDTH = 480
__C.HKO.ITERATOR.HEIGHT = 480
__C.HKO.ITERATOR.FILTER_RAINFALL = True           # Whether to discard part of the rainfall, has a denoising effect
__C.HKO.ITERATOR.FILTER_RAINFALL_THRESHOLD = 0.28 # All the pixel values that are smaller than round(threshold * 255) will be discarded

__C.MODEL = edict()
from nowcasting.models.model import activation
__C.MODEL.RNN_ACT_TYPE = activation('leaky', negative_slope=0.2, inplace=True)
