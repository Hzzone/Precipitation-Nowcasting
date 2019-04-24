import sys
sys.path.insert(0, '../../')
import logging
import torch
from nowcasting.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from collections import OrderedDict
from nowcasting.models.model import EF
from nowcasting.models.loss import WeightedCrossEntropyLoss
from nowcasting.models.trajGRU import TrajGRU
from nowcasting.train_and_test import train_and_test
from experiments.net_params import encoder_params, forecaster_params
from torch.optim import lr_scheduler
import os
import numpy as np
from nowcasting.utils import *
from nowcasting.models.probToPixel import ProbToPixel
from torch import nn



# batch_size = cfg.GLOBAL.BATCH_SZIE
batch_size = 2
max_iterations = 50000
test_iteration_interval = 1000
test_and_save_checkpoint_iterations = 1000

LR = 1e-5
LR_step_size = 20000

### Config

encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)

forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
encoder_forecaster.load_state_dict(torch.load('/home/hzzone/save/trajGRU_balanced_mse_mae/models/encoder_forecaster_77000.pth'))
for param in encoder_forecaster.parameters():
    param.requires_grad = False

# thresholds = [9, 11, 15, 18, 20, 24, 26, 31, 36, 39, 43, 49, 53, 56, 58]
# thresholds = [9, 18, 26, 36, 37, 46, 50]
# thresholds = [9, 18, 36, 46, 53]
thresholds = []

thresholds = thresholds + rainfall_to_dBZ(cfg.HKO.EVALUATION.THRESHOLDS).tolist()
thresholds = np.array(sorted(thresholds))
encoder_forecaster.forecaster.stage1.conv3_3 = nn.Conv2d(8, len(thresholds)+1, kernel_size=(1, 1), stride=(1, 1)).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=0.7)

thresholds = dBZ_to_rainfall(thresholds)
weights = np.ones_like(thresholds)
balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
for i, threshold in enumerate(cfg.HKO.EVALUATION.THRESHOLDS):
    weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (thresholds >= threshold)
weights = weights + 1
weights = np.array([1] + weights.tolist())
weights = torch.from_numpy(weights).to(cfg.GLOBAL.DEVICE).float()
criterion = WeightedCrossEntropyLoss(thresholds, weights).to(cfg.GLOBAL.DEVICE)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]

ts = rainfall_to_dBZ(thresholds).tolist()
middle_value_dbz = np.array([-10.0] + [(x+y)/2 for x, y in zip(ts, ts[1:]+[60.0])])
middle_value = dBZ_to_pixel(middle_value_dbz).astype(np.float32)
probToPixel = ProbToPixel(middle_value, requires_grad=False)

train_and_test(encoder_forecaster, optimizer, criterion, exp_lr_scheduler, batch_size, max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations, folder_name, probToPixel)