import sys
sys.path.insert(0, '../../')
import torch
from nowcasting.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from collections import OrderedDict
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.models.loss import Weighted_mse_mae
from nowcasting.models.trajGRU import TrajGRU
from nowcasting.train_and_test import train_and_test
import os
from nowcasting.utils import make_layers
from torch import nn
from experiments.net_params import conv2d_params
from nowcasting.models.model import Predictor



### Config


# batch_size = cfg.GLOBAL.BATCH_SZIE
batch_size = 4
max_iterations = 80000
test_iteration_interval = 10000
test_and_save_checkpoint_iterations = 10000

LR = 1e-4

criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)


model = Predictor(conv2d_params).to(cfg.GLOBAL.DEVICE)
# data = torch.randn(5, 4, 1, 480, 480)
# output = model(data)
# print(output.size())

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.7)
folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]

if __name__ == '__main__':
    train_and_test(model, optimizer, criterion, exp_lr_scheduler, batch_size, max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations, folder_name)