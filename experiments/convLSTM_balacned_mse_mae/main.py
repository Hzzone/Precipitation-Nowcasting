import sys
sys.path.insert(0, '../../')
import torch
from nowcasting.config import cfg
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.models.loss import Weighted_mse_mae
from nowcasting.train_and_test import train_and_test
import os
from experiments.net_params import convlstm_encoder_params, convlstm_forecaster_params



### Config


batch_size = cfg.GLOBAL.BATCH_SZIE
max_iterations = 100000
test_iteration_interval = 1000
test_and_save_checkpoint_iterations = 1000
LR_step_size = 20000
gamma = 0.7

LR = 1e-4

criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)

encoder = Encoder(convlstm_encoder_params[0], convlstm_encoder_params[1]).to(cfg.GLOBAL.DEVICE)

forecaster = Forecaster(convlstm_forecaster_params[0], convlstm_forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=gamma)

folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]

train_and_test(encoder_forecaster, optimizer, criterion, exp_lr_scheduler, batch_size, max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations, folder_name)