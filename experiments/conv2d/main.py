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



### Config


# batch_size = cfg.GLOBAL.BATCH_SZIE
batch_size = 4
max_iterations = 80000
test_iteration_interval = 10000
test_and_save_checkpoint_iterations = 10000

LR = 1e-4

criterion = Weighted_mse_mae().to(cfg.GLOBAL.DEVICE)

# build model
params = OrderedDict({
    'conv1_relu_1': [5, 64, 7, 5, 1],
    'conv2_relu_1': [64, 192, 5, 3, 1],
    'conv3_relu_1': [192, 192, 3, 2, 1],
    'deconv1_relu_1': [192, 192, 4, 2, 1],
    'deconv2_relu_1': [192, 64, 5, 3, 1],
    'deconv3_relu_1': [64, 64, 7, 5, 1],
    'conv3_relu_2': [64, 20, 3, 1, 1],
    'conv3_3': [20, 20, 1, 1, 0]
})

class Predictor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.model = make_layers(params)

    def forward(self, input):
        '''
        input: S*B*1*H*W
        :param input:
        :return:
        '''
        input = input.squeeze(2).permute((1, 0, 2, 3))
        output = self.model(input)
        return output.unsqueeze(2).permute((1, 0, 2, 3, 4))

model = Predictor(params).to(cfg.GLOBAL.DEVICE)
# data = torch.randn(5, 4, 1, 480, 480)
# output = model(data)
# print(output.size())

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=20000, gamma=0.7)
folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
train_and_test(model, optimizer, criterion, exp_lr_scheduler, batch_size, max_iterations, test_iteration_interval, test_and_save_checkpoint_iterations, folder_name)