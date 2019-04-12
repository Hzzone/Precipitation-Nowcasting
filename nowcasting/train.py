import sys
sys.path.insert(0, '../')
import logging
import torch
from nowcasting.hko.dataloader import HKOIterator
from nowcasting.config import cfg
import numpy as np
from nowcasting.models.forecaster import Forecaster
from nowcasting.models.encoder import Encoder
from nowcasting.models.convLSTM import ConvLSTM
from collections import OrderedDict
from nowcasting.models.model import EF
from torch.optim import lr_scheduler
from nowcasting.hko.evaluation import HKOEvaluation
from tqdm import tqdm



### Config

IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN

train_batch_size = 2
test_batch_size = 2
max_iterations = 2000
test_iteration_interval = 100
test_and_save_checkpoint_iterations = 100

LR = 1e-4
LR_step_size = 200

criterion = torch.nn.MSELoss(reduction='mean')

# build model
encoder_params = [
    [
        OrderedDict({'conv1_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_1': [192, 192, 3, 2, 1]}),
    ],

    [
        ConvLSTM(8, 64, 3),
        ConvLSTM(192, 192, 3),
        ConvLSTM(192, 192, 3)
    ]
]


encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)

forecaster_params = [
    [
        OrderedDict({'deconv1_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_1': [64, 8, 7, 5, 1],
            'conv3_2': [64, 8, 7, 5, 1],
            'deconv4_2': [8, 1, 1, 1, 0] # 忘了删除激活函数了，妈的
            # 忘了卷积层，分类
        }),
    ],

    [
        ConvLSTM(192, 192, 3),
        ConvLSTM(192, 192, 3),
        ConvLSTM(64, 64, 3)
    ]
]

forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_step_size, gamma=0.1)



# HKO-7 evaluater and dataloader
evaluater = HKOEvaluation(seq_len=OUT_LEN, use_central=False)
train_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TRAIN,
                                 sample_mode="random",
                                 seq_len=IN_LEN+OUT_LEN)

valid_hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_VALID,
                                 sample_mode="sequent",
                                 seq_len=IN_LEN+OUT_LEN,
                                 stride=cfg.HKO.BENCHMARK.STRIDE)

train_loss = 0.0
test_loss = 0.0

for itera in tqdm(range(max_iterations)):
    if itera != 0 and itera % test_iteration_interval == 0:
        print(train_loss)
        train_loss = 0.0
        test_loss = 0.0
        # evaluater.print_stat_readable()
        # evaluater.clear_all()
        with torch.no_grad():
            pass
    train_batch, train_mask, sample_datetimes, _ = \
        train_hko_iter.sample(batch_size=train_batch_size)
    train_batch = torch.from_numpy(train_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
    train_data = train_batch[:IN_LEN, ...]
    train_label = train_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
    mask = torch.from_numpy(train_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)

    encoder_forecaster.train()
    optimizer.zero_grad()
    output = encoder_forecaster(train_data)
    loss = criterion(output, train_label)
    loss.backward()
    optimizer.step()
    train_loss += loss.item()
    evaluater.update(train_label.cpu().numpy(), output.detach().cpu().numpy(), mask.cpu().numpy())
