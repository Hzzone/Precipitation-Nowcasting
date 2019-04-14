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
from nowcasting.models.loss import MSE, Weighted_MSE
from tensorboardX import SummaryWriter
import os.path as osp
import os
from nowcasting.utils import plot_result
import shutil
from nowcasting.models.trajGRU import TrajGRU


### Config

IN_LEN = cfg.HKO.BENCHMARK.IN_LEN
OUT_LEN = cfg.HKO.BENCHMARK.OUT_LEN

batch_size = 2
max_iterations = 200000
test_iteration_interval = 1000
test_and_save_checkpoint_iterations = 10000

LR = 1e-4
LR_step_size = 90000

# criterion = torch.nn.MSELoss(reduction='mean')
# criterion = MSE()
criterion = Weighted_MSE().to(cfg.GLOBAL.DEVICE)

# build model
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
        # ConvLSTM(8, 64, 3),
        # ConvLSTM(192, 192, 3),
        # ConvLSTM(192, 192, 3)
        TrajGRU(input_channel=8, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]


encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)

forecaster_params = [
    [
        OrderedDict({'deconv1_leaky_1': [192, 192, 4, 2, 1]}),
        OrderedDict({'deconv2_leaky_1': [192, 64, 5, 3, 1]}),
        OrderedDict({
            'deconv3_leaky_1': [64, 8, 7, 5, 1],
            'conv3_leaky_2': [8, 8, 3, 1, 1],
            'conv3_3': [8, 1, 1, 1, 0]
        }),
    ],

    [
        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 16, 16), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(3, 3), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),

        TrajGRU(input_channel=192, num_filter=192, b_h_w=(batch_size, 32, 32), zoneout=0.0, L=13,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE),
        TrajGRU(input_channel=64, num_filter=64, b_h_w=(batch_size, 96, 96), zoneout=0.0, L=9,
                i2h_kernel=(3, 3), i2h_stride=(1, 1), i2h_pad=(1, 1),
                h2h_kernel=(5, 5), h2h_dilate=(1, 1),
                act_type=cfg.MODEL.RNN_ACT_TYPE)
    ]
]

forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

optimizer = torch.optim.Adam(encoder_forecaster.parameters(), lr=LR, weight_decay=1e-6)
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

log_dir = '../logs'
if osp.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)

writer = SummaryWriter(log_dir)

for itera in tqdm(range(1, max_iterations+1)):
    train_batch, train_mask, sample_datetimes, _ = \
        train_hko_iter.sample(batch_size=batch_size)
    train_batch = torch.from_numpy(train_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
    train_data = train_batch[:IN_LEN, ...]
    train_label = train_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
    mask = torch.from_numpy(train_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)

    encoder_forecaster.train()
    optimizer.zero_grad()
    output = encoder_forecaster(train_data)
    loss = criterion(output, train_label, mask)
    loss.backward()
    torch.nn.utils.clip_grad_value_(encoder_forecaster.parameters(), clip_value=50.0)
    optimizer.step()
    train_loss += loss.item()


    train_label_numpy = train_label.cpu().numpy()
    output_numpy = np.clip(output.detach().cpu().numpy(), 0.0, 1.0)
    evaluater.update(train_label_numpy, output_numpy, mask.cpu().numpy())

    if itera % test_iteration_interval == 0:
        _, _, train_csi, train_hss, _, train_mse, train_mae, train_balanced_mse, train_balanced_mae, _ = evaluater.calculate_stat()

        train_loss = train_loss/test_iteration_interval

        evaluater.clear_all()

        with torch.no_grad():
            encoder_forecaster.eval()
            valid_hko_iter.reset()
            valid_loss = 0.0
            valid_time = 0
            while not valid_hko_iter.use_up:
                valid_batch, valid_mask, sample_datetimes, _ = \
                    valid_hko_iter.sample(batch_size=batch_size)
                if valid_batch.shape[1] == 0:
                    break
                if not cfg.HKO.EVALUATION.VALID_DATA_USE_UP and valid_time > cfg.HKO.EVALUATION.VALID_TIME:
                    break
                valid_time += 1
                valid_batch = torch.from_numpy(valid_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
                valid_data = valid_batch[:IN_LEN, ...]
                valid_label = valid_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
                mask = torch.from_numpy(valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)
                output = encoder_forecaster(valid_data)

                loss = criterion(output, valid_label, mask)
                valid_loss += loss.item()

                valid_label_numpy = valid_label.cpu().numpy()
                output_numpy = np.clip(output.detach().cpu().numpy(), 0.0, 1.0)
                evaluater.update(valid_label_numpy, output_numpy, mask.cpu().numpy())
            _, _, valid_csi, valid_hss, _, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae, _ = evaluater.calculate_stat()

            evaluater.clear_all()
            valid_loss = valid_loss/valid_time

        writer.add_scalars("loss", {
            "train": train_loss,
            "valid": valid_loss
        }, itera)

        plot_result(writer, itera, (train_csi, train_hss, train_mse, train_mae, train_balanced_mse, train_balanced_mae),
                    (valid_csi, valid_hss, valid_mse, valid_mae, valid_balanced_mse, valid_balanced_mae))


        train_loss = 0.0

    if itera % test_and_save_checkpoint_iterations == 0:
        torch.save(encoder_forecaster, 'model.pkl')



writer.export_scalars_to_json("../all_scalars.json")
writer.close()