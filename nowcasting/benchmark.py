import sys
sys.path.insert(0, '..')
from nowcasting.hko.dataloader import HKOIterator
from nowcasting.config import cfg
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
import numpy as np
from nowcasting.hko.evaluation import *
from nowcasting.hko.benchmark import HKOBenchmarkEnv

IN_LEN = 5
OUT_LEN = 20

batch_size = 1

# build model
encoder_params = [
    [
        OrderedDict({'conv1_leaky_1': [1, 8, 7, 5, 1]}),
        OrderedDict({'conv2_leaky_1': [64, 192, 5, 3, 1]}),
        OrderedDict({'conv3_leaky_1': [192, 192, 3, 2, 1]}),
    ],

    [
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
encoder_forecaster.load_state_dict(torch.load('/home/hzzone/save/trajGRU_balanced_mse_mae/models/encoder_forecaster_77000.pth'))

with torch.no_grad():
    encoder_forecaster.eval()
    evaluator = HKOEvaluation(seq_len=OUT_LEN, use_central=False)
    hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TEST,
    # hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_VALID,
                                 sample_mode="sequent",
                                 seq_len=IN_LEN + OUT_LEN,
                                 stride=cfg.HKO.BENCHMARK.STRIDE)
    valid_time = 0
    while not hko_iter.use_up:
        valid_batch, valid_mask, sample_datetimes, _ = \
            hko_iter.sample(batch_size=batch_size)
        if valid_batch.shape[1] == 0:
            break
        if not cfg.HKO.EVALUATION.VALID_DATA_USE_UP and valid_time > cfg.HKO.EVALUATION.VALID_TIME:
            break

        valid_batch = torch.from_numpy(valid_batch.astype(np.float32)).to(cfg.GLOBAL.DEVICE) / 255.0
        valid_data = valid_batch[:IN_LEN, ...]
        valid_label = valid_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
        mask = torch.from_numpy(valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)).to(cfg.GLOBAL.DEVICE)
        output = encoder_forecaster(valid_data)

        valid_label_numpy = valid_label.cpu().numpy()
        output_numpy = np.clip(output.detach().cpu().numpy(), 0.0, 1.0)

        evaluator.update(valid_label_numpy, output_numpy, mask.cpu().numpy())

        valid_time += 1
        print(valid_time)

    evaluator.print_stat_readable()
