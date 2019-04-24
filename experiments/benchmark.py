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
import copy
from experiments.net_params import *
from nowcasting.models.model import Predictor
from experiments.rover_and_last_frame import LastFrame, Rover


encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(forecaster_params[0], forecaster_params[1])
encoder_forecaster1 = EF(encoder, forecaster)
encoder_forecaster2 = copy.deepcopy(encoder_forecaster1)
conv2d_network = Predictor(conv2d_params).to(cfg.GLOBAL.DEVICE)

encoder_forecaster1 = encoder_forecaster1.to(cfg.GLOBAL.DEVICE)
encoder_forecaster2 = encoder_forecaster2.to(cfg.GLOBAL.DEVICE)
encoder_forecaster1.load_state_dict(torch.load('/home/hzzone/save/trajGRU_balanced_mse_mae/models/encoder_forecaster_77000.pth'))
encoder_forecaster2.load_state_dict(torch.load('/home/hzzone/save/trajGRU_frame_weighted_mse/models/encoder_forecaster_45000.pth'))
conv2d_network.load_state_dict(torch.load('/home/hzzone/save/conv2d/models/encoder_forecaster_60000.pth'))

models = OrderedDict({
    'conv2d': conv2d_network,
    'trajGRU_balanced_mse_mae': encoder_forecaster1,
    'trajGRU_frame_weighted_mse': encoder_forecaster2,
    'last_frame': LastFrame,
    'rover_nonlinear': Rover()
})

with torch.no_grad():
    for name, model in models.items():
        is_deeplearning_model = (torch.nn.Module in model.__class__.__bases__)
        if is_deeplearning_model:
            model.eval()
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

            valid_batch = valid_batch.astype(np.float32) / 255.0
            valid_data = valid_batch[:IN_LEN, ...]
            valid_label = valid_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
            mask = valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)

            if is_deeplearning_model:
                valid_data = torch.from_numpy(valid_data).to(cfg.GLOBAL.DEVICE)

            output = model(valid_data)

            if is_deeplearning_model:
                output = output.cpu().numpy()

            output = np.clip(output, 0.0, 1.0)

            evaluator.update(valid_label, output, mask)

            valid_time += 1
            # print(valid_time)
        #     if valid_time%1000==0:
        #         print('{} '.format(valid_time), end='')
        #
        # print()

        # evaluator.save_txt_readable(osp.join('./benchmark_stat', name + '.txt'))
        evaluator.save_pkl(osp.join('./benchmark_stat', name + '.pkl'))
