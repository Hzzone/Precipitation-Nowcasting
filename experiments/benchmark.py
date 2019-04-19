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


encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(forecaster_params[0], forecaster_params[1]).to(cfg.GLOBAL.DEVICE)

encoder_forecaster1 = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)
encoder_forecaster1.load_state_dict(torch.load('/home/hzzone/save/trajGRU_balanced_mse_mae/models/encoder_forecaster_77000.pth'))

models = {
    'trajGRU_balanced_mse_mae': encoder_forecaster1,
}

with torch.no_grad():
    for name, encoder_forecaster in models.items():
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
        evaluator.save_txt_readable(osp.join('.', name + '.txt'))
