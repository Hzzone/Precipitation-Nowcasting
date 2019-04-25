import sys
import cv2
import shutil
sys.path.insert(0, '.')
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
from experiments.net_params import *
from nowcasting.models.model import Predictor
from nowcasting.helpers.visualization import save_hko_movie


encoder = Encoder(encoder_params[0], encoder_params[1]).to(cfg.GLOBAL.DEVICE)
forecaster = Forecaster(forecaster_params[0], forecaster_params[1])
encoder_forecaster = EF(encoder, forecaster).to(cfg.GLOBAL.DEVICE)

encoder_forecaster.load_state_dict(torch.load('/home/hzzone/save/trajGRU_frame_weighted_mse/models/encoder_forecaster_45000.pth'))

hko_iter = HKOIterator(pd_path=cfg.HKO_PD.RAINY_TEST,
                       sample_mode="random",
                       seq_len=IN_LEN + OUT_LEN)

valid_batch, valid_mask, sample_datetimes, _ = \
    hko_iter.sample(batch_size=1)

valid_batch = valid_batch.astype(np.float32) / 255.0
valid_data = valid_batch[:IN_LEN, ...]
valid_label = valid_batch[IN_LEN:IN_LEN + OUT_LEN, ...]
mask = valid_mask[IN_LEN:IN_LEN + OUT_LEN, ...].astype(int)
torch_valid_data = torch.from_numpy(valid_data).to(cfg.GLOBAL.DEVICE)

with torch.no_grad():
    output = encoder_forecaster(torch_valid_data)

output = np.clip(output.cpu().numpy(), 0.0, 1.0)

base_dir = '.'
# S*B*1*H*W
label = valid_label[:, 0, 0, :, :]
output = output[:, 0, 0, :, :]
mask = mask[:, 0, 0, :, :].astype(np.uint8)
save_hko_movie(label, sample_datetimes[0], mask, masked=True,
                       save_path=os.path.join(base_dir, 'ground_truth.mp4'))
save_hko_movie(output, sample_datetimes[0], mask, masked=True,
               save_path=os.path.join(base_dir, 'pred.mp4'))
