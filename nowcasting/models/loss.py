from torch import nn
import torch
from nowcasting.config import cfg
from nowcasting.utils import rainfall_to_pixel, dBZ_to_pixel
import torch.nn.functional as F

class Weighted_mse_mae(nn.Module):
    def __init__(self, mse_weight=1.0, mae_weight=1.0, NORMAL_LOSS_GLOBAL_SCALE=0.00005, LAMBDA=None):
        super().__init__()
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self._lambda = LAMBDA

    def forward(self, input, target, mask):
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        weights = weights * mask.float()
        # input: S*B*1*H*W
        # error: S*B
        mse = torch.sum(weights * ((input-target)**2), (2, 3, 4))
        mae = torch.sum(weights * (torch.abs((input-target))), (2, 3, 4))
        if self._lambda is not None:
            S, B = mse.size()
            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(mse.get_device())
            mse = (w * mse.permute(1, 0)).permute(1, 0)
            mae = (w * mae.permute(1, 0)).permute(1, 0)
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))


class WeightedCrossEntropyLoss(nn.Module):

    # weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, thresholds, weight=None, LAMBDA=None):
        super().__init__()
        # 每个类别的权重，使用原文章的权重。
        self._weight = weight
        # 每一帧 Loss 递进参数
        self._lambda = LAMBDA
        # thresholds: 雷达反射率
        self._thresholds = thresholds

    # input: output prob, S*B*C*H*W
    # target: S*B*1*H*W, original data, range [0, 1]
    # mask: S*B*1*H*W
    def forward(self, input, target, mask):
        assert input.size(0) == cfg.HKO.BENCHMARK.OUT_LEN
        # F.cross_entropy should be B*C*S*H*W
        input = input.permute((1, 2, 0, 3, 4))
        # B*S*H*W
        target = target.permute((1, 2, 0, 3, 4)).squeeze(1)
        class_index = torch.zeros_like(target).long()
        thresholds = [0.0] + rainfall_to_pixel(self._thresholds).tolist()
        # print(thresholds)
        for i, threshold in enumerate(thresholds):
            class_index[target >= threshold] = i
        error = F.cross_entropy(input, class_index, self._weight, reduction='none')
        if self._lambda is not None:
            B, S, H, W = error.size()

            w = torch.arange(1.0, 1.0 + S * self._lambda, self._lambda)
            if torch.cuda.is_available():
                w = w.to(error.get_device())
                # B, H, W, S
            error = (w * error.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # S*B*1*H*W
        error = error.permute(1, 0, 2, 3).unsqueeze(2)
        return torch.mean(error*mask.float())



