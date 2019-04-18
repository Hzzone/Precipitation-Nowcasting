from torch import nn
import torch
from nowcasting.config import cfg
from nowcasting.utils import rainfall_to_pixel
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
            for i in range(S):
                w = (1 + self._lambda * i)
                mse[i, ...] = mse[i, ...] * w
                mae[i, ...] = mae[i, ...] * w
        return self.NORMAL_LOSS_GLOBAL_SCALE * (self.mse_weight*torch.mean(mse) + self.mae_weight*torch.mean(mae))


class WeightedCrossEntropyLoss(nn.Module):

    # weight should be a 1D Tensor assigning weight to each of the classes.
    def __init__(self, thresholds, weight=None, LAMDA=None):
        super().__init__()
        self._weight = weight
        self._lambda = LAMDA
        self.thresholds = thresholds

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
        thresholds = [rainfall_to_pixel(ele) for ele in self.thresholds]
        for i, threshold in enumerate(thresholds, 1):
            class_index[target >= threshold] = i
        # Loss: B*S*H*W
        error = F.cross_entropy(input, class_index, self._weight, reduction='none')
        if self._lambda is not None:
            B, S, H, W = error.size()
            for i in range(S):
                error[:, i, ...] = error[:, i, ...] * (1 + self._lambda * i)
        error = error.unsqueeze(2)
        return torch.mean(error*mask)



