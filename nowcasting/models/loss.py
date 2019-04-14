from torch import nn
import torch
from nowcasting.config import cfg
from nowcasting.utils import rainfall_to_pixel

class Weighted_MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask):
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(input) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (target >= threshold).float()
        weights = weights * mask.float()
        return torch.mean(torch.sum(weights * ((input-target)**2), (2, 3, 4)))

class MSE(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, input, target, mask):
        return torch.mean(mask*torch.sum((input-target)**2, (2, 3, 4)))
