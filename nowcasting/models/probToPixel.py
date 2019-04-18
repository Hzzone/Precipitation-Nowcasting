import torch
from nowcasting.config import cfg
from nowcasting.utils import rainfall_to_pixel

class ProbToPixel(object):

    def __init__(self, middle_value, NORMAL_LOSS_GLOBAL_SCALE=0.00005):
        '''
        middle_value: 分类之后类别的替代值，类别数，理应是像素值 [0, 1]
        middle_value_need_update: 需要更新的替代值
        :param middle_value:
        '''
        self._middle_value = torch.from_numpy(middle_value)
        self._middle_value_need_update = torch.from_numpy(middle_value, requires_grad=True)
        self.NORMAL_LOSS_GLOBAL_SCALE = NORMAL_LOSS_GLOBAL_SCALE

    def __call__(self, prediction, ground_truth, mask, lr):
        '''
        prediction: 输入的类别预测值，S*B*C*H*W
        ground_truth: 实际值，像素/255.0, [0, 1]
        lr: 学习率
        :param prediction:
        :return:
        '''
        # 分类结果，0 到 classes - 1
        result = torch.argmax(prediction, dim=2)
        prediction_result1 = torch.zeros_like(result).float()
        prediction_result2 = torch.zeros_like(result).float()
        for i in range(len(self._middle_value)):
            prediction_result1[result==i] = self._middle_value[i]
        # 如果需要更新替代值
        # 更新替代值

        # 权重
        balancing_weights = cfg.HKO.EVALUATION.BALANCING_WEIGHTS
        weights = torch.ones_like(prediction_result1) * balancing_weights[0]
        thresholds = [rainfall_to_pixel(ele) for ele in cfg.HKO.EVALUATION.THRESHOLDS]
        for i, threshold in enumerate(thresholds):
            weights = weights + (balancing_weights[i + 1] - balancing_weights[i]) * (ground_truth >= threshold).float()
        weights = weights * mask.float()


        loss = torch.zeros(1, requires_grad=True).float()
        for i in range(len(self._middle_value_need_update)):
            m = (result == i)
            prediction_result2[m] = self._middle_value_need_update.data[i]
            tmp = (ground_truth[m]-self._middle_value_need_update[i])
            mse = torch.sum(weights[m] * (tmp ** 2), (2, 3, 4))
            mae = torch.sum(weights[m] * (torch.abs(tmp)), (2, 3, 4))
            loss = self.NORMAL_LOSS_GLOBAL_SCALE * (torch.mean(mse) + torch.mean(mae))
        loss.backward()
        self._middle_value_need_update -= lr * self._middle_value_need_update.grad

        # prediction_result1: 直接用中间值替代
        # prediction_result2: 更新中间值
        return prediction_result1.cpu().numpy(), prediction_result2.cpu().numpy()
