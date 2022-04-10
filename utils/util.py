import math
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
from sklearn.manifold import TSNE


class EarlyStopping(object):
    def __init__(self, patience):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.counter = 0
        self.best_score = None

    def __call__(self, score):
        is_save, is_terminate = True, False
        if self.best_score is None:
            self.best_score = score
        elif self.best_score >= score:
            self.counter += 1
            if self.counter >= self.patience:
                is_terminate = True
            is_save = False
        else:
            self.best_score = score
            self.counter = 0
        return is_save, is_terminate


class TencentLoss(object):
    def __init__(self, class_num, pos_weight=12.0):
        super(TencentLoss, self).__init__()
        self.pos_weight = torch.FloatTensor(class_num).fill_(pos_weight).cuda()
        self.pre_status = torch.IntTensor(class_num).fill_(-1).cuda()
        self.t = None

    def __call__(self, input, target):
        r = self._get_adaptive_weight(target)
        output = F.binary_cross_entropy_with_logits(input, target, weight=r, pos_weight=self.pos_weight)
        return output

    def _get_adaptive_weight(self, target):
        class_status = torch.sum(target, dim=0)
        cur_status = class_status > torch.tensor(0.0).cuda()
        cur_status = cur_status.type_as(self.pre_status)
        if torch.all(torch.eq(self.pre_status, cur_status)):
            self.t += 1
        else:
            self.t = 1
            self.pre_status = cur_status

        pos_r = max(0.01, math.log10(10/(0.01+self.t)))
        neg_r = max(0.01, math.log10(10/(8+self.t)))
        pos_r = target.clone().fill_(pos_r)
        neg_r = target.clone().fill_(neg_r)

        r = torch.where(target == 1, pos_r, neg_r)
        return r


class FocalLoss(object):
    def __init__(self, alpha=0.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def __call__(self, input, target):
        # input_prob = torch.sigmoid(input)
        input_prob = input
        hard_easy_weight = (1 - input_prob) * target + input_prob * (1 - target)
        posi_nega_weight = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = (posi_nega_weight * torch.pow(hard_easy_weight, self.gamma)).detach()
        focal_loss = F.binary_cross_entropy_with_logits(input, target, weight=focal_weight)
        return focal_loss


class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

