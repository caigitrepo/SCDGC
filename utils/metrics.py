import abc
import numpy as np


class VOC12mAP(object):
    def __init__(self, num_classes):
        super(VOC12mAP, self).__init__()
        self._num_classes = num_classes

    def reset(self):
        self._predicted = np.array([], dtype=np.float32).reshape(0, self._num_classes)
        self._gt_label = np.array([], dtype=np.float32).reshape(0, self._num_classes)

    def update(self, predicted, gt_label):
        self._predicted = np.vstack((self._predicted, predicted))
        self._gt_label = np.vstack((self._gt_label, gt_label))

    def compute(self):
        return self._voc12_mAP()
    
    def _voc12_mAP(self):
        sample_num, num_classes = self._gt_label.shape
        ap_list = []

        for class_id in range(num_classes):
            confidence = self._predicted[:, class_id]
            sorted_ind = np.argsort(-confidence)
            sorted_label = self._gt_label[sorted_ind, class_id]

            tp = (sorted_label == 1).astype(np.int64)   # true positive
            fp = (sorted_label == 0).astype(np.int64)   # false positive
            tp_num = max(sum(tp), np.finfo(np.float64).eps)
            tp = np.cumsum(tp)
            fp = np.cumsum(fp)
            recall = tp / float(tp_num)
            precision = tp / np.arange(1, sample_num + 1, dtype=np.float64)

            ap = self._voc_AP(recall, precision, tp_num)    # average precision
            ap_list.append(ap)

        mAP = np.mean(ap_list)  # mean average precision
        return ap_list, mAP

    def _voc_AP(self, recall, precision, tp_num):
        mrec = np.concatenate(([0.], recall, [1.]))
        mpre = np.concatenate(([0.], precision, [0.]))
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap


class AverageLoss(object):
    def __init__(self, batch_size):
        super(AverageLoss, self).__init__()
        self._batch_size = batch_size

    def reset(self):
        self._sum = 0
        self._counter = 0

    def update(self, loss):
        self._sum += loss * self._batch_size
        self._counter += self._batch_size

    def compute(self):
        return self._sum / self._counter


class AverageMeter(object):
    def __init__(self, num_classes):
        super(AverageMeter, self).__init__()
        self.num_classes = num_classes

    def reset(self):
        self._right_pred_counter = np.zeros(self.num_classes)  # right predicted image per-class counter
        self._pred_counter = np.zeros(self.num_classes)    # predicted image per-class counter
        self._gt_counter = np.zeros(self.num_classes)  # ground-truth image per-class counter

    def update(self, confidence, gt_label):
        self._count(confidence, gt_label)

    def compute(self):
        self._op = sum(self._right_pred_counter) / sum(self._pred_counter)
        self._or = sum(self._right_pred_counter) / sum(self._gt_counter)
        self._of1 = 2 * self._op * self._or / (self._op + self._or)
        self._right_pred_counter = np.maximum(self._right_pred_counter, np.finfo(np.float64).eps)
        self._pred_counter = np.maximum(self._pred_counter, np.finfo(np.float64).eps)
        self._gt_counter = np.maximum(self._gt_counter, np.finfo(np.float64).eps)
        self._cp = np.mean(self._right_pred_counter / self._pred_counter)
        self._cr = np.mean(self._right_pred_counter / self._gt_counter)
        self._cf1 = 2 * self._cp * self._cr / (self._cp + self._cr)

    @abc.abstractmethod
    def _count(self, confidence, gt_label):
        pass

    @property
    def op(self):   # overall precision
        return self._op

    @property   # overall recall
    def or_(self):
        return self._or

    @property   # overall F1
    def of1(self):
        return self._of1

    @property   # per-class precision
    def cp(self):
        return self._cp

    @property   # per-class recall
    def cr(self):
        return self._cr

    @property   # per-class F1
    def cf1(self):
        return self._cf1


class TopkAverageMeter(AverageMeter):
    def __init__(self, num_classes, topk=3):
        super(TopkAverageMeter, self).__init__(num_classes)
        self.topk = topk

    def _count(self, confidence, gt_label):
        sample_num = confidence.shape[0]
        sorted_inds = np.argsort(-confidence, axis=-1)
        for i in range(sample_num):
            sample_gt_label = gt_label[i]
            topk_inds = sorted_inds[i][:self.topk]
            self._gt_counter[sample_gt_label == 1] += 1
            self._pred_counter[topk_inds] += 1
            correct_inds = topk_inds[sample_gt_label[topk_inds] == 1]
            self._right_pred_counter[correct_inds] += 1


class ThresholdAverageMeter(AverageMeter):
    def __init__(self, num_classes, threshold=0.5):
        super(ThresholdAverageMeter, self).__init__(num_classes)
        self.threshold = threshold

    def _count(self, confidence, gt_label):
        sample_num = confidence.shape[0]
        for i in range(sample_num):
            sample_gt_label = gt_label[i]
            self._gt_counter[sample_gt_label == 1] += 1
            inds = np.argwhere(confidence[i] > self.threshold).squeeze(-1)
            self._pred_counter[inds] += 1
            correct_inds = inds[sample_gt_label[inds] == 1]
            self._right_pred_counter[correct_inds] += 1


# import os, sys, pdb
# import math
# import torch
# from PIL import Image
# import numpy as np
# import random

# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self, name, fmt=':f'):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
    
#     def average(self):
#         return self.avg
    
#     def value(self):
#         return self.val

#     def __str__(self):
#         fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
#         return fmtstr.format(**self.__dict__)

# class AveragePrecisionMeter(object):
#     """
#     The APMeter measures the average precision per class.
#     The APMeter is designed to operate on `NxK` Tensors `output` and
#     `target`, and optionally a `Nx1` Tensor weight where (1) the `output`
#     contains model output scores for `N` examples and `K` classes that ought to
#     be higher when the model is more convinced that the example should be
#     positively labeled, and smaller when the model believes the example should
#     be negatively labeled (for instance, the output of a sigmoid function); (2)
#     the `target` contains only values 0 (for negative examples) and 1
#     (for positive examples); and (3) the `weight` ( > 0) represents weight for
#     each sample.
#     """

#     def __init__(self, difficult_examples=True):
#         super(AveragePrecisionMeter, self).__init__()
#         self.reset()
#         self.difficult_examples = difficult_examples

#     def reset(self):
#         """Resets the meter with empty member variables"""
#         self.scores = torch.FloatTensor(torch.FloatStorage())
#         self.targets = torch.LongTensor(torch.LongStorage())
#         self.filenames = []

#     def add(self, output, target, filename):
#         """
#         Args:
#             output (Tensor): NxK tensor that for each of the N examples
#                 indicates the probability of the example belonging to each of
#                 the K classes, according to the model. The probabilities should
#                 sum to one over all classes
#             target (Tensor): binary NxK tensort that encodes which of the K
#                 classes are associated with the N-th input
#                     (eg: a row [0, 1, 0, 1] indicates that the example is
#                          associated with classes 2 and 4)
#             weight (optional, Tensor): Nx1 tensor representing the weight for
#                 each example (each weight > 0)
#         """
#         if not torch.is_tensor(output):
#             output = torch.from_numpy(output)
#         if not torch.is_tensor(target):
#             target = torch.from_numpy(target)

#         if output.dim() == 1:
#             output = output.view(-1, 1)
#         else:
#             assert output.dim() == 2, \
#                 'wrong output size (should be 1D or 2D with one column \
#                 per class)'
#         if target.dim() == 1:
#             target = target.view(-1, 1)
#         else:
#             assert target.dim() == 2, \
#                 'wrong target size (should be 1D or 2D with one column \
#                 per class)'
#         if self.scores.numel() > 0:
#             assert target.size(1) == self.targets.size(1), \
#                 'dimensions for output should match previously added examples.'

#         # make sure storage is of sufficient size
#         if self.scores.storage().size() < self.scores.numel() + output.numel():
#             new_size = math.ceil(self.scores.storage().size() * 1.5)
#             self.scores.storage().resize_(int(new_size + output.numel()))
#             self.targets.storage().resize_(int(new_size + output.numel()))

#         # store scores and targets
#         offset = self.scores.size(0) if self.scores.dim() > 0 else 0
#         self.scores.resize_(offset + output.size(0), output.size(1))
#         self.targets.resize_(offset + target.size(0), target.size(1))
#         self.scores.narrow(0, offset, output.size(0)).copy_(output)
#         self.targets.narrow(0, offset, target.size(0)).copy_(target)

#         self.filenames += filename # record filenames

#     def value(self):
#         """Returns the model's average precision for each class
#         Return:
#             ap (FloatTensor): 1xK tensor, with avg precision for each class k
#         """

#         if self.scores.numel() == 0:
#             return 0
#         ap = torch.zeros(self.scores.size(1))
#         rg = torch.arange(1, self.scores.size(0)).float()
#         # compute average precision for each class
#         for k in range(self.scores.size(1)):
#             # sort scores
#             scores = self.scores[:, k]
#             targets = self.targets[:, k]
#             # compute average precision
#             ap[k] = AveragePrecisionMeter.average_precision(scores, targets, self.difficult_examples)
#         return ap

#     @staticmethod
#     def average_precision(output, target, difficult_examples=True):

#         # sort examples
#         sorted, indices = torch.sort(output, dim=0, descending=True)

#         # Computes prec@i
#         pos_count = 0.
#         total_count = 0.
#         precision_at_i = 0.
#         for i in indices:
#             label = target[i]
#             if difficult_examples and label == 0:
#                 continue
#             if label == 1:
#                 pos_count += 1
#             total_count += 1
#             if label == 1:
#                 precision_at_i += pos_count / total_count
#         precision_at_i /= pos_count
#         return precision_at_i

#     def overall(self):
#         if self.scores.numel() == 0:
#             return 0
#         scores = self.scores.cpu().numpy()
#         targets = self.targets.clone().cpu().numpy()
#         targets[targets == -1] = 0
#         return self.evaluation(scores, targets)

#     def overall_topk(self, k):
#         targets = self.targets.clone().cpu().numpy()
#         targets[targets == -1] = 0
#         n, c = self.scores.size()
#         scores = np.zeros((n, c)) - 1
#         index = self.scores.topk(k, 1, True, True)[1].cpu().numpy()
#         tmp = self.scores.cpu().numpy()
#         for i in range(n):
#             for ind in index[i]:
#                 scores[i, ind] = 1 if tmp[i, ind] >= 0 else -1
#         return self.evaluation(scores, targets)

#     def evaluation(self, scores_, targets_):
#         n, n_class = scores_.shape
#         Nc, Np, Ng = np.zeros(n_class), np.zeros(n_class), np.zeros(n_class)
#         for k in range(n_class):
#             scores = scores_[:, k]
#             targets = targets_[:, k]
#             targets[targets == -1] = 0
#             Ng[k] = np.sum(targets == 1)
#             Np[k] = np.sum(scores >= 0)
#             Nc[k] = np.sum(targets * (scores >= 0))
#         Np[Np == 0] = 1
#         OP = np.sum(Nc) / np.sum(Np)
#         OR = np.sum(Nc) / np.sum(Ng)
#         OF1 = (2 * OP * OR) / (OP + OR)

#         CP = np.sum(Nc / Np) / n_class
#         CR = np.sum(Nc / Ng) / n_class
#         CF1 = (2 * CP * CR) / (CP + CR)
#         return OP, OR, OF1, CP, CR, CF1