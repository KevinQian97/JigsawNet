import numpy as np
import torch
import json
import os
import decord
from decord import VideoReader

def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy_nll(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)

    prob, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # print(pred,prob)
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_bce(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    # print(output.size())
    # print(target.size())
    inner_target = []
    inner_output = []
    batch_size = target.size(0)
    for i in range(batch_size):
        if torch.max(target[i])>0:
            inner_output.append(output[i])
            inner_target.append(target[i])
    if len(inner_target)==0:
        return [torch.tensor(-1).cuda(),torch.tensor(-1).cuda()],0
    inner_output = torch.stack(inner_output)
    inner_target = torch.stack(inner_target)
    inner_target = torch.argmax(inner_target,dim=1)
    

    _, pred = inner_output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(inner_target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, len(inner_target)