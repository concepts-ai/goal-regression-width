
import torch

import torch.nn.functional as F
from jactorch.utils.meta import as_tensor, as_float

__all__ = ['binary_accuracy', 'rms', 'monitor_saturation', 'monitor_paramrms', 'monitor_gradrms']


def binary_accuracy(label, raw_pred, eps=1e-20, return_float=True):
    pred = as_tensor(raw_pred).squeeze(-1)  # Binary accuracy
    pred = (pred > 0.5).float()
    label = as_tensor(label).float()
    acc = label.eq(pred).float()

    nr_total = torch.ones(label.size(), dtype=label.dtype, device=label.device).sum(dim=-1)
    nr_pos = label.sum(dim=-1)
    nr_neg = nr_total - nr_pos
    pos_cnt = (acc * label).sum(dim=-1)
    neg_cnt = acc.sum(dim=-1) - pos_cnt
    balanced_acc = ((pos_cnt + eps) / (nr_pos + eps) + (neg_cnt + eps) / (nr_neg + eps)) / 2.0

    sat = 1 - (raw_pred - pred).abs()
    if return_float:
        acc = as_float(acc.mean())
        balanced_acc = as_float(balanced_acc.mean())
        sat_mean = as_float(sat.mean())
        sat_min = as_float(sat.min())
    else:
        sat_mean = sat.mean(dim=-1)
        sat_min = sat.min(dim=-1)[0]
    return {
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'satuation/mean': sat_mean,
        'satuation/min': sat_min,
    }


def rms(p):
    return as_float((as_tensor(p) ** 2).mean() ** 0.5)


def monitor_saturation(model):
    monitors = {}
    for name, p in model.named_parameters():
        p = F.sigmoid(p)
        sat = 1 - (p - (p > 0.5).float()).abs()
        monitors['sat/' + name] = sat
    return monitors


def monitor_paramrms(model):
    monitors = {}
    for name, p in model.named_parameters():
        monitors['paramrms/' + name] = rms(p)
    return monitors


def monitor_gradrms(model):
    monitors = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            monitors['gradrms/' + name] = rms(p.grad) / max(rms(p), 1e-8)
    return monitors
