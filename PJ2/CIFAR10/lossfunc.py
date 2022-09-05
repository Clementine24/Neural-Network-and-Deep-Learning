# -*- coding = utf-8 -*-
# @Time : 2022/4/29 15:16
# @Author : fan
# @File:lossfunc.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Some kinds of loss function, including SmoothL1Loss, KLDivLoss and MultiMarginLoss
"""

class smoothl1loss(nn.Module):
    """
    My smooth L1-loss function
    """
    def __init__(self, smooth = 0.1):
        super(smoothl1loss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, real):
        num_class = pred.size(1)
        real_vec = torch.full_like(pred, fill_value = self.smooth/(num_class - 1))
        # print(real.size())
        # print(real_vec.size())
        real_vec.scatter_(1, real.unsqueeze(1), value = 1 - self.smooth)
        pred_vec = F.softmax(pred, dim = 1)

        return F.smooth_l1_loss(pred_vec, real_vec).mean()


class klloss(nn.Module):
    """
    My KL loss function
    """
    def __init__(self, smooth = 0.1):
        super(klloss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, real):
        num_class = pred.size(1)
        real_vec = torch.full_like(pred, fill_value=self.smooth / (num_class - 1))
        real_vec.scatter_(1, real.unsqueeze(1), value=1 - self.smooth)
        pred_vec = F.log_softmax(pred, dim=1)

        return F.kl_div(pred_vec, real_vec, reduction='batchmean').mean()

class smoothcrossentropy(nn.Module):
    """
    My label smooth cross entropy loss function
    """
    def __init__(self, smooth = 0.1):
        super(smoothcrossentropy, self).__init__()
        self.smooth = smooth

    def forward(self, pred, real):
        num_class = pred.size(1)
        real_vec = torch.full_like(pred, fill_value=self.smooth / (num_class - 1))
        real_vec.scatter_(1, real.unsqueeze(1), value=1 - self.smooth)
        loss = -(real_vec * F.log_softmax(pred, dim=1)).sum(dim=1)

        return loss.mean()


class multimarginloss(nn.Module):
    """
    My multi margin loss function
    """
    def __init__(self):
        super(multimarginloss, self).__init__()

    def forward(self, pred, real):
        return F.multi_margin_loss(pred, real).mean()