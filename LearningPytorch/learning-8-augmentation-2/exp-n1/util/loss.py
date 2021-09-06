import torch.nn as nn

from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.base.modules import Activation
import torch
from torch.autograd import Variable


class BCELoss(base.Loss):

    def __init__(self, w=1, **kwargs):
        super().__init__(**kwargs)
        self.loss = torch.nn.BCELoss()
        self.w = w
        return

    def forward(self, y_pr, y_gt):
        return self.w * self.loss(y_pr, y_gt)


class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., weight=1, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.weight = weight

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return self.weight * (1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        ))


class CombineLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        pr_cls, pr_reg = y_pr
        tg_cls, tg_reg, params_reg = y_gt

        # classification: BCELoss + DiceLoss
        focal_loss = F_focal_loss(pr_cls, tg_cls)
        pr_cls = self.activation(pr_cls)
        dice_loss = 1 - F.f_score(
            pr_cls, tg_cls,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=[0],
        )
        loss_cls = 0.2 * dice_loss + focal_loss
        # regression: iou loss

        loss_reg = F.bbox_iou(pr_reg, tg_reg, params_reg, type='diou')
        loss = loss_cls + 0.5 * loss_reg
        return loss


class FocalLoss(base.Loss):
    def __init__(self, alpha=1, gamma=2, w=1, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.w = w
        return

    def forward(self, y_pr, y_gt):
        '''
        y_pr: B, C, W, H batch, channel(类别), width, height
        y_gt: B, C, W, H batch, channel(类别), width, height
        '''
        y_pr = y_pr - y_pr.max(dim=1, keepdim=True)[0]
        logits = torch.exp(y_pr)
        logits_sum = logits.sum(dim=1, keepdim=True)
        _, gt_index = y_gt.max(dim=1, keepdim=True)
        pt = Variable((logits / logits_sum).gather(dim=1, index=gt_index))
        pt = (1 - pt) ** self.gamma
        loss = -y_gt * (y_pr - torch.log(logits_sum))
        loss = self.alpha * pt * loss
        loss = loss.sum(dim=1)
        loss = loss.mean(dim=[1, 2])
        return self.w * loss.mean()


def F_focal_loss(y_pr, y_gt, alpha=1, gamma=2):
    '''
    y_pr: B, C, W, H batch, channel(类别), width, height
    y_gt: B, C, W, H batch, channel(类别), width, height
    '''
    y_pr = y_pr - y_pr.max(dim=1, keepdim=True)[0]
    logits = torch.exp(y_pr)
    logits_sum = logits.sum(dim=1, keepdim=True)
    _, gt_index = y_gt.max(dim=1, keepdim=True)
    pt = Variable((logits / logits_sum).gather(dim=1, index=gt_index))
    pt = (1 - pt) ** gamma
    loss = -y_gt * (y_pr - torch.log(logits_sum))
    loss = alpha * pt * loss
    return loss.mean()


class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


# class BCELoss(nn.BCELoss, base.Loss):
#     pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
