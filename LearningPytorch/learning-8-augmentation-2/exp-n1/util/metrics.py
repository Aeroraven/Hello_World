from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils import functional as F
from segmentation_models_pytorch.base.modules import Activation
import torch
from sklearn.metrics import classification_report
from pytorch_lightning.metrics import FBeta, Accuracy, Precision, Recall, IoU


"""
this file contains various metrics to evaluate the performance of model on train/test dataset
all custom metrics should inheret segmentation_models_pytorch.utils.base.Metrics in order to be embedded into SMP framework
"""

class L_IoU(base.Metric):
    def __init__(self, num_classes, **kwargs):
        super(L_IoU, self).__init__()
        self.measure = IoU(num_classes=num_classes, ignore_index=0)
        self.activation = torch.nn.Softmax2d()
        return
    
    def forward(self, pr, gt):
        pr_a = pr.clone().detach()
        gt_a = gt.clone().detach()
        gt_a[:,0,:,:] = gt_a[:,0,:,:] + 1e-5
        gt_a = torch.argmax(gt_a, dim=1)
        pr_a = self.activation(pr_a)
        pr_a[:,0,:,:] = pr_a[:,0,:,:] + 1e-5
        pr_a = torch.argmax(pr_a, dim=1)
        return self.measure(pr_a, gt_a)
    

class L_Recall(base.Metric):
    def __init__(self, num_classes, average='macro', **kwargs):
        super(L_Recall, self).__init__()
        self.measure = Recall(num_classes=num_classes, average=average)
        return
    
    def forward(self, pr, gt):
        return self.measure(torch.argmax(pr, dim=1), torch.argmax(gt, dim=1))
    
    
class L_Precision(base.Metric):
    def __init__(self, num_classes, average='macro', **kwargs):
        super(L_Precision, self).__init__()
        self.measure = Precision(num_classes=num_classes, average=average)
        return
    
    def forward(self, pr, gt):
        return self.measure(torch.argmax(pr, dim=1), torch.argmax(gt, dim=1))
    
    
class L_FBeta(base.Metric):
    def __init__(self, num_classes, beta=0.5, **kwargs):
        super(L_FBeta, self).__init__()
        self.measure = FBeta(num_classes=num_classes, beta=beta)
        return
    
    def forward(self, pr, gt):
        return self.measure(torch.argmax(pr, dim=1), torch.argmax(gt, dim=1))
    
    
class L_Accuracy(base.Metric):
    def __init__(self, **kwargs):
        super(L_Accuracy, self).__init__()
        self.measure = Accuracy()
        return
    
    def forward(self, pr, gt):
        return self.measure(torch.argmax(pr, dim=1), torch.argmax(gt, dim=1))
    

class LabelAccuracy(base.Metric):
    def __init__(self, **kwargs):
        super(LabelAccuracy, self).__init__()
        return
    
    def forward(self, pr, gt):
        pr = pr.clone().detach()
        pr[pr>=0.5] = 1
        pr[pr<0.5] = 0
        batch_mean = torch.sum((pr+gt)==2, dim=-1).float()
        return batch_mean.mean()
    
    
class LabelPrecision(base.Metric):
    def __init__(self, **kwargs):
        super(LabelPrecision, self).__init__()
        return
    
    def forward(self, pr, gt):
        pr = pr.clone().detach()
        pr[pr>=0.5] = 1
        pr[pr<0.5] = 0
        tp = torch.sum((pr+gt)==2, dim=-1)
        p = torch.sum((pr+gt)>=1, dim=-1)
        return (tp/p).mean()
    
    
class LabelRecall(base.Metric):
    def __init__(self, **kwargs):
        super(LabelRecall, self).__init__()
        return
    
    def forward(self, pr, gt):
        pr = pr.clone().detach()
        pr[pr>=0.5] = 1
        pr[pr<0.5] = 0
        tp = torch.sum((pr+gt)==2, dim=-1)
        p = torch.sum(gt==1, dim=-1)
        return (tp/p).mean()
    
    
class AggregatedLabelAccuracy(base.Metric):
    def __init__(self, **kwargs):
        super(AggregatedLabelAccuracy, self).__init__()
        return
    
    def forward(self, pr, gt):
        gt_positive = gt[gt==1]
        pr_positive = pr[gt==1]
        pr_positive[pr_positive>=0.5] = 1.
        pr_positive[pr_positive<0.5] = 0
        batch_acc = (pr_positive==gt_positive).sum()/gt_positive.sum()
        return batch_acc
    

class AggregatedLabelRecall(base.Metric):
    def __init__(self, **kwargs):
        super(AggregatedLabelRecall, self).__init__()
        return
    
    def forward(self, pr, gt):
        pr = pr.clone().detach()
        pr[pr>=0.5] = 1.
        pr[pr<0.5] = 0
        pr_positive = pr[pr==1]
        gt_positive = gt[pr==1]
        batch_recall = (pr_positive==gt_positive).sum()/pr_positive.sum()
        return batch_recall


class SMPIoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(
            y_pr, y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class L1Distance(base.Metric):
    __name__ = 'l1_distance'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt, params_delta):
        return F.l1_distance(y_pr, y_gt, params_delta)


class BBox_IOU(base.Metric):
    __name__ = 'bbox_iou'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt, params_delta):
        return 1 - F.bbox_iou(y_pr, y_gt, params_delta)


class Fscore(base.Metric):
    __name__ = 'dice'

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr, y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
