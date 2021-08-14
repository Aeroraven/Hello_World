import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


"""
this files defines Epoch, TrainEpoch, and TestEpoch for MRP pretext/downstream task traning/testing
这里Valid Epoch 其实就是 Test Epoch
如果你们需要 Valid Epoch，可能需要你们自己写，事实上没啥区别
注释掉的内容不看，实验用的
"""

###########################################
# abstract Epoch defines public interfaces for train and test epoch
###########################################
class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for f, x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                
                ### data conversion: 不同的任务的训练条件下有不同的 input shape
                if len(x.shape)==5: ### rotation batch dim
                    x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]).float()
                if len(y.shape)==3: ### rotation batch dim for label
                    y = y.view(-1, y.shape[-1]).float()
                elif len(y.shape)==5: ### rotation batch dim for map
                    y = y.view(-1, y.shape[-3], y.shape[-2], y.shape[-1]).float()
                elif len(y.shape)==4: ### normal segmentation
                    pass
                else:
                    raise ValueError(f'y shape = {y.shape}')
                    
                loss, y_pred = self.batch_update(x, y)
                
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        loss = self.loss(prediction, y)
        loss.backward()
        self.optimizer.step()
        return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction

# ###############################################################
# # multi output version
# ###############################################################

# class EpochMultiOutput(Epoch):

#     def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
#         super(EpochMultiOutput, self).__init__(model, loss, metrics, stage_name, device, verbose)
#         return

#     def _to_device(self):
#         self.model.to(self.device)
#         self.loss['mask'].to(self.device)
#         self.loss['label'].to(self.device)
#         for metric in self.metrics['mask']:
#             metric.to(self.device)
#         for metric in self.metrics['label']:
#             metric.to(self.device)
    
#     def run(self, dataloader):

#         self.on_epoch_start()

#         logs = {}
#         loss_meter = {'mask': AverageValueMeter(), 'label': AverageValueMeter()}
#         metrics_meters_mask = {metric.__name__: AverageValueMeter() for metric in self.metrics['mask']}
#         metrics_meters_label = {metric.__name__: AverageValueMeter() for metric in self.metrics['label']}

#         with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
#             for f, x, y, z in iterator:
#                 x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                
#                 if len(x.shape)==5: ### rotation batch dim
#                     x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]).float()
                    
#                 if len(y.shape)==3: ### rotation batch dim for label
#                     y = y.view(-1, y.shape[-1]).float()
#                 elif len(y.shape)==2:
#                     pass
#                 else:
#                     raise ValueError(f'y shape = {y.shape}')
                    
#                 if len(z.shape)==5: ### rotation batch dim for map
#                     z = z.view(-1, z.shape[-3], z.shape[-2], z.shape[-1]).float()
#                 elif len(z.shape)==4: ### normal segmentation
#                     pass
#                 else:
#                     raise ValueError(f'z shape = {z.shape}')
                    
#                 loss, pred = self.batch_update(x, y, z)

#                 # update loss logs
#                 loss_value = (loss['mask']+loss['label']).cpu().detach().numpy()
#                 loss_meter.add(loss_value)
#                 loss_logs = {self.loss.__name__: loss_meter.mean}
#                 logs.update(loss_logs)

#                 # update metrics logs
#                 for metric_fn in self.metrics['mask']:
#                     metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
#                     metrics_meters_mask[metric_fn.__name__].add(metric_value)
#                 metrics_logs = {k: v.mean for k, v in metrics_meters_mask.items()}
#                 for metric_fn in self.metrics['label']:
#                     metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
#                     metrics_meters_label[metric_fn.__name__].add(metric_value)
#                 metrics_logs.update({k: v.mean for k, v in metrics_meters_label.items()})
#                 logs.update(metrics_logs)

#                 if self.verbose:
#                     s = self._format_logs(logs)
#                     iterator.set_postfix_str(s)

#         return logs
    
    
# class TrainEpochMultiOutput(EpochMultiOutput):

#     def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
#         super().__init__(
#             model=model,
#             loss=loss,
#             metrics=metrics,
#             stage_name='train',
#             device=device,
#             verbose=verbose,
#         )
#         self.optimizer = optimizer

#     def on_epoch_start(self):
#         self.model.train()

#     def batch_update(self, x, label, mask):
#         self.optimizer.zero_grad()
#         prediction_mask, prediction_label = self.model.forward(x)
#         prediction = {'mask': prediction_mask, 'label': prediction_label}
#         loss1 = self.loss['mask'](prediction_mask, mask)
#         loss2 = self.loss['label'](prediction_label, label)
#         loss = loss1 + loss2
#         loss.backward()
#         self.optimizer.step()
#         return {'mask': loss1, 'label': loss2}, prediction
    
    
# class ValidEpochMultiOutput(EpochMultiOutput):

#     def __init__(self, model, loss, metrics, device='cpu', verbose=True):
#         super().__init__(
#             model=model,
#             loss=loss,
#             metrics=metrics,
#             stage_name='valid',
#             device=device,
#             verbose=verbose,
#         )

#     def on_epoch_start(self):
#         self.model.eval()

#     def batch_update(self, x, label, mask):
#         with torch.no_grad():
#             prediction_mask, prediction_label = self.model.forward(x)
#             loss1 = self.loss['mask'](prediction_mask, mask)
#             loss2 = self.loss['label'](prediction_label, label)
#         prediction = {'mask': prediction_mask, 'label': prediction_label}
#         loss = {'mask': loss1, 'label': loss2}
#         return loss, prediction
    
    
# #######################################################
# # BYOL
# #######################################################

# def input_transformation_xy(x, y):
#     if len(x.shape)==5: ### rotation batch dim
#         x = x.view(-1, x.shape[-3], x.shape[-2], x.shape[-1]).float()
#     if len(y.shape)==3: ### rotation batch dim for label
#         y = y.view(-1, y.shape[-1]).float()
#     elif len(y.shape)==5: ### rotation batch dim for map
#         y = y.view(-1, y.shape[-3], y.shape[-2], y.shape[-1]).float()
#     elif len(y.shape)==4: ### normal segmentation
#         pass
#     else:
#         raise ValueError(f'y shape = {y.shape}')
#     return x, y
        
# class Epoch_BYOL(Epoch):

#     def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True):
#         super(Epoch_BYOL, self).__init__(model, loss, metrics, stage_name, device, verbose)
#         return

#     def _to_device(self):
#         self.model.to(self.device)
#         self.loss.to(self.device)
#         for metric in self.metrics:
#             metric.to(self.device)

#     def _format_logs(self, logs):
#         str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
#         s = ', '.join(str_logs)
#         return s

#     def batch_update(self, x, y):
#         raise NotImplementedError

#     def on_epoch_start(self):
#         pass

#     def run(self, dataloader):

#         self.on_epoch_start()

#         logs = {}
#         loss_meter = AverageValueMeter()
#         metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

#         with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
#             for f, x, y in iterator:
#                 x, y = x.to(self.device), y.to(self.device)
                
#                 x, y = input_transformation_xy(x, y)
                    
#                 loss = self.batch_update(x, y)

#                 # update loss logs
#                 loss_value = loss.cpu().detach().numpy()
#                 loss_meter.add(loss_value)
#                 loss_logs = {self.loss.__name__: loss_meter.mean}
#                 logs.update(loss_logs)

# #                 # update metrics logs
# #                 for metric_fn in self.metrics:
# #                     metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
# #                     metrics_meters[metric_fn.__name__].add(metric_value)
# #                 metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
# #                 logs.update(metrics_logs)

#                 if self.verbose:
#                     s = self._format_logs(logs)
#                     iterator.set_postfix_str(s)

#         return logs
    
# class TrainEpoch_BYOL(Epoch_BYOL):

#     def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True):
#         super().__init__(
#             model=model,
#             loss=loss,
#             metrics=metrics,
#             stage_name='train',
#             device=device,
#             verbose=verbose,
#         )
#         self.optimizer = optimizer

#     def on_epoch_start(self):
#         self.model.train()

#     def batch_update(self, x, y):
#         self.optimizer.zero_grad()
#         loss = self.model.forward(x)
# #         loss = self.loss(prediction, y)
# #         loss.backward()
#         self.optimizer.step()
#         return loss


# class ValidEpoch_BYOL(Epoch_BYOL):

#     def __init__(self, model, loss, metrics, device='cpu', verbose=True):
#         super().__init__(
#             model=model,
#             loss=loss,
#             metrics=metrics,
#             stage_name='valid',
#             device=device,
#             verbose=verbose,
#         )

#     def on_epoch_start(self):
#         self.model.eval()

#     def batch_update(self, x, y):
#         with torch.no_grad():
#             loss = self.model.forward(x)
#         return loss
