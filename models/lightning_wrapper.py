import comet_ml
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn

import lightning.pytorch as pl
from torch import Tensor
from torch import functional as F
from torchvision import models

from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection

from .vig import ViG, PyramidViG
from typing import Any, Optional, Union, Tuple, Dict, List, Type

class ViGLT(pl.LightningModule):
    def __init__(self,
                in_channels: int,
                out_channels: List[int],
                heads: int,
                n_classes: int,
                input_resolution: Tuple[int, int],
                reduce_factor: int, act: str='relu',
                k: int=9,
                overlapped_patch_emb: bool=True,
                enable_pos_encoding: bool=True,
                task: str='classification',
                **kwargs) -> None:
        super(ViGLT, self).__init__()
        self.model = ViG(in_channels,
                        out_channels,
                        heads,
                        n_classes,
                        input_resolution,
                        reduce_factor,
                        act,
                        k,
                        overlapped_patch_emb,
                        task,
                        enable_pos_encoding=enable_pos_encoding,
                        **kwargs)
        self.loss = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=n_classes)

        self.train_count = 0
        self.val_count = 0
        return

    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.model(x).squeeze(-1).squeeze(-1)
        loss = self.loss(out, y)
        
        self.train_count += 1
        if self.train_count % 10 == 0:
            self.train_count = 0
            acc = self.acc(out, y)
            print(f'Train loss: {loss} - Accuracy: {acc}')
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        out = self.model(x).squeeze(-1).squeeze(-1)
        loss = self.loss(out, y)

        self.val_count += 1
        if self.val_count % 10 == 0:
            self.val_count = 0
            acc = self.acc(out, y)
            print(f'Validation loss: {loss} - Acc: {acc}')
        return

    def backward(self, loss: Tensor, optimizer, optimizer_idx) -> None:
        loss.backward()
        return

class PyramidViGLT(pl.LightningModule):
    def __init__(self,
                in_channels: int,
                out_channels: List[int],
                heads: int,
                n_classes: int,
                input_resolution: Tuple[int, int],
                reduce_factor: int,
                pyramid_reduction: int=2,
                act: str = 'relu',
                k: int = 9,
                overlapped_patch_emb: bool = True,
                enable_pos_encoding: bool=True,
                lr: float=1e-4,
                metric_args: Dict=None,
                wd: float=1e-2,
                **kwargs) -> None:
        super(PyramidViGLT, self).__init__()
        self.lr = lr
        self.wd = wd
        self.model = PyramidViG(in_channels,
                                out_channels,
                                heads,
                                n_classes,
                                input_resolution,
                                reduce_factor,
                                pyramid_reduction,
                                act,
                                k,
                                overlapped_patch_emb,
                                enable_pos_encoding=enable_pos_encoding,
                                **kwargs)
        self.metric_args = {} if metric_args is None else metric_args
        if 'task' in self.metric_args and \
            self.metric_args['task'] == 'multilabel':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.create_metrics()
        self.save_hyperparameters()
        
        return
    
    def create_metrics(self) -> None:
        m = {
            'accuracy': Accuracy(**self.metric_args),
            'precision': Precision(**self.metric_args),
            'recall': Recall(**self.metric_args),
            'f1': F1Score(**self.metric_args)
        }
        metric = MetricCollection(m)
        
        self.train_metrics = metric.clone(prefix='train_')
        self.val_metric = metric.clone(prefix='val_')
        self.test_metric = metric.clone(prefix='test_')
        return
    
    def forward(self, x) -> Tensor:
        x = self.model(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}]

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch['image'], train_batch['label']
        out = self.model(x)
        if self.model.task == 'classification':
            out = out.squeeze(-1).squeeze(-1)
        if 'task' in self.metric_args and \
            self.metric_args['task'] == 'multilabel':
            y = y.float()
        loss = self.loss(out, y)
        
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.train_metrics.update(out, y)
        return loss
    
    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()
        return
        
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['image'], val_batch['label']
        out = self.model(x)
        if self.model.task == 'classification':
            out = out.squeeze(-1).squeeze(-1)
        if 'task' in self.metric_args and \
            self.metric_args['task'] == 'multilabel':
            y = y.float()
        loss = self.loss(out, y)

        self.log('val_loss', loss)
        self.val_metric.update(out, y)
        return
    
    def on_validation_epoch_end(self):
        metrics = self.val_metric.compute()
        self.log_dict(metrics)
        self.val_metric.reset()
        return

    def backward(self, loss: Tensor) -> None:
        loss.backward()
        return
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch['image'], test_batch['label']
        out = self.model(x)
        if self.model.task == 'classification':
            out = out.squeeze(-1).squeeze(-1)
        if 'task' in self.metric_args and \
            self.metric_args['task'] == 'multilabel':
            y = y.float()
        loss = self.loss(out, y)

        self.log('test_loss', loss, on_epoch=True, on_step=True)
        self.test_metric.update(out, y)
        return
    
    def on_test_epoch_end(self):
        metric = self.test_metric.compute()
        self._log_dict(metric)
        self.test_metric.reset()
        return
    
    def linearize_dict(self, d: Dict[str, Tensor]) -> Dict[str, Tensor]:
        res = {}
        for k, v in d.items():
            v = v.cpu()
            n_kls = v.numel()
            if n_kls > 1:
                for idx in range(n_kls):
                    res[f'{k}_{idx}'] = v[idx]
            else:
                if len(v.shape) > 0:
                    res[k] = v[0]
                else:
                    res[k] = v.item()
        return res
    
    def _log_dict(self, d: Dict[str, Tensor]) -> None:
        if self.model.n_classes > 1:
            self.log_dict(self.linearize_dict(d))
        else:
            self.log_dict(d)
        return
    
class ResNetLT(pl.LightningModule):
    def __init__(self, resnet: str, lr: float, in_channels: int, n_classes: int, metric_args: Dict=None) -> None:
        super(ResNetLT, self).__init__()
        self.resnet = resnet
        self.metric_args = metric_args

        if resnet == 'resnet50':
            self.model = models.resnet.resnet50()
        elif resnet == 'resnet18':
            self.model = models.resnet.resnet18()
        elif resnet == 'resnet34':
            self.model = models.resnet.resnet34()
        elif resnet == 'resnet101':
            self.model = models.resnet.resnet101()
        elif resnet == 'resnet152':
            self.model = models.resnet152()
        else:
            raise ValueError(f'Invalid resnet model specified {resnet}')

        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=n_classes)

        self.lr = lr
        self.in_ch = in_channels
        self.nclasses = n_classes
        if 'task' in self.metric_args and \
            self.metric_args['task'] == 'multilabel':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            self.loss = nn.CrossEntropyLoss()
        self.create_metrics(metrics_args=metric_args)
        self.save_hyperparameters()
        return
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, threshold=1e-3)
        return [optimizer], [{'scheduler': scheduler, 'monitor': 'val_loss', 'interval': 'epoch'}]
    
    def backward(self, loss: Tensor) -> None:
        loss.backward()
        return
    
    def create_metrics(self, metrics_args: Optional[Dict]=None) -> None:
        if metrics_args is None:
            metrics_args = {}
        # TODO change this
        metrics = {
            'accuracy': Accuracy(**metrics_args),
            'precision': Precision(**metrics_args),
            'recall': Recall(**metrics_args),
            'f1': F1Score(**metrics_args)
        }
        m = MetricCollection(metrics)
        self.train_metrics = m.clone(prefix='train_')
        self.val_metrics = m.clone(prefix='val_')
        self.test_metrics = m.clone(prefix='test_')
        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, train_batch, batch_idx):
        x, y = train_batch['image'], train_batch['label']
        out = self.model(x)
        if 'task' in self.metric_args and \
            self.metric_args['task'] == 'multilabel':
            y = y.float()
        loss = self.loss(out, y)

        self.log('train_loss', loss, on_epoch=True, on_step=True)
        self.train_metrics.update(out, y)
        return loss
    
    def linearize_dict(self, d: Dict[str, Tensor]) -> Dict[str, Tensor]:
        res = {}
        for k, v in d.items():
            v = v.cpu()
            n_kls = v.numel()
            if n_kls > 1:
                for idx in range(n_kls):
                    res[f'{k}_{idx}'] = v[idx]
            else:
                if len(v.shape) > 0:
                    res[k] = v[0]
                else:
                    res[k] = v.item()
        return res
    
    # def _log_dict(self, d: Dict[str, Tensor]) -> None:
    #     if self.nclasses > 1:
    #         self.log_dict(self.linearize_dict(d))
    #     else:
    #         self.log_dict(d)
    #     return
    
    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute()
        self.log_dict(metrics)
        self.train_metrics.reset()
        return
    
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['image'], val_batch['label']
        out = self.model(x)
        if 'task' in self.metric_args and \
            self.metric_args['task'] == 'multilabel':
            y = y.float()
        loss = self.loss(out, y)

        self.log('val_loss', loss)
        self.val_metrics.update(out, y)
        return
    
    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()
        return
    
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch['image'], test_batch['label']
        out = self.model(x)
        if 'task' in self.metric_args and \
            self.metric_args['task'] == 'multilabel':
            y = y.float()
        loss = self.loss(out, y)

        self.log('test_loss', loss)
        self.test_metrics.update(out, y)
        return
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()
        return