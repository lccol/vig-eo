import torch

from torch import nn
from timm import models
from torchmetrics import Accuracy, Recall, Precision, F1Score, MetricCollection
from lightning import pytorch as pl
from typing import Union, Dict, List, Tuple, Optional, Any

class ViTLT(pl.LightningModule):
    def __init__(self,
        vit_type: str,
        lr: float,
        in_channels: int,
        n_classes: int,
        patch_size: int=8,
        img_size: int=120,
        metric_args: Optional[Dict]=None,
        **kwargs
    ) -> None:
        super(ViTLT, self).__init__()
        self.vit_type = vit_type
        self.lr = lr
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.img_size = img_size
        self.metric_args = metric_args

        model_kwargs = {
            'in_chans': in_channels,
            'patch_size': patch_size,
            'num_classes': n_classes,
            'img_size': img_size
        }
        if vit_type == 'tiny':
            model_kwargs['embed_dim'] = 192
            model_kwargs['depth'] = 12
            model_kwargs['num_heads'] = 3
        elif vit_type == 'base':
            model_kwargs['embed_dim'] = 384
            model_kwargs['depth'] = 12
            model_kwargs['num_heads'] = 6
        else:
            raise ValueError(f'Invalid vit model specified {vit_type}')

        self.model = models.vision_transformer.VisionTransformer(**model_kwargs)
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
    
    def backward(self, loss: torch.Tensor) -> None:
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
    
    def linearize_dict(self, d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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
    
    def _log_dict(self, d: Dict[str, torch.Tensor]) -> None:
        if self.n_classes > 1:
            self.log_dict(self.linearize_dict(d))
        else:
            self.log_dict(d)
        return
    
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