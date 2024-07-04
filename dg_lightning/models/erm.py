
import os
import typing
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from typing import Dict, List, Tuple, Union, Any

from dg_lightning.networks import NetworkInitializer
from dg_lightning.transforms import SupervisedLearningTransforms
from dg_lightning.optimization import (create_optimizer,
                                       create_learning_rate_scheduler)
from dg_lightning.metrics import MetricEvaluator
from dg_lightning.utils.lightning_utils import from_argparse_args


# FIXME: move
data2task = {
    'pacs': 'multiclass',
    'camelyon17': 'binary',
    'povertymap': 'regression',
    'iwildcam': 'multiclass',
    'rxrx1': 'multiclass',
}

# FIXME: move
task2loss = {
    'regression': F.mse_loss,
    'binary': F.binary_cross_entropy_with_logits,
    'multiclass': F.binary_cross_entropy,
}


class EmpiricalRiskMinimization(L.LightningModule):
    def __init__(
        self,
        data: str,
        backbone: str,
        pretrained: bool = False,
        augmentation: bool = False,
        randaugment: bool = False,
        optimizer: str = 'sgd',
        learning_rate: float = 3e-2,
        weight_decay: float = 1e-5,
        lr_scheduler: str = 'cosine_decay',
        max_epochs: int = 5,
        ) -> None:
        super().__init__()
        self.save_hyperparameters()                    # init arguments saved under self.hparams
        self.automatic_optimization = False            # manual optimization
        self.task: str = data2task[self.hparams.data]

        # (0) feature extractor
        self.encoder = \
            NetworkInitializer.initialize_backbone(
                name=self.hparams.backbone,
                data=self.hparams.data,
                pretrained=self.hparams.pretrained,
            )

        # (1) linear predictor (classifier or regressor)
        self.predictor = \
            nn.Linear(
                in_features=self.encoder.out_features,
                out_features=\
                    NetworkInitializer._out_features[self.hparams.data][0]
            )
        self.predictor.weight.data.normal_(mean=0., std=.02)
        self.predictor.bias.data.fill_(0.)

        # (2) input transforms
        self.train_transform: nn.Module = \
            SupervisedLearningTransforms[self.hparams.data](
                augmentation=self.hparams.augmentation,
                randaugment=self.hparams.randaugment,
            )
        self.eval_transform: nn.Module = \
            SupervisedLearningTransforms[self.hparams.data](
                augmentation=False,
                randaugment=False,
            )
        
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        return self._forward(x)

    def _forward(self, x: torch.Tensor) -> torch.FloatTensor:
        x = self.train_transform(x) if self.training else self.eval_transform(x)
        logits_or_resp = self.predictor(self.encoder(x))
        if (logits_or_resp.ndim == 2) and (logits_or_resp.size(1) == 1):
            logits_or_resp = logits_or_resp.squeeze(1)
        return logits_or_resp  # (B, J) or (B,  )
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = None) -> torch.FloatTensor:

        # fetch data & forward
        y, y_pred = self._shared_step(batch, training=True)

        # gradients & update
        opt = self.optimizers()
        loss = self.loss_function(y_pred, y)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        # update learning rate scheduler
        if self.trainer.is_last_batch:
            sch = self.lr_schedulers()
            if isinstance(sch, torch.optim.lr_scheduler._LRScheduler):
                sch.step()

        # log training loss
        self.log('train_loss', loss, on_step=True, reduce_fx='mean')

        # accumulate data for future use
        self.training_step_outputs.append(
            {
                'y': y,
                'y_pred': y_pred.detach(),
                'eval_group': batch['eval_group'],
            }
        )

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = None) -> torch.FloatTensor:
        
        # fetch data & forward
        y, y_pred = self._shared_step(batch, training=False)

        # accumulate data for future use
        self.validation_step_outputs.append(
            {
                'y': y,
                'y_pred': y_pred,
                'eval_group': batch['eval_group'],
            }
        )

        return self.loss_function(y, y_pred)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int = None) -> torch.FloatTensor:
        
        # fetch data & forward
        y, y_pred = self._shared_step(batch, training=False)

        # accumulate data for future use
        self.test_step_outputs.append(
            {
                'y': y,
                'y_pred': y_pred,
                'eval_group': batch['eval_group'],
            }
        )

        return self.loss_function(y, y_pred)

    def _shared_step(self, batch: Dict[str, torch.Tensor], training: bool = True) -> Tuple[torch.Tensor]:
        
        # fetch data
        x, y = batch['x'], batch['y']
        
        # apply input transforms
        if training:
            x = self.train_transform(x)
        else:
            x = self.eval_transform(x)
        
        # forward
        logits_or_resp = self.predictor(self.encoder(x))
        if (logits_or_resp.ndim == 2) and (logits_or_resp.size(1) == 1):
            logits_or_resp.squeeze_(1)

        return y, logits_or_resp

    def on_train_epoch_end(self) -> None:
        self._on_shared_epoch_end(self.training_step_outputs, prefix='train')
        self.training_step_outputs.clear()  # clear cache

    def on_validation_epoch_end(self) -> None:
        self._on_shared_epoch_end(self.validation_step_outputs, prefix='val')
        self.validation_step_outputs.clear()  # clear cache

    def on_test_epoch_end(self) -> None:
        self._on_shared_epoch_end(self.test_step_outputs, prefix='test')
        self.test_step_outputs.clear()  # clear cache

    def _on_shared_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], prefix: str) -> None:

        # concatenate batch outputs
        y = torch.cat([out['y'] for out in outputs], dim=0)
        y_pred = torch.cat([out['y_pred'] for out in outputs], dim=0)
        y_pred = self._process_pred_for_eval(y_pred)
        eval_group = torch.cat([out['eval_group'] for out in outputs], dim=0)

        # evaluate metrics
        evaluator = MetricEvaluator(data=self.hparams.data)
        metrics = evaluator(y_pred, y, eval_group)
        self.log_dict(
            {f'{prefix}_{n}': v for n, v in metrics.items()},
            prog_bar=True,
        )

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer],
                                            List[torch.optim.lr_scheduler._LRScheduler]]:
        
        optimizer = \
            create_optimizer(
                params=filter(lambda p: p.requires_grad, self.parameters()),
                name=self.hparams.optimizer,
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        
        lr_scheduler = \
            create_learning_rate_scheduler(
                optimizer=optimizer,
                name=self.hparams.lr_scheduler,
                epochs=self.hparams.max_epochs,
            )

        return [optimizer], [lr_scheduler]

    def loss_function(self,
                      y_pred: torch.FloatTensor,
                      y_true: typing.Union[torch.FloatTensor, torch.LongTensor],
                      **kwargs) -> torch.FloatTensor:
        loss_fx: callable = task2loss[self.task]
        if self.task != 'multiclass':
            y_true = y_true.float()
        return loss_fx(y_pred, y_true, **kwargs)

    def _process_pred_for_eval(self, y_pred: torch.FloatTensor) -> torch.FloatTensor:
        """
        Helper function to modify predictions into
        the right format expected by `MetricEvaluator.__call__`.
        """
        if self.task == 'regression':
            return y_pred
        elif self.task == 'binary':
            return torch.sigmoid(y_pred)
        elif self.task == 'multiclass':
            return torch.softmax(y_pred, dim=1)
        else:
            raise NotImplementedError

    @classmethod
    def from_argparse_args(cls,
                           args: Union[argparse.Namespace, Dict[str, Any]],
                           **kwargs) -> L.LightningModule:
        return from_argparse_args(cls, args, **kwargs)

    @classmethod
    def add_model_specific_args(cls,
                                parent_parser: argparse.ArgumentParser,
                                ) -> argparse.ArgumentParser:
        
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

        group = parser.add_argument_group(f"{cls.__name__}")
        group.add_argument('--data', type=str, default='camelyon17', help='')
        group.add_argument('--backbone', type=str, default='densnet121', help='')
        group.add_argument('--pretrained', action='store_true', help='')
        group.add_argument('--augmentation', action='store_true', help='')
        group.add_argument('--randaugment', action='store_true', help='')
        group.add_argument('--optimizer', type=str, default='sgd', help='')
        group.add_argument('--learning_rate', type=float, default=3e-2, help='')
        group.add_argument('--weight_decay', type=float, default=1e-5, help='')
        group.add_argument('--lr_scheduler', type=str, default=None, help='')

        return parser