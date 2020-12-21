from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)
from uncertainty_est.models.priornet.dpn_losses import (
    DirichletKLLoss,
    PriorNetMixedLoss,
)


class PriorNet(pl.LightningModule):
    def __init__(
        self, arch_name, arch_config, learning_rate, momentum, weight_decay, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.__dict__.update(kwargs)

        arch = get_arch(arch_name, arch_config)
        self.backbone = arch
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

        id_criterion = DirichletKLLoss(
            target_concentration=self.target_concentration,
            concentration=self.concentration,
            reverse=self.reverse_kl,
            alpha_fix=self.alpha_fix,
        )

        ood_criterion = DirichletKLLoss(
            target_concentration=0.0,
            concentration=self.concentration,
            reverse=self.reverse_kl,
            alpha_fix=self.alpha_fix,
        )

        self.criterion = PriorNetMixedLoss(
            [id_criterion, ood_criterion], mixing_params=[1.0, self.gamma]
        )

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        (x, y), (x_ood, _) = batch

        y_hat = self(torch.cat((x, x_ood)))
        y_hat_ood = y_hat[len(x) :]
        y_hat = y_hat[: len(x)]

        loss = self.criterion((y_hat, y_hat_ood), (y, None))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)
        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        (x, y), (_, _) = batch

        y_hat = self(x)
        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=0.5)
        return [optim], [scheduler]

    def get_gt_preds(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        gt, preds = [], []
        for x, y in tqdm(loader):
            x = x.to(self.device)
            y_hat = self(x).cpu()
            gt.append(y)
            preds.append(y_hat)
        return torch.cat(gt), torch.cat(preds)

    def ood_detect(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        logits = []
        for x, _ in tqdm(loader):
            x = x.to(self.device)
            logit = self(x).cpu().numpy()
            logits.append(logit)

        logits = np.concatenate(logits)
        uncertanties = dirichlet_prior_network_uncertainty(
            logits, alpha_correction=self.alpha_fix
        )
        return uncertanties
