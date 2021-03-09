from collections import defaultdict

import torch
from torch import nn
from tqdm import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F

from uncertainty_est.utils.utils import to_np
from uncertainty_est.archs.real_nvp.real_nvp import RealNVP
from uncertainty_est.models.ood_detection_model import OODDetectionModel


class RealNVPModel(OODDetectionModel):
    def __init__(
        self,
        num_scales,
        in_channels,
        mid_channels,
        num_blocks,
        learning_rate,
        momentum,
        weight_decay,
        num_classes=1,
        vis_every=-1,
        test_ood_dataloaders=[],
    ):
        super().__init__(test_ood_dataloaders)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.conditional_densities = nn.ModuleList()
        for _ in range(num_classes):
            self.conditional_densities.append(
                RealNVP(num_scales, in_channels, mid_channels, num_blocks)
            )

    def forward(self, x):
        log_p_xy = []
        for cd in self.conditional_densities:
            log_p_xy.append(cd.log_prob(x))
        log_p_xy = torch.stack(log_p_xy, 1)

        return log_p_xy

    def training_step(self, batch, batch_idx):
        x, y = batch
        log_p_xy = self(x)
        log_p_x = torch.logsumexp(log_p_xy, dim=1)

        if self.num_classes > 1:
            loss = F.cross_entropy(log_p_xy, y)
            self.log("train/clf_loss", loss)
        else:
            loss = -log_p_x.mean()
            self.log("train/loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        log_p_xy = self(x)
        log_p_x = torch.logsumexp(log_p_xy, dim=1)

        loss = -log_p_x.mean()
        self.log("val/loss", loss)

        acc = (y == log_p_xy.argmax(1)).float().mean(0).item()
        self.log("val/acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        log_p_xy = self(x)
        log_p_x = torch.logsumexp(log_p_xy, dim=1)
        self.log("log_likelihood", log_p_x.mean())

        acc = (y == log_p_xy.argmax(1)).float().mean(0).item()
        self.log("acc", acc)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optim

    def ood_detect(self, loader):
        with torch.no_grad():
            log_p = []
            for x, _ in tqdm(loader):
                x = x.to(self.device)
                log_p_xy = self(x)
                log_p_x = torch.logsumexp(log_p_xy, dim=1)
                log_p.append(log_p_x)
        log_p = torch.cat(log_p)

        dir_uncert = {}
        dir_uncert["p(x)"] = log_p.cpu().numpy()
        return dir_uncert
