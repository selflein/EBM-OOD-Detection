import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F

from uncertainty_est.models.JEM.model import HDGE
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.priornet.dpn_losses import dirichlet_kl_divergence
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)
from uncertainty_est.models.JEM.utils import (
    KHotCrossEntropyLoss,
    smooth_one_hot,
)


class HDGEPriorNetModel(pl.LightningModule):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        pyxce,
        pxcontrast,
        pxycontrast,
        smoothing,
        n_classes,
        contrast_k,
        contrast_t,
        target_concentration,
        alpha_fix,
        kl_weight,
        concentration,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        arch = get_arch(arch_name, arch_config)
        self.model = HDGE(arch, n_classes, contrast_k, contrast_t)

    def forward(self, x):
        return self.model.classify(x)

    def compute_losses(self, x_lab, dist, logits=None, evaluation=False):
        l_pyxce, l_pxcontrast, l_pxycontrast = 0.0, 0.0, 0.0
        # log p(y|x) cross entropy loss
        if self.pyxce > 0:
            if logits is None:
                logits = self.model.classify(x_lab)
            l_pyxce = KHotCrossEntropyLoss()(logits, dist)
            l_pyxce *= self.pyxce

        # log p(x) using contrastive learning
        if self.pxcontrast > 0:
            # ones like dist to use all indexes
            ones_dist = torch.ones_like(dist).to(self.device)
            output, target, _, _ = self.model.joint(
                img=x_lab, dist=ones_dist, evaluation=evaluation
            )
            l_pxcontrast = F.cross_entropy(output, target)
            l_pxcontrast *= self.pxycontrast

        # log p(x|y) using contrastive learning
        if self.pxycontrast > 0:
            output, target, _, _ = self.model.joint(
                img=x_lab, dist=dist, evaluation=evaluation
            )
            l_pxycontrast = F.cross_entropy(output, target)
            l_pxycontrast *= self.pxycontrast

        return l_pyxce, l_pxcontrast, l_pxycontrast

    def training_step(self, batch, batch_idx, evaluation=False):
        x_lab, y_lab = batch
        dist = smooth_one_hot(y_lab, self.n_classes, self.smoothing)
        logits = self.model.classify(x_lab)

        loss = sum(
            self.compute_losses(x_lab, dist, logits=logits, evaluation=evaluation)
        )

        alphas = torch.exp(logits)
        if self.alpha_fix:
            alphas = alphas + 1

        if self.target_concentration is None:
            target_concentration = torch.exp(-self.model(x_lab)) + self.concentration
        else:
            target_concentration = (
                torch.empty(len(alphas))
                .fill_(self.target_concentration)
                .to(self.device)
            )

        target_alphas = torch.empty_like(alphas).fill_(self.concentration)
        target_alphas[torch.arange(len(y_lab)), y_lab] = target_concentration
        kl_term = dirichlet_kl_divergence(target_alphas, alphas)
        loss += self.kl_weight * kl_term.mean()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.classify(x)

        val_loss = self.training_step(batch, batch_idx, evaluation=True)
        self.log("val_loss", val_loss)

        acc = (y == logits.argmax(1)).float().mean(0).item()
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)
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
        uncert = {}

        px, logits = [], []
        for x, _ in tqdm(loader):
            x = x.to(self.device)
            px.append(torch.exp(-self.model(x).cpu()))
            logits.append(self.model.classify(x).cpu().numpy())
        uncert["p(x)"] = torch.cat(px)
        dirichlet_uncerts = dirichlet_prior_network_uncertainty(
            np.concatenate(logits), alpha_correction=self.alpha_fix
        )
        uncert = {**uncert, **dirichlet_uncerts}

        return uncert
