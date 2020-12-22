import torch
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F

from uncertainty_est.models.JEM.model import HDGE
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.JEM.utils import (
    KHotCrossEntropyLoss,
    smooth_one_hot,
)


class HDGEModel(pl.LightningModule):
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
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        arch = get_arch(arch_name, arch_config)
        self.model = HDGE(arch, n_classes, contrast_k, contrast_t)

    def forward(self, x):
        return self.model.classify(x)

    def training_step(self, batch, batch_idx):
        x_lab, y_lab = batch
        dist = smooth_one_hot(y_lab, self.n_classes, self.smoothing)

        loss = 0.0
        # log p(y|x) cross entropy loss
        if self.pyxce > 0:
            logits = self.model.classify(x_lab)
            l_pyxce = KHotCrossEntropyLoss()(logits, dist)
            loss += self.pyxce * l_pyxce

        # log p(x) using contrastive learning
        if self.pxcontrast > 0:
            # ones like dist to use all indexes
            ones_dist = torch.ones_like(dist).to(self.device)
            output, target, _, _ = self.model.joint(img=x_lab, dist=ones_dist)
            l_pxcontrast = F.cross_entropy(output, target)
            loss += self.pxycontrast * l_pxcontrast

        # log p(x|y) using contrastive learning
        if self.pxycontrast > 0:
            output, target, _, _ = self.model.joint(img=x_lab, dist=dist)
            l_pxycontrast = F.cross_entropy(output, target)
            loss += self.pxycontrast * l_pxycontrast
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
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

        px = []
        for x, _ in tqdm(loader):
            x = x.to(self.device)
            px.append(self.model(x).cpu())
        uncert["p(x)"] = -torch.cat(px)
        return uncert
