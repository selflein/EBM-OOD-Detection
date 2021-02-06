from collections import defaultdict

import torch
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_lightning.core.decorators import auto_move_data

from uncertainty_est.utils.utils import to_np
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class RegularizedEBM(pl.LightningModule):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        vis_every=-1,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.backbone = get_arch(arch_name, arch_config)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_noisy = x + (torch.randn_like(x) * 5)

        x_noisy.requires_grad_()

        y_all = self.backbone(torch.cat([x, x_noisy]))
        y_hat = y_all[len(x) :]

        p_x = torch.exp(y_all).sum(1)

        # Maximize at position of data
        loss = -p_x[: len(x)].mean(0)
        self.log("train/px", loss.item())

        p_x_noise = p_x[len(x) :].mean(0)
        loss += p_x_noise
        self.log("train/px_noise", p_x_noise.item())

        # Maximize gradient at positions around data
        grad_ld = (
            torch.autograd.grad(p_x.mean(), x_noisy, create_graph=True)[0]
            .flatten(start_dim=1)
            .norm(2, 1)
        ).mean()
        self.log("train/grad_ld", grad_ld)
        loss -= 10 * grad_ld

        if y_hat.shape[1] > 1:
            clf_loss = 10 * F.cross_entropy(y_hat, y)
            loss += clf_loss
            self.log("train/clf_loss", clf_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        if y_hat.shape[1] > 1:
            acc = (y == y_hat.argmax(1)).float().mean(0).item()
            self.log("val_acc", acc)

        return y_hat

    def validation_epoch_end(self, outputs):
        if self.vis_every > 0 and self.current_epoch % self.vis_every == 0:
            interp = torch.linspace(-4, 4, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1)
            p_xy = torch.exp(self(data.to(self.device)))
            px = to_np(p_xy.sum(1))

            x, y = to_np(x), to_np(y)
            for i in range(p_xy.shape[1]):
                fig, ax = plt.subplots()
                mesh = ax.pcolormesh(x, y, to_np(p_xy[:, i]).reshape(*x.shape))
                fig.colorbar(mesh)
                self.logger.experiment.add_figure(
                    f"dist/p(x,y={i})", fig, self.current_epoch
                )
                plt.close()

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(x, y, px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("dist/p(x)", fig, self.current_epoch)
            plt.close()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

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
        _, logits = self.get_gt_preds(loader)

        dir_uncert = dirichlet_prior_network_uncertainty(
            logits.cpu().numpy(),
        )
        dir_uncert["p(x)"] = logits.logsumexp(1).numpy()
        return dir_uncert
