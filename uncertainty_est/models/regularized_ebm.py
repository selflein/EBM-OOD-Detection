from collections import defaultdict

import torch
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch_lightning.core.decorators import auto_move_data

from uncertainty_est.utils.utils import (
    to_np,
    eval_func_on_grid,
    estimate_normalizing_constant,
)
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class RegularizedEBM(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        noise_sigma=1,
        clf_weight=0.0,
        grad_weight=1.0,
        noisy_regularizer=1.0,
        vis_every=-1,
        is_toy_dataset=False,
        toy_dataset_dim=2,
        test_ood_dataloaders=[],
    ):
        super().__init__(test_ood_dataloaders)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.backbone = get_arch(arch_name, arch_config)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_noisy = x + (torch.randn_like(x) * 1)

        x_noisy.requires_grad_()

        y_all = self.backbone(torch.cat([x, x_noisy]))
        y_hat = y_all[len(x) :]

        p_x = torch.exp(y_all).sum(1)

        # Maximize at position of data
        loss = -p_x[: len(x)].mean(0)
        self.log("train/px_loss", loss.item())

        # Minimize at noisy positions
        p_x_noise = self.noisy_regularizer * p_x[len(x) :].mean(0)
        self.log("train/px_noisy_loss", p_x_noise.item())
        loss += p_x_noise

        # Maximize gradient at positions around data
        grad_ld = (
            self.grad_weight
            * -(
                torch.autograd.grad(p_x.mean(), x_noisy, create_graph=True)[0]
                .flatten(start_dim=1)
                .norm(2, 1)
            ).mean()
        )
        self.log("train/grad_ld", grad_ld)
        loss += grad_ld

        if y_hat.shape[1] > 1 and self.clf_weight > 0:
            clf_loss = self.clf_weight * F.cross_entropy(y_hat, y)
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
            (x, y), p_xy = eval_func_on_grid(
                lambda x: torch.exp(self(x)),
                interval=(-4, 4),
                num_samples=500,
                device=self.device,
                dimensions=2,
            )
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
        self.log("test/accuracy", acc)
        return y_hat

    def test_epoch_end(self, logits):
        if self.is_toy_dataset:
            # Estimate normalizing constant Z by numerical integration
            log_Z = torch.log(
                estimate_normalizing_constant(
                    lambda x: self(x).exp().sum(1),
                    device=self.device,
                    dimensions=self.toy_dataset_dim,
                )
            ).float()

            logits = torch.cat(logits, 0)
            log_px = logits.logsumexp(1) - log_Z
            self.log("test/log_likelihood", log_px.mean())

        super().test_epoch_end()

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
