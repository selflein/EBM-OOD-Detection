from collections import defaultdict

import torch
from tqdm import tqdm
from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import distributions
import matplotlib.pyplot as plt

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.utils.utils import to_np
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class VAE(OODDetectionModel):
    def __init__(
        self,
        encoder_arch_name,
        encoder_arch_config,
        decoder_arch_name,
        decoder_arch_config,
        z_dim,
        learning_rate,
        momentum,
        weight_decay,
        test_ood_dataloaders=[],
        vis_every=-1,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder = get_arch(encoder_arch_name, encoder_arch_config)
        self.decoder = get_arch(decoder_arch_name, decoder_arch_config)

        self.z_dist = distributions.Normal(
            torch.zeros(z_dim), torch.diag(torch.ones(z_dim))
        )
        self.log_sigma_out = nn.Linear(z_dim, z_dim)
        self.mu_out = nn.Linear(z_dim, z_dim)

    def forward(self, num):
        z_sample = torch.randn(num, self.z_dim).to(self.device)
        return self.decoder(z_sample)

    def compute_errors(self, x):
        out = self.encoder(x)
        sigma = self.log_sigma_out(out).exp()
        mu = self.mu_out(out)

        kl_div = -0.5 * torch.sum(1 + sigma.pow(2).log() - mu.pow(2) - sigma.pow(2), 1)

        z = (torch.randn_like(mu) * sigma) + mu
        x_rec = self.decoder(z)

        return (
            F.mse_loss(x_rec, x, reduction="none").mean(list(range(1, len(x.shape)))),
            kl_div,
        )

    def step(self, batch):
        x, _ = batch
        rec_err, kl_div = self.compute_errors(x)
        return rec_err.mean(), kl_div.mean()

    def training_step(self, batch, batch_idx):
        loss = sum(self.step(batch))
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = sum(self.step(batch))
        self.log("val/loss", loss)

    def validation_epoch_end(self, training_step_outputs):
        if self.vis_every > 0 and self.current_epoch % self.vis_every == 0:
            interp = torch.linspace(-4, 4, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1).to(self.device)

            px = -to_np(self.compute_errors(data)[0])

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(to_np(x), to_np(y), px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("p(x)", fig, self.current_epoch)
            plt.close()

            fig, ax = plt.subplots()
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            samples = to_np(self.forward(100))
            ax.scatter(samples[:, 0], samples[:, 1])
            self.logger.experiment.add_figure("samples", fig, self.current_epoch)
            plt.close()

    def test_step(self, batch, batch_idx):
        loss = sum(self.step(batch))
        self.log("test/loss", loss)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)
        return [optim], [scheduler]

    def ood_detect(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        errs = []
        for x, _ in tqdm(loader):
            x = x.to(self.device)
            rec_err = self.compute_errors(x)[0]
            errs.append(rec_err)

        return -to_np(torch.cat(errs))
