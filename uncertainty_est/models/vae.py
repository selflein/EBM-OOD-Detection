import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel


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
        **kwargs
    ):
        super().__init__(**kwargs)
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

    def get_ood_scores(self, x):
        x = x.to(self.device)
        rec_err = self.compute_errors(x)[0]

        enc = self.encoder(x)
        mu = self.mu_out(enc)

        return {
            "Reconstruction error": rec_err,
            "Encoder errors": torch.linalg.norm(mu, 2, dim=-1),
        }
