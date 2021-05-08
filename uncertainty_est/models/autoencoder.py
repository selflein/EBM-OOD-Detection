import torch

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel


class Autoencoder(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        decoder_arch_name,
        decoder_arch_config,
        learning_rate,
        momentum,
        weight_decay,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.encoder = get_arch(arch_name, arch_config)
        self.decoder = get_arch(decoder_arch_name, decoder_arch_config)

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_rec = self(x)

        loss = torch.norm((x - x_rec).flatten(start_dim=1), p=2, dim=1).mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_rec = self(x)

        loss = torch.norm((x - x_rec).flatten(start_dim=1), p=2, dim=1).mean()
        self.log("val/loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, _ = batch
        x_rec = self(x)

        loss = torch.norm((x - x_rec).flatten(start_dim=1), p=2, dim=1).mean()
        self.log("test/loss", loss)
        return loss

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
        x_rec = self(x)
        rec_err = torch.norm((x - x_rec).flatten(start_dim=1), p=2, dim=1)
        return {"Reconstruction error": rec_err}
