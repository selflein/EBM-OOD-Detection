import torch

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel


class NormalizingFlow(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.density_estimation = get_arch(arch_name, arch_config)

    def forward(self, x):
        return self.density_estimation.log_prob(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        log_p = self.density_estimation.log_prob(x)

        loss = -log_p.mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        log_p = self.density_estimation.log_prob(x)

        loss = -log_p.mean()
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        log_p = self.density_estimation.log_prob(x)
        self.log("log_likelihood", log_p.mean())

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optim

    def get_ood_scores(self, x):
        return {"p(x)": self.density_estimation.log_prob(x)}
