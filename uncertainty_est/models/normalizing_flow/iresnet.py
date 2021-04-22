import torch

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel


class IResNetFlow(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        warmup_steps=0,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()
        assert arch_name in ("iresnet_fc", "iresnet_conv")

        self.model = get_arch(arch_name, arch_config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x.requires_grad_()
        log_p = self.model.log_prob(x)

        loss = -log_p.mean()
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        with torch.enable_grad():
            x.requires_grad_()
            log_p = self.model.log_prob(x)

            sigmas = []
            for k, v in self.model.state_dict().items():
                if "_sigma" in k:
                    sigmas.append(v.item())
            sigmas = torch.tensor(sigmas)
            self.log("val/sigma_mean", sigmas.mean().item())

        loss = -log_p.mean()
        self.log("val/loss", loss)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        with torch.enable_grad():
            x.requires_grad_()
            log_p = self.model.log_prob(x)
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
        with torch.enable_grad():
            x.requires_grad_()
            log_p = self.model.log_prob(x).detach()
        return {"p(x)": log_p}
