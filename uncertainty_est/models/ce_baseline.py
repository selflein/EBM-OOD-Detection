import torch
import torch.nn.functional as F

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel


class CEBaseline(OODDetectionModel):
    def __init__(
        self, arch_name, arch_config, learning_rate, momentum, weight_decay, **kwargs
    ):
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.backbone = get_arch(arch_name, arch_config)

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("val/loss", loss)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("val/acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test/acc", acc)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)
        return [optim], [scheduler]

    def classify(self, x):
        return torch.softmax(self.backbone(x), -1)

    def get_ood_scores(self, x):
        logits = self(x).cpu()
        dir_uncert = dirichlet_prior_network_uncertainty(logits)
        dir_uncert["p(x)"] = logits.logsumexp(1)
        dir_uncert["max p(y|x)"] = logits.softmax(1).max(1)[0]
        return dir_uncert
