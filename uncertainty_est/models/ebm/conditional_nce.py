import torch
from torch import distributions

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel


class PerSampleNCE(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        noise_sigma=0.01,
        p_control_weight=0.0,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.model = get_arch(arch_name, arch_config)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        sample_shape = x.shape[1:]
        sample_dim = sample_shape.numel()
        noise_dist = distributions.multivariate_normal.MultivariateNormal(
            torch.zeros(sample_dim).to(self.device),
            torch.eye(sample_dim).to(self.device) * self.noise_sigma,
        )
        noise = noise_dist.sample(x.size()[:1])

        # Implements Eq. 9 in "Conditional Noise-Contrastive Estimation of Unnormalised Models"
        # Uses symmetry of noise distribution meaning p(u1|u2) = p(u2|u1) to simplify
        # Sets k = 1
        x_noisy = x + noise.reshape_as(x)
        log_p_model = self.model(torch.cat((x, x_noisy))).squeeze()
        log_p_x = log_p_model[: len(x)]
        log_p_x_noisy = log_p_model[len(x) :]

        loss = torch.log(1 + (-(log_p_x - log_p_x_noisy)).exp()).mean()

        p_control = log_p_model.abs().mean()
        loss += self.p_control_weight * p_control

        self.log("train/log_p_magnitude", log_p_x.mean(), prog_bar=True)
        self.log("train/log_p_noisy_magnitude", log_p_x_noisy.mean(), prog_bar=True)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return

    def test_step(self, batch, batch_idx):
        self.to(torch.float32)
        x, y = batch
        y_hat = self.model(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)
        return y_hat

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
        return {"p(x)": self.model(x)}
