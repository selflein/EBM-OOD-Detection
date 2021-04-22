import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import distributions
import matplotlib.pyplot as plt

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.utils.utils import (
    to_np,
    estimate_normalizing_constant,
    sum_except_batch,
)


class NoiseContrastiveEstimation(OODDetectionModel):
    """Implementation of Noise Contrastive Estimation http://proceedings.mlr.press/v9/gutmann10a.html"""

    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        noise_distribution="uniform",
        noise_distribution_kwargs={"low": 0, "high": 1},
        **kwargs,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.model = get_arch(arch_name, arch_config)

        if noise_distribution == "uniform":
            noise_dist = distributions.Uniform
        if noise_distribution == "gaussian":
            noise_dist = distributions.Normal
        else:
            raise NotImplementedError(
                f"Requested noise distribution {noise_distribution} not implemented."
            )

        self.dist_parameters = torch.nn.ParameterDict(
            {
                k: torch.nn.Parameter(torch.tensor(v).float(), requires_grad=False)
                for k, v in noise_distribution_kwargs.items()
            }
        )
        self.noise_dist = noise_dist(**self.dist_parameters)

    def forward(self, x):
        return self.model(x)

    def compute_ebm_loss(self, batch, return_outputs=False):
        x, _ = batch
        noise = self.noise_dist.sample(x.shape).to(self.device)
        inp = torch.cat((x, noise))

        logits = self.model(inp)
        log_p_model = logits.logsumexp(-1)
        log_p_noise = sum_except_batch(self.noise_dist.log_prob(inp))

        loss = F.binary_cross_entropy_with_logits(
            log_p_model - log_p_noise,
            torch.cat((torch.ones(len(x)), torch.zeros(len(x)))).to(self.device),
        )
        if return_outputs:
            return loss, logits[: len(x)]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_ebm_loss(batch)

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
