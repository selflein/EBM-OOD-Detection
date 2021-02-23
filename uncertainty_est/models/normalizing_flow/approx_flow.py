from collections import defaultdict

import torch
from tqdm import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F

from uncertainty_est.utils.utils import to_np, estimate_normalizing_constant
from uncertainty_est.models.normalizing_flow.norm_flow import NormalizingFlow


class ApproxNormalizingFlow(NormalizingFlow):
    def __init__(
        self,
        density_type,
        latent_dim,
        n_density,
        learning_rate,
        momentum,
        weight_decay,
        vis_every=-1,
        is_toy_dataset=True,
        toy_dataset_dim=2,
        test_ood_dataloaders=[],
        weight_penalty_weight=1.0,
    ):
        super().__init__(
            density_type,
            latent_dim,
            n_density,
            learning_rate,
            momentum,
            weight_decay,
            vis_every,
            test_ood_dataloaders,
        )
        assert density_type in ("orthogonal_flow", "reparameterized_flow")
        self.__dict__.update(locals())
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        log_p = self.density_estimation.log_prob(x)

        loss = -log_p.mean()
        self.log("train/ml_loss", loss, on_epoch=True)

        weight_penalty = 0.0
        transforms = self.density_estimation.transforms
        for t in transforms:
            weight_penalty += t.compute_weight_penalty()
        self.log("train/weight_penalty", weight_penalty, on_epoch=True)
        loss += self.weight_penalty_weight * weight_penalty

        return loss

    def validation_epoch_end(self, outputs):
        if self.is_toy_dataset:
            log_Z = estimate_normalizing_constant(
                lambda x: self.density_estimation.log_prob(x),
                device=self.device,
                dimensions=self.toy_dataset_dim,
                dtype=torch.float32,
            )
            self.log("val/log_norm_constant", log_Z)

        super().validation_epoch_end(outputs)
