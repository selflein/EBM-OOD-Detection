from collections import defaultdict

import torch
from tqdm import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torch.nn.functional as F

from uncertainty_est.utils.utils import to_np
from uncertainty_est.models.normalizing_flow.flows import NormalizingFlowDensity
from uncertainty_est.models.ood_detection_model import OODDetectionModel


class NormalizingFlow(OODDetectionModel):
    def __init__(
        self,
        density_type,
        latent_dim,
        n_density,
        learning_rate,
        momentum,
        weight_decay,
        vis_every=-1,
        test_ood_dataloaders=[],
    ):
        super().__init__(test_ood_dataloaders)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.density_estimation = NormalizingFlowDensity(
            dim=self.latent_dim, flow_length=n_density, flow_type=self.density_type
        )

    def forward(self, x):
        return self.density_estimation(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        log_p = self.density_estimation.log_prob(x)

        loss = -log_p.mean()
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        log_p = self.density_estimation.log_prob(x)

        loss = -log_p.mean()
        self.log("val_loss", loss)

    def validation_epoch_end(self, training_step_outputs):
        if self.vis_every > 0 and self.current_epoch % self.vis_every == 0:
            interp = torch.linspace(-4, 4, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1)

            px = to_np(
                torch.exp(self.density_estimation.log_prob(data.to(self.device)))
            )

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(to_np(x), to_np(y), px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("p(x)", fig, self.current_epoch)
            plt.close()

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

    def ood_detect(self, loader):
        with torch.no_grad():
            log_p = []
            for x, _ in loader:
                x = x.to(self.device)
                log_p.append(self.density_estimation.log_prob(x))
        log_p = torch.cat(log_p)

        dir_uncert = {}
        dir_uncert["log p(x)"] = log_p.cpu().numpy()
        dir_uncert["p(x)"] = log_p.exp().cpu().numpy()
        return dir_uncert
