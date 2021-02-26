from collections import defaultdict

import torch
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import distributions
import matplotlib.pyplot as plt

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.utils.utils import to_np, estimate_normalizing_constant


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
        is_toy_dataset=False,
        toy_dataset_dim=2,
        test_ood_dataloaders=[],
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.model = get_arch(arch_name, arch_config)

    def setup(self, phase):
        self.len_ds = torch.tensor(
            float(len(self.train_dataloader.dataloader.dataset)), requires_grad=False
        )

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

    def validation_epoch_end(self, outputs):
        if self.is_toy_dataset and self.toy_dataset_dim == 2:
            interp = torch.linspace(-4, 4, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1)
            p_xy = torch.exp(self(data.to(self.device)))
            px = to_np(p_xy.sum(1))

            x, y = to_np(x), to_np(y)
            for i in range(p_xy.shape[1]):
                fig, ax = plt.subplots()
                mesh = ax.pcolormesh(x, y, to_np(p_xy[:, i]).reshape(*x.shape))
                fig.colorbar(mesh)
                self.logger.experiment.add_figure(
                    f"dist/p(x,y={i})", fig, self.current_epoch
                )
                plt.close()

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(x, y, px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("dist/p(x)", fig, self.current_epoch)
            plt.close()

    def test_step(self, batch, batch_idx):
        self.to(torch.float32)
        x, y = batch
        y_hat = self.model(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)
        return y_hat

    def test_epoch_end(self, logits):
        if self.is_toy_dataset:
            # Estimate normalizing constant Z by numerical integration
            log_Z = torch.log(
                estimate_normalizing_constant(
                    lambda x: self(x).exp().sum(1),
                    device=self.device,
                    dimensions=self.toy_dataset_dim,
                    dtype=torch.float32,
                )
            )

            logits = torch.cat(logits, 0)
            log_px = logits.logsumexp(1) - log_Z
            self.log("log_likelihood", log_px.mean())

        super().test_epoch_end()

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)
        return [optim], [scheduler]

    def get_gt_preds(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        gt, preds = [], []
        for x, y in tqdm(loader):
            x = x.to(self.device)
            y_hat = self(x).cpu()
            gt.append(y)
            preds.append(y_hat)
        return torch.cat(gt), torch.cat(preds)

    def ood_detect(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        scores = []
        for x, y in tqdm(loader):
            x = x.to(self.device)
            score = self.model(x).cpu()
            scores.append(score)

        uncert = {}
        uncert["p(x)"] = torch.cat(scores).cpu().numpy()
        return uncert
