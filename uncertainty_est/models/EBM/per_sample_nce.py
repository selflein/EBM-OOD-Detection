from collections import defaultdict

import torch
from tqdm import tqdm
import pytorch_lightning as pl
import torch.nn.functional as F
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
        test_ood_dataloaders=[],
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        self.model = get_arch(arch_name, arch_config)

    def setup(self, phase):
        if phase != "fit":
            return
        self.len_ds = torch.tensor(
            len(self.train_dataloader.dataset), requires_grad=False
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch

        noise_dist = distributions.Normal(0, 1)
        noise = noise_dist.sample(x.size())
        log_p_noise = torch.cat(
            (
                noise_dist.log_prob(torch.zeros(len(x))) - torch.log(self.len_ds),
                noise_dist.log_prob(noise) - torch.log(self.len_ds),
            )
        )

        x_noisy = x + noise
        log_p_model = self.model(torch.cat((x, x_noisy)))

        posterior_prob = log_p_model - torch.log(log_p_noise.exp() + log_p_model.exp())

        labels = torch.cat((torch.ones(x.size(1)), torch.zeros(x_noisy.size(1))))
        loss = F.binary_cross_entropy_with_logits(posterior_prob, labels)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)

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
        uncert["log p(x)"] = torch.cat(scores).cpu().numpy()
        uncert["p(x)"] = torch.cat(scores).exp().cpu().numpy()
        return uncert
