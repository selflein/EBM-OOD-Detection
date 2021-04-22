import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from uncertainty_est.utils.utils import to_np
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ebm.utils.model import HDGE
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.models.ebm.utils.utils import (
    KHotCrossEntropyLoss,
    smooth_one_hot,
)


class HDGEModel(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        pyxce,
        pxcontrast,
        pxycontrast,
        smoothing,
        n_classes,
        contrast_k,
        contrast_t,
        warmup_steps=-1,
    ):
        super().__init__()
        self.__dict__.update(locals())
        self.save_hyperparameters()

        arch = get_arch(arch_name, arch_config)
        self.model = HDGE(arch, n_classes, contrast_k, contrast_t)

    def forward(self, x):
        return self.model(x)

    def compute_losses(self, x_lab, dist, logits=None, evaluation=False):
        l_pyxce, l_pxcontrast, l_pxycontrast = 0.0, 0.0, 0.0
        # log p(y|x) cross entropy loss
        if self.pyxce > 0:
            if logits is None:
                logits = self.model.classify(x_lab)
            l_pyxce = KHotCrossEntropyLoss()(logits, dist)
            l_pyxce *= self.pyxce

        # log p(x) using contrastive learning
        if self.pxcontrast > 0:
            # ones like dist to use all indexes
            ones_dist = torch.ones_like(dist).to(self.device)
            output, target, _, _ = self.model.joint(
                img=x_lab, dist=ones_dist, evaluation=evaluation
            )
            l_pxcontrast = F.cross_entropy(output, target)
            l_pxcontrast *= self.pxycontrast

        # log p(x|y) using contrastive learning
        if self.pxycontrast > 0:
            output, target, _, _ = self.model.joint(
                img=x_lab, dist=dist, evaluation=evaluation
            )
            l_pxycontrast = F.cross_entropy(output, target)
            l_pxycontrast *= self.pxycontrast

        return l_pyxce, l_pxcontrast, l_pxycontrast

    def training_step(self, batch, batch_idx):
        x_lab, y_lab = batch
        dist = smooth_one_hot(y_lab, self.n_classes, self.smoothing)

        loss = sum(self.compute_losses(x_lab, dist))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.classify(x)
        dist = smooth_one_hot(y, self.n_classes, self.smoothing)

        self.log(
            "val_loss",
            sum(self.compute_losses(x, dist, logits=logits, evaluation=True)),
        )

        acc = (y == logits.argmax(1)).float().mean(0).item()
        self.log("val_acc", acc)

    def validation_epoch_end(self, training_step_outputs):
        if self.vis_every > 0 and self.current_epoch % self.vis_every == 0:
            interp = torch.linspace(-10, 10, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1)

            log_px = to_np(self.model(data.to(self.device)))

            fig, ax = plt.subplots()
            ax.set_title(f"log p(x)")
            mesh = ax.pcolormesh(to_np(x), to_np(y), log_px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("log p(x)", fig, self.current_epoch)
            plt.close()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

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

    def classify(self, x):
        return torch.softmax(self.model.classify(x), -1)

    def get_ood_scores(self, x):
        return {"p(x)": self.model(x)}
