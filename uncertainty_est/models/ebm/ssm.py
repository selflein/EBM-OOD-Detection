import torch
from torch import autograd
import torch.nn.functional as F

from uncertainty_est.utils.utils import to_np
from uncertainty_est.models.ebm.utils.model import JEM
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel


class SSM(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        n_classes,
        clf_weight,
        noise_type="radermacher",
        n_particles=1,
        warmup_steps=2500,
        lr_step_size=50,
        is_toy_dataset=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        arch = get_arch(arch_name, arch_config)
        self.model = JEM(arch)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        (x_lab, y_lab), (x_p_d, _) = batch
        dup_samples = (
            x_p_d.unsqueeze(0)
            .expand(self.n_particles, *x_p_d.shape)
            .contiguous()
            .view(-1, *x_p_d.shape[1:])
        )
        dup_samples.requires_grad_(True)

        vectors = torch.randn_like(dup_samples)
        if self.noise_type == "radermacher":
            vectors = vectors.sign()
        elif self.noise_type == "gaussian":
            pass
        else:
            raise ValueError("Noise type not implemented")

        logp = self.model(dup_samples).sum()

        grad1 = autograd.grad(logp, dup_samples, create_graph=True)[0]
        loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.0
        gradv = torch.sum(grad1 * vectors)

        grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]
        loss2 = torch.sum(vectors * grad2, dim=-1)

        loss1 = loss1.view(self.n_particles, -1).mean(dim=0)
        loss2 = loss2.view(self.n_particles, -1).mean(dim=0)

        ssm_loss = (loss1 + loss2).mean()
        self.log("train/ssm_loss", ssm_loss)

        clf_loss = 0.0
        if self.clf_weight > 0.0:
            _, logits = self.model(x_lab, return_logits=True)
            clf_loss = self.clf_weight * F.cross_entropy(logits, y_lab)
            self.log("train/clf_loss", clf_loss)
        return ssm_loss + clf_loss

    def validation_step(self, batch, batch_idx):
        (x_lab, y_lab), (_, _) = batch

        if self.n_classes < 2:
            return

        _, logits = self.model(x_lab, return_logits=True)
        acc = (y_lab == logits.argmax(1)).float().mean(0).item()
        self.log("val/acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.classify(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test.acc", acc)

        return y_hat

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optim, step_size=self.lr_step_size, gamma=0.5
        )
        return [optim], [scheduler]

    def classify(self, x):
        return torch.softmax(self.model.classify(x), -1)

    def get_ood_scores(self, x):
        return {"p(x)": self.model(x)}
