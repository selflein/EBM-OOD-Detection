import random

import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

from uncertainty_est.utils.utils import to_np
from uncertainty_est.models.ebm.utils.model import JEM
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.models.ebm.utils.utils import (
    KHotCrossEntropyLoss,
    smooth_one_hot,
)


def init_random(buffer_size, data_shape):
    return torch.FloatTensor(buffer_size, *data_shape).uniform_(-1, 1)


class MCMC(OODDetectionModel):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        buffer_size,
        n_classes,
        data_shape,
        smoothing,
        pyxce,
        pxsgld,
        pxysgld,
        class_cond_p_x_sample,
        sgld_batch_size,
        sgld_lr,
        sgld_std,
        reinit_freq,
        annealed_sgld=True,
        entropy_term=0.3,
        approx_ent_samples=100,
        kl_term=1.0,
        sgld_steps=20,
        entropy_reg_weight=0.0,
        warmup_steps=2500,
        lr_step_size=50,
        is_toy_dataset=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if len(data_shape) == 3:
            data_shape = [data_shape[-1], data_shape[0], data_shape[1]]

        self.__dict__.update(locals())
        self.save_hyperparameters()

        if class_cond_p_x_sample:
            assert n_classes > 1

        arch = get_arch(arch_name, arch_config)
        self.model = JEM(arch)

        self.buffer_size = self.buffer_size - (self.buffer_size % self.n_classes)
        self._init_buffer()

    def forward(self, x):
        return self.model(x)[:, None]

    def _init_buffer(self):
        self.replay_buffer = init_random(self.buffer_size, self.data_shape).cpu()

    def training_step(self, batch, batch_idx):
        (x_lab, y_lab), (x_p_d, _) = batch
        if self.n_classes > 1:
            dist = smooth_one_hot(y_lab, self.n_classes, self.smoothing)
        else:
            dist = y_lab[None, :]

        l_pyxce, l_pxsgld, l_pxysgld = 0.0, 0.0, 0.0
        # log p(y|x) cross entropy loss
        if self.pyxce > 0:
            logits = self.model.classify(x_lab)
            l_pyxce = KHotCrossEntropyLoss()(logits, dist)
            l_pyxce *= self.pyxce

            l_pyxce += (
                self.entropy_reg_weight
                * -torch.distributions.Categorical(logits=logits).entropy().mean()
            )

        # log p(x) using sgld
        if self.pxsgld > 0:
            if self.class_cond_p_x_sample:
                y_q = torch.randint(0, self.n_classes, (self.sgld_batch_size,)).to(
                    self.device
                )
                x_q, idxs = self.sample_q(self.replay_buffer, y=y_q, return_idxs=True)
            else:
                x_q, idxs = self.sample_q(
                    self.replay_buffer, n_steps=self.sgld_steps, return_idxs=True
                )  # sample from log-sumexp

            fp = self.model(x_p_d)
            fq = self.model(x_q.detach())
            l_pxsgld = -(fp.mean() - fq.mean()) + (fp ** 2).mean() + (fq ** 2).mean()

            if self.kl_term > 0.0:
                self.model.requires_grad_(False)
                kl_loss = self.model(x_q)
                self.model.requires_grad_(True)
                l_pxsgld -= self.kl_term * kl_loss.mean()

                # Approx. entropy term by comparing with samples from buffer
                if self.entropy_term > 0.0:
                    buffer_idxs = random.sample(
                        [i for i in range(len(self.replay_buffer)) if i not in idxs],
                        self.approx_ent_samples,
                    )
                    buffer_samples = (
                        self.replay_buffer[buffer_idxs]
                        .flatten(start_dim=1)
                        .to(self.device)
                    )

                    data_flat = x_q.flatten(start_dim=1)
                    dist_matrix = torch.norm(
                        data_flat[:, None, :] - buffer_samples[None, :, :], p=2, dim=-1
                    )
                    loss_repel = torch.log(dist_matrix.min(dim=1)[0]).mean()
                    l_pxsgld += self.entropy_term * loss_repel

            l_pxsgld *= self.pxsgld

        # log p(x|y) using sgld
        if self.pxysgld > 0:
            x_q_lab = self.sample_q(self.replay_buffer, y=y_lab)
            fp, fq = self.model(x_lab).mean(), self.model(x_q_lab).mean()
            l_pxysgld = -(fp - fq)
            l_pxysgld *= self.pxysgld

        loss = l_pxysgld + l_pxsgld + l_pyxce
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        (x_lab, y_lab), (_, _) = batch

        if self.n_classes < 2:
            return

        _, logits = self.model(x_lab, return_logits=True)
        acc = (y_lab == logits.argmax(1)).float().mean(0).item()
        self.log("val/acc", acc)

    def validation_epoch_end(self, outputs):
        if self.is_toy_dataset:
            interp = torch.linspace(-4, 4, 500)
            x, y = torch.meshgrid(interp, interp)
            data = torch.stack((x.reshape(-1), y.reshape(-1)), 1).to(self.device)
            p_xy = torch.exp(self(data))
            px = to_np(p_xy.sum(1))

            fig, ax = plt.subplots()
            mesh = ax.pcolormesh(x, y, px.reshape(*x.shape))
            fig.colorbar(mesh)
            self.logger.experiment.add_figure("dist/p(x)", fig, self.current_epoch)
            plt.close()

        super().validation_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.classify(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test/acc", acc)

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

    def sample_p_0(self, replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(bs, self.data_shape), []

        buffer_size = (
            len(replay_buffer) if y is None else len(replay_buffer) // self.n_classes
        )
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds

        buffer_samples = replay_buffer[inds].to(self.device)
        random_samples = init_random(bs, self.data_shape).to(self.device)
        choose_random = (torch.rand(bs) < self.reinit_freq).to(buffer_samples)[
            (...,) + (None,) * len(self.data_shape)
        ]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(self.device), inds

    def sample_q(
        self, replay_buffer, y=None, n_steps=20, contrast=False, return_idxs=False
    ):
        self.model.eval()
        bs = self.sgld_batch_size if y is None else y.size(0)

        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = self.sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True)

        # sgld
        for step in range(n_steps):
            noise = self.sgld_std * torch.randn_like(x_k)
            if self.annealed_sgld:
                noise *= (n_steps - step - 1) / n_steps
            x_k = x_k + noise

            if not contrast:
                energy = self.model(x_k, y=y).sum(0)
            else:
                if y is not None:
                    dist = smooth_one_hot(y, self.n_classes, self.smoothing)
                else:
                    dist = torch.ones((bs, self.n_classes)).to(self.device)
                output, target, _, _ = self.model.joint(
                    img=x_k, dist=dist, evaluation=True
                )
                energy = -1.0 * F.cross_entropy(output, target)
            f_prime = torch.autograd.grad(
                energy, [x_k], retain_graph=True, create_graph=True
            )[0]

            # Propagte gradients through final step of MCMC
            # Following "https://arxiv.org/abs/2012.01316"
            if step == n_steps - 1:
                x_k = self.sgld_lr * f_prime
            else:
                x_k.data += self.sgld_lr * f_prime

        self.model.train()

        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = x_k.detach().cpu()

        if return_idxs:
            return x_k, buffer_inds

        return x_k
