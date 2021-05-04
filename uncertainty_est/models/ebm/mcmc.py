import torch
from matplotlib import pyplot as plt

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
        sgld_steps=20,
        entropy_reg_weight=0.0,
        warmup_steps=2500,
        lr_step_size=50,
        is_toy_dataset=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        self.save_hyperparameters()

        if len(data_shape) == 3:
            self.sample_shape = [data_shape[-1], data_shape[0], data_shape[1]]
        else:
            self.sample_shape = data_shape

        if class_cond_p_x_sample:
            assert n_classes > 1

        arch = get_arch(arch_name, arch_config)
        self.model = JEM(arch)

        self.buffer_size = self.buffer_size - (self.buffer_size % self.n_classes)
        self._init_buffer()

    def forward(self, x):
        return self.model(x)

    def _init_buffer(self):
        self.replay_buffer = init_random(self.buffer_size, self.sample_shape).cpu()

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
            self.log("train/clf_loss", l_pyxce)

        # log p(x) using sgld
        if self.pxsgld > 0:
            if self.class_cond_p_x_sample:
                y_q = torch.randint(0, self.n_classes, (self.sgld_batch_size,)).to(
                    self.device
                )
                x_q = self.sample_q(self.replay_buffer, y=y_q)
            else:
                x_q = self.sample_q(
                    self.replay_buffer, n_steps=self.sgld_steps
                )  # sample from log-sumexp

            fp = self.model(x_p_d)
            fq = self.model(x_q)
            l_pxsgld = -(fp.mean() - fq.mean()) + (fp ** 2).mean() + (fq ** 2).mean()
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
            px = to_np(torch.exp(self(data)))

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
            return init_random(bs, self.sample_shape), []

        buffer_size = (
            len(replay_buffer) if y is None else len(replay_buffer) // self.n_classes
        )
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds

        buffer_samples = replay_buffer[inds].to(self.device)
        random_samples = init_random(bs, self.sample_shape).to(self.device)
        choose_random = (torch.rand(bs) < self.reinit_freq).to(buffer_samples)[
            (...,) + (None,) * len(self.sample_shape)
        ]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(self.device), inds

    def sample_q(self, replay_buffer, y=None, n_steps=20, contrast=False):
        self.model.eval()
        bs = self.sgld_batch_size if y is None else y.size(0)

        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = self.sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True)

        # sgld
        for _ in range(n_steps):
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
            f_prime = torch.autograd.grad(energy, [x_k], retain_graph=True)[0]
            x_k.data += self.sgld_lr * f_prime + self.sgld_std * torch.randn_like(x_k)
        self.model.train()
        final_samples = x_k.detach()

        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
