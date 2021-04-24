import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from uncertainty_est.models.ebm.mcmc import MCMC


def init_random(buffer_size, data_shape, dim):
    buffer = torch.FloatTensor(buffer_size, *data_shape).random_(0, dim)
    buffer = F.one_hot(buffer.long(), num_classes=dim).float()
    return buffer


class DiscreteMCMC(MCMC):
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
        reinit_freq,
        num_cat,
        sgld_steps=20,
        entropy_reg_weight=0.0,
        warmup_steps=0,
        lr_step_size=50,
        **kwargs
    ):
        self.num_cat = num_cat
        super().__init__(
            arch_name=arch_name,
            arch_config=arch_config,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            buffer_size=buffer_size,
            n_classes=n_classes,
            data_shape=data_shape,
            smoothing=smoothing,
            pyxce=pyxce,
            pxsgld=pxsgld,
            pxysgld=pxysgld,
            class_cond_p_x_sample=class_cond_p_x_sample,
            sgld_batch_size=sgld_batch_size,
            sgld_lr=0.0,
            sgld_std=0.0,
            reinit_freq=reinit_freq,
            sgld_steps=sgld_steps,
            entropy_reg_weight=entropy_reg_weight,
            warmup_steps=warmup_steps,
            lr_step_size=lr_step_size,
        )
        self.save_hyperparameters()

    def _init_buffer(self):
        self.replay_buffer = init_random(
            self.buffer_size, self.data_shape, self.num_cat
        ).cpu()

    def sample_p_0(self, replay_buffer, bs, y=None):
        if len(replay_buffer) == 0:
            return init_random(bs, self.data_shape, self.num_cat), []

        buffer_size = (
            len(replay_buffer) if y is None else len(replay_buffer) // self.n_classes
        )
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds
        if y is not None:
            inds = y.cpu() * buffer_size + inds

        buffer_samples = replay_buffer[inds].to(self.device)
        if self.reinit_freq > 0.0:
            random_samples = init_random(bs, self.data_shape, self.num_cat).to(
                self.device
            )
            choose_random = (torch.rand(bs) < self.reinit_freq).to(buffer_samples)[
                (...,) + (None,) * (len(self.data_shape) + 1)
            ]
            samples = (
                choose_random * random_samples + (1 - choose_random) * buffer_samples
            )
        else:
            samples = buffer_samples
        return samples.to(self.device), inds

    def sample_q(self, replay_buffer, y=None, n_steps=20, contrast=False):
        self.model.eval()
        bs = self.sgld_batch_size if y is None else y.size(0)

        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = self.sample_p_0(replay_buffer, bs=bs, y=y)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True)

        # Gradient with Gibbs "http://arxiv.org/abs/2102.04509"
        for _ in range(n_steps):
            energy = self.model(x_k, y=y)
            f_prime = torch.autograd.grad(energy.sum(), [x_k], retain_graph=True)[0]

            d = f_prime - (x_k * f_prime).sum(-1, keepdim=True)
            q_i_given_x = Categorical(logits=(d / 2.0).flatten(start_dim=1))
            i = q_i_given_x.sample()
            prob_i_given_x = q_i_given_x.log_prob(i).exp()

            # Flip sampled dimension
            x_q_idx = x_k.argmax(-1)
            x_q_idx.flatten(start_dim=1)[torch.arange(len(x_k)), i // self.num_cat] = (
                i % self.num_cat
            )
            x_q = F.one_hot(x_q_idx, self.num_cat).float()
            x_q.requires_grad_()

            energy_q = self.model(x_q, y=y)
            f_prime = torch.autograd.grad(energy_q.sum(), [x_q], retain_graph=True)[0]
            d = f_prime - (x_q * f_prime).sum(-1, keepdim=True)
            q_i_given_x = Categorical(logits=(d / 2.0).flatten(start_dim=1))
            prob_i_given_x_q = q_i_given_x.log_prob(i).exp()

            # Update samples dependig on Metropolis-Hastings Probability
            keep_prob = torch.exp(energy_q - energy) * (
                prob_i_given_x_q / prob_i_given_x
            )
            keep_prob[keep_prob > 1.0] = 1.0
            keep = torch.rand(len(x_k)).to(self.device) < keep_prob

            x_k = x_k.detach()
            x_k[keep] = x_q[keep]

        self.model.train()
        final_samples = x_k.detach()

        # update replay buffer
        if len(replay_buffer) > 0:
            replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples
