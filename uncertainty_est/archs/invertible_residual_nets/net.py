import torch
from torch import nn
from torch import distributions as tdist

from uncertainty_est.archs.invertible_residual_nets.act_norm import ActNorm
from uncertainty_est.archs.invertible_residual_nets.bound_spectral_norm import (
    spectral_norm_fc,
)
from uncertainty_est.archs.invertible_residual_nets.log_det import (
    compute_log_det,
    power_series_matrix_logarithm_trace,
)


class IResNetBlock(nn.Module):
    def __init__(
        self,
        inp_dim,
        hidden_dims,
        coeff,
        n_trace_samples=1,
        n_series_terms=1,
        act_norm=True,
        exact=False,
    ):
        super().__init__()
        self.n_trace_samples = n_trace_samples
        self.n_series_terms = n_series_terms
        self.exact = exact

        dims = (
            [
                inp_dim,
            ]
            + hidden_dims
            + [
                inp_dim,
            ]
        )

        layers = []
        for i, o in zip(dims[:-1], dims[1:]):
            layers.append(spectral_norm_fc(nn.Linear(i, o), coeff))
            layers.append(nn.ELU())
        self.block = nn.Sequential(*layers)

        self.act_norm = ActNorm(inp_dim) if act_norm else None

    def forward(self, x):
        if self.act_norm is not None:
            x, an_logdet = self.act_norm(x)

        out = self.block(x)
        if self.exact:
            log_det, _ = compute_log_det(x, out)
        else:
            log_det = power_series_matrix_logarithm_trace(
                out, x, self.n_series_terms, self.n_trace_samples
            )

        # Apply residual to output
        y = x + out
        return y, an_logdet + log_det

    def inverse(self, y, max_iters):
        raise NotImplementedError


class IResNet(nn.Module):
    def __init__(
        self,
        n_blocks,
        dim,
        block_h_dims,
        coeff,
        n_trace_samples,
        n_series_terms,
        act_norm=True,
        exact=False,
    ):
        super().__init__()
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)

        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            block = IResNetBlock(
                dim,
                block_h_dims,
                coeff,
                n_trace_samples,
                n_series_terms,
                act_norm,
                exact,
            )
            self.blocks.append(block)

    def forward(self, x):
        sum_log_det = 0.0
        for block in self.blocks:
            x, log_det = block(x)
            sum_log_det += log_det

        return x, sum_log_det

    def log_prob(self, x):
        z, log_det = self(x)
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        return log_prob_z + log_det

    def inverse(self, y, max_iters=100):
        raise NotImplementedError
