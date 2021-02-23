""" from https://github.com/sharpenb/Posterior-Network/blob/main/src/posterior_networks/NormalizingFlowDensity.py """
import math

import torch
from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms.affine_autoregressive import (
    AffineAutoregressive,
    affine_autoregressive,
)
from pyro.distributions import TransformModule
from torch import nn
import torch.distributions as tdist


class OrthogonalTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.transform = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        return self.transform(x)

    @staticmethod
    def log_abs_det_jacobian(z, z_next):
        return 0.0

    def compute_weight_penalty(self):
        sq_weight = torch.mm(self.transform.weight.T, self.transform.weight)
        identity = torch.eye(self.dim).to(sq_weight)
        penalty = torch.linalg.norm(identity - sq_weight, ord="fro")

        return penalty


class ReparameterizedTransform(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.bias = nn.Parameter(torch.zeros(dim))

        self.u_mat = nn.Parameter(torch.Tensor(dim, dim))
        self.v_mat = nn.Parameter(torch.Tensor(dim, dim))
        self.sigma = nn.Parameter(torch.ones(dim))
        nn.init.kaiming_uniform_(self.u_mat, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_mat, a=math.sqrt(5))

    def forward(self, x):
        weight = self.u_mat @ torch.diag_embed(self.sigma) @ self.v_mat.T
        return torch.nn.functional.linear(x, weight, self.bias)

    def log_abs_det_jacobian(self, z, z_next):
        return torch.sum(self.sigma)

    def compute_weight_penalty(self):
        return self._weight_penalty(self.u_mat) + self._weight_penalty(self.v_mat)

    @staticmethod
    def _weight_penalty(weight):
        sq_weight = torch.mm(weight.T, weight)
        identity = torch.eye(weight.size(0)).to(sq_weight)
        penalty = torch.linalg.norm(identity - sq_weight, ord="fro")
        return penalty


class NormalizingFlowDensity(nn.Module):
    def __init__(self, dim, flow_length, flow_type="planar_flow"):
        super(NormalizingFlowDensity, self).__init__()
        self.dim = dim
        self.flow_length = flow_length
        self.flow_type = flow_type

        self.mean = nn.Parameter(torch.zeros(self.dim), requires_grad=False)
        self.cov = nn.Parameter(torch.eye(self.dim), requires_grad=False)

        if self.flow_type == "radial_flow":
            self.transforms = nn.ModuleList([Radial(dim) for _ in range(flow_length)])
        elif self.flow_type == "iaf_flow":
            self.transforms = nn.ModuleList(
                [
                    affine_autoregressive(dim, hidden_dims=[128, 128])
                    for _ in range(flow_length)
                ]
            )
        elif self.flow_type == "planar_flow":
            self.transforms = nn.ModuleList([Planar(dim) for _ in range(flow_length)])
        elif self.flow_type == "orthogonal_flow":
            self.transforms = nn.ModuleList(
                [OrthogonalTransform(dim) for _ in range(flow_length)]
            )
        elif self.flow_type == "reparameterized_flow":
            self.transforms = nn.ModuleList(
                [ReparameterizedTransform(dim) for _ in range(flow_length)]
            )
        else:
            raise NotImplementedError

    def forward(self, z):
        sum_log_jacobians = 0
        for transform in self.transforms:
            z_next = transform(z)
            sum_log_jacobians = sum_log_jacobians + transform.log_abs_det_jacobian(
                z, z_next
            )
            z = z_next

        return z, sum_log_jacobians

    def log_prob(self, x):
        z, sum_log_jacobians = self.forward(x)
        log_prob_z = tdist.MultivariateNormal(self.mean, self.cov).log_prob(z)
        log_prob_x = log_prob_z + sum_log_jacobians  # [batch_size]
        return log_prob_x
