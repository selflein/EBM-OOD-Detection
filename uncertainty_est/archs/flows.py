""" from https://github.com/sharpenb/Posterior-Network/blob/main/src/posterior_networks/NormalizingFlowDensity.py """
import math

import torch
from pyro.distributions.transforms.planar import Planar
from pyro.distributions.transforms.radial import Radial
from pyro.distributions.transforms import ELUTransform, LeakyReLUTransform
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
        return torch.zeros(z.shape[0], device=z.device)

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
        lin_out = torch.nn.functional.linear(x, weight, self.bias)
        return lin_out

    def log_abs_det_jacobian(self, z, z_next):
        ladj = torch.sum(torch.log(self.sigma.abs()))
        ladj = torch.empty(z.size(0), device=z.device).fill_(ladj)
        return ladj

    def compute_weight_penalty(self):
        return self._weight_penalty(self.u_mat) + self._weight_penalty(self.v_mat)

    @staticmethod
    def _weight_penalty(weight):
        sq_weight = torch.mm(weight.T, weight)
        identity = torch.eye(weight.size(0)).to(sq_weight)
        penalty = torch.linalg.norm(identity - sq_weight, ord="fro")
        return penalty


@torch.jit.script
def sequential_mult(V, X):
    for row in range(V.shape[0] - 1, -1, -1):
        X = X - 2 * V[row : row + 1, :].t() @ (V[row : row + 1, :] @ X)
    return X


@torch.jit.script
def sequential_inv_mult(V, X):
    for row in range(V.shape[0]):
        X = X - 2 * V[row : row + 1, :].t() @ (V[row : row + 1, :] @ X)
    return X


class Orthogonal(nn.Module):
    def __init__(self, d, m=28, strategy="sequential"):
        super(Orthogonal, self).__init__()
        self.d = d
        self.strategy = strategy
        self.U = torch.nn.Parameter(torch.zeros((d, d)).normal_(0, 0.05))

        if strategy == "fast":
            assert d % m == 0, (
                "The CUDA implementation assumes m=%i divides d=%i which, for current parameters, is not true.  "
                % (d, m)
            )
            HouseProd.m = m

        if not strategy in ["fast", "sequential"]:
            raise NotImplementedError(
                "The only implemented strategies are 'fast' and 'sequential'. "
            )

    def forward(self, X):
        if self.strategy == "fast":
            X = HouseProd.apply(X, self.U)
        elif self.strategy == "sequential":
            X = sequential_mult(self.U, X.t()).t()
        else:
            raise NotImplementedError(
                "The only implemented strategies are 'fast' and 'sequential'. "
            )
        return X

    def inverse(self, X):
        if self.strategy == "fast":
            X = HouseProd.apply(X, torch.flip(self.U, dims=[0]))
        elif self.strategy == "sequential":
            X = sequential_inv_mult(self.U, X.t()).t()
        else:
            raise NotImplementedError(
                "The only implemented strategies are 'fast' and 'sequential'. "
            )
        return X

    def lgdet(self, X):
        return 0


class MatExpOrthogonal(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.param = nn.Parameter(torch.randn(d, d))
        self.cached = False
        self._cache = None

    @property
    def w(self):
        if self.cached:
            return self._cache

        # Compute skew symetric matrix
        aux = self.param.triu(diagonal=1)
        aux = aux - aux.t()

        w = torch.matrix_exp(aux)
        self.cached = True
        self._cache = w
        return w

    def forward(self, x):
        return x @ self.w

    def backward(*args, **kwargs):
        self.cached = False
        return super().backward(*args, **kwargs)


class LinearSVD(torch.nn.Module):
    def __init__(self, d, orthogonal_param="householder", **kwargs):
        super(LinearSVD, self).__init__()
        self.d = d

        if orthogonal_param == "mat_exp":
            self.U = MatExpOrthogonal(d, **kwargs)
            self.V = MatExpOrthogonal(d, **kwargs)
        elif orthogonal_param == "householder":
            self.U = Orthogonal(d, **kwargs)
            self.V = Orthogonal(d, **kwargs)
        else:
            raise NotImplementedError(
                f"Orthogonal parameterization {orthogonal_param} not implemented."
            )
        self.D = nn.Parameter(torch.empty(d).uniform_(0.99, 1.01))
        self.bias = nn.Parameter(torch.zeros(d))

    def forward(self, X):
        X = self.U(X)
        X = X * self.D
        X = self.V(X)
        return X + self.bias

    def log_abs_det_jacobian(self, z, z_next):
        ladj = torch.log(torch.prod(self.D).abs())
        return torch.empty(z.size(0), device=z.device).fill_(ladj)


class NonLinearity(nn.Module):
    def __init__(self, type_="elu"):
        super().__init__()
        nonlins = {
            "elu": ELUTransform,
            "leaky_relu": LeakyReLUTransform,
        }
        self.transform = nonlins[type_]()

    def forward(self, x):
        return self.transform(x)

    def log_abs_det_jacobian(self, z, z_next):
        return self.transform.log_abs_det_jacobian(z, z_next).sum(1)

    def compute_weight_penalty(self, *args, **kwargs):
        return 0.0


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
            self.transforms = nn.ModuleList()
            for i in range(flow_length):
                self.transforms.append(ReparameterizedTransform(dim))
                if i != (flow_length - 1):
                    self.transforms.append(NonLinearity("elu"))
        elif self.flow_type == "svd":
            self.transforms = nn.ModuleList()
            for i in range(flow_length):
                self.transforms.append(LinearSVD(dim))
                if i != (flow_length - 1):
                    self.transforms.append(NonLinearity("elu"))
        elif self.flow_type == "svd_mat_exp":
            self.transforms = nn.ModuleList()
            for i in range(flow_length):
                self.transforms.append(LinearSVD(dim, orthogonal_param="mat_exp"))
                if i != (flow_length - 1):
                    self.transforms.append(NonLinearity("elu"))
        else:
            raise NotImplementedError
        print(self)

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
