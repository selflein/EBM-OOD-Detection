""" from https://github.com/johannbrehmer/manifold-flow """

import torch
import numpy as np
from torch import nn

from uncertainty_est.utils.utils import sum_except_batch, split_leading_dim


def product(x):
    try:
        prod = 1
        for factor in x:
            prod *= factor
        return prod
    except:
        return x


class StandardNormal:
    """A multivariate Normal with zero mean and unit covariance."""

    def __init__(self, shape):
        super().__init__()
        self._shape = torch.Size(shape)
        self._log_z = 0.5 * np.prod(shape) * np.log(2 * np.pi)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        neg_energy = -0.5 * sum_except_batch(inputs ** 2, num_batch_dims=1)
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = torch.randn(context_size * num_samples, *self._shape)
            return split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)


class RescaledNormal:
    """A multivariate Normal with zero mean and a diagonal covariance that is epsilon^2 along each diagonal entry of the matrix."""

    def __init__(self, shape, std=1.0, clip=10.0):
        super().__init__()
        self._shape = torch.Size(shape)
        self.std = std
        self._clip = clip
        self._log_z = 0.5 * np.prod(shape) * np.log(2 * np.pi) + np.prod(
            shape
        ) * np.log(self.std)

    def _log_prob(self, inputs, context):
        # Note: the context is ignored.
        if inputs.shape[1:] != self._shape:
            raise ValueError(
                "Expected input of shape {}, got {}".format(
                    self._shape, inputs.shape[1:]
                )
            )
        if self._clip is not None:
            inputs = torch.clamp(inputs, -self._clip, self._clip)
        neg_energy = (
            -0.5 * sum_except_batch(inputs ** 2, num_batch_dims=1) / self.std ** 2
        )
        return neg_energy - self._log_z

    def _sample(self, num_samples, context):
        if context is None:
            return self.std * torch.randn(num_samples, *self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            context_size = context.shape[0]
            samples = self.std * torch.randn(context_size * num_samples, *self._shape)
            return split_leading_dim(samples, [context_size, num_samples])

    def _mean(self, context):
        if context is None:
            return torch.zeros(self._shape)
        else:
            # The value of the context is ignored, only its size is taken into account.
            return torch.zeros(context.shape[0], *self._shape)


class IdentityTransform(nn.Module):
    """Transform that leaves input unchanged."""

    def forward(self, inputs, context=None, full_jacobian=False):
        batch_size = inputs.shape[0]
        if full_jacobian:
            jacobian = torch.eye(inputs.shape[1:]).unsqueeze(0)
            return inputs, jacobian
        else:
            logabsdet = torch.zeros(batch_size)
            return inputs, logabsdet

    def inverse(self, inputs, context=None, full_jacobian=False):
        return self(inputs, context, full_jacobian)


class ProjectionSplit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_dim_total = product(input_dim)
        self.output_dim_total = product(output_dim)
        self.mode_in = "vector" if isinstance(input_dim, int) else "image"
        self.mode_out = "vector" if isinstance(input_dim, int) else "image"

        assert (
            self.input_dim_total >= self.output_dim_total
        ), "Input dimension has to be larger than output dimension"

    def forward(self, inputs, **kwargs):
        if self.mode_in == "vector" and self.mode_out == "vector":
            u = inputs[:, : self.output_dim]
            rest = inputs[:, self.output_dim :]
        elif self.mode_in == "image" and self.mode_out == "vector":
            h = inputs.view(inputs.size(0), -1)
            u = h[:, : self.output_dim]
            rest = h[:, self.output_dim :]
        else:
            raise NotImplementedError(
                "Unsuppoorted projection modes {}, {}".format(
                    self.mode_in, self.mode_out
                )
            )
        return u, rest

    def inverse(self, inputs, **kwargs):
        orthogonal_inputs = kwargs.get(
            "orthogonal_inputs",
            torch.zeros(inputs.size(0), self.input_dim_total - self.output_dim),
        )
        if self.mode_in == "vector" and self.mode_out == "vector":
            x = torch.cat((inputs, orthogonal_inputs), dim=1)
        elif self.mode_in == "image" and self.mode_out == "vector":
            c, h, w = self.input_dim
            x = torch.cat((inputs, orthogonal_inputs), dim=1)
            x = x.view(inputs.size(0), c, h, w)
        else:
            raise NotImplementedError(
                "Unsuppoorted projection modes {}, {}".format(
                    self.mode_in, self.mode_out
                )
            )
        return x


class ManifoldFlow(nn.Module):
    """ Manifold-based flow (base class for FOM, M-flow, PIE) """

    def __init__(
        self,
        data_dim,
        latent_dim,
        outer_transform,
        inner_transform=None,
        pie_epsilon=1.0e-2,
        apply_context_to_outer=True,
        clip_pie=False,
    ):
        super(ManifoldFlow, self).__init__()

        self.data_dim = data_dim
        self.latent_dim = latent_dim
        self.apply_context_to_outer = apply_context_to_outer
        self.total_data_dim = product(data_dim)
        self.total_latent_dim = product(latent_dim)

        assert self.total_latent_dim < self.total_data_dim

        self.manifold_latent_distribution = StandardNormal((self.total_latent_dim,))
        self.orthogonal_latent_distribution = RescaledNormal(
            (self.total_data_dim - self.total_latent_dim,),
            std=pie_epsilon,
            clip=None if not clip_pie else clip_pie * pie_epsilon,
        )
        self.projection = ProjectionSplit(self.total_data_dim, self.total_latent_dim)

        self.outer_transform = outer_transform
        if inner_transform is None:
            self.inner_transform = IdentityTransform()
        else:
            self.inner_transform = inner_transform

    def forward(self, x, mode="mf", context=None, return_hidden=False):
        """
        Transforms data point to latent space, evaluates likelihood, and transforms it back to data space.

        mode can be "mf" (calculating the exact manifold density based on the full Jacobian), "pie" (calculating the density in x), "slice"
        (calculating the density on x, but projected onto the manifold), or "projection" (calculating no density at all).
        """

        assert mode in [
            "mf",
            "pie",
            "slice",
            "projection",
            "pie-inv",
            "mf-fixed-manifold",
        ]

        if mode == "mf" and not x.requires_grad:
            x.requires_grad = True

        # Encode
        u, h_manifold, h_orthogonal, log_det_outer, log_det_inner = self._encode(
            x, context
        )

        # Decode
        (
            x_reco,
            inv_log_det_inner,
            inv_log_det_outer,
            inv_jacobian_outer,
            h_manifold_reco,
        ) = self._decode(u, mode=mode, context=context)

        # Log prob
        log_prob = self._log_prob(
            mode,
            u,
            h_orthogonal,
            log_det_inner,
            log_det_outer,
            inv_log_det_inner,
            inv_log_det_outer,
            inv_jacobian_outer,
        )

        if return_hidden:
            return x_reco, log_prob, u, torch.cat((h_manifold, h_orthogonal), -1)
        return x_reco, log_prob, u

    def encode(self, x, context=None):
        """ Transforms data point to latent space. """

        u, _, _, _, _ = self._encode(x, context=context)
        return u

    def decode(self, u, u_orthogonal=None, context=None):
        """ Decodes latent variable to data space."""

        x, _, _, _, _ = self._decode(
            u, mode="projection", u_orthogonal=u_orthogonal, context=context
        )
        return x

    def log_prob(self, x, mode="mf", context=None):
        """ Evaluates log likelihood for given data point."""

        return self.forward(x, mode, context)[1]

    def sample(self, u=None, n=1, context=None, sample_orthogonal=False):
        """
        Generates samples from model.

        Note: this is PIE / MF sampling! Cannot sample from slice of PIE efficiently.
        """

        if u is None:
            u = self.manifold_latent_distribution.sample(n, context=None)
        u_orthogonal = (
            self.orthogonal_latent_distribution.sample(n, context=None)
            if sample_orthogonal
            else None
        )
        x = self.decode(u, u_orthogonal=u_orthogonal, context=context)
        return x

    def project(self, x, context=None):
        return self.decode(self.encode(x, context), context)

    def _encode(self, x, context=None):
        # Encode
        h, log_det_outer = self.outer_transform(
            x,
            full_jacobian=False,
            context=context if self.apply_context_to_outer else None,
        )
        h_manifold, h_orthogonal = self.projection(h)
        u, log_det_inner = self.inner_transform(
            h_manifold, full_jacobian=False, context=context
        )

        return u, h_manifold, h_orthogonal, log_det_outer, log_det_inner

    def _decode(self, u, mode, u_orthogonal=None, context=None):
        if mode == "mf" and not u.requires_grad:
            u.requires_grad = True

        h, inv_log_det_inner = self.inner_transform.inverse(
            u, full_jacobian=False, context=context
        )

        if u_orthogonal is not None:
            h = self.projection.inverse(h, orthogonal_inputs=u_orthogonal)
        else:
            h = self.projection.inverse(h)

        if mode in ["pie", "slice", "projection", "mf-fixed-manifold"]:
            x, inv_log_det_outer = self.outer_transform.inverse(
                h,
                full_jacobian=False,
                context=context if self.apply_context_to_outer else None,
            )
            inv_jacobian_outer = None
        else:
            x, inv_jacobian_outer = self.outer_transform.inverse(
                h,
                full_jacobian=True,
                context=context if self.apply_context_to_outer else None,
            )
            inv_log_det_outer = None

        return x, inv_log_det_inner, inv_log_det_outer, inv_jacobian_outer, h

    def _log_prob(
        self,
        mode,
        u,
        h_orthogonal,
        log_det_inner,
        log_det_outer,
        inv_log_det_inner,
        inv_log_det_outer,
        inv_jacobian_outer,
    ):
        if mode == "pie":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(
                h_orthogonal, context=None
            )
            log_prob = log_prob + log_det_outer + log_det_inner

        elif mode == "pie-inv":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(
                h_orthogonal, context=None
            )
            log_prob = log_prob - inv_log_det_outer - inv_log_det_inner

        elif mode == "slice":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(
                torch.zeros_like(h_orthogonal), context=None
            )
            log_prob = log_prob - inv_log_det_outer - inv_log_det_inner

        elif mode == "mf":
            # inv_jacobian_outer is dx / du, but still need to restrict this to the manifold latents
            inv_jacobian_outer = inv_jacobian_outer[:, :, : self.latent_dim]
            # And finally calculate log det (J^T J)
            jtj = torch.bmm(
                torch.transpose(inv_jacobian_outer, -2, -1), inv_jacobian_outer
            )

            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob - 0.5 * torch.slogdet(jtj)[1] - inv_log_det_inner

        elif mode == "mf-fixed-manifold":
            log_prob = self.manifold_latent_distribution._log_prob(u, context=None)
            log_prob = log_prob + self.orthogonal_latent_distribution._log_prob(
                h_orthogonal, context=None
            )
            log_prob = log_prob + log_det_outer + log_det_inner

        else:
            log_prob = None

        return log_prob
