""" from https://github.com/KaosEngineer/PriorNetworks """

from typing import Optional, Iterable

import numpy as np
import torch
import torch.nn.functional as F


class MixedLoss:
    def __init__(self, losses, mixing_params: Optional[Iterable[float]]):
        assert isinstance(losses, (list, tuple))
        assert isinstance(mixing_params, (list, tuple, np.ndarray))
        assert len(losses) == len(mixing_params)

        self.losses = losses
        if mixing_params is not None:
            self.mixing_params = mixing_params
        else:
            self.mixing_params = [1.0] * len(self.losses)

    def __call__(self, logits_list, labels_list):
        return self.forward(logits_list, labels_list)

    def forward(self, logits_list, labels_list):
        total_loss = []
        for i, loss in enumerate(self.losses):
            weighted_loss = loss(logits_list[i], labels_list[i]) * self.mixing_params[i]
            total_loss.append(weighted_loss)
        total_loss = torch.stack(total_loss, dim=0)
        # Normalize by target concentration, so that loss  magnitude is constant wrt lr and other losses
        return torch.sum(total_loss)


class PriorNetMixedLoss:
    def __init__(self, losses, mixing_params: Optional[Iterable[float]]):
        assert isinstance(losses, (list, tuple))
        assert isinstance(mixing_params, (list, tuple, np.ndarray))
        assert len(losses) == len(mixing_params)

        self.losses = losses
        if mixing_params is not None:
            self.mixing_params = mixing_params
        else:
            self.mixing_params = [1.0] * len(self.losses)

    def __call__(self, logits_list, labels_list):
        return self.forward(logits_list, labels_list)

    def forward(self, logits_list, labels_list):
        total_loss = []
        target_concentration = 0.0
        for i, loss in enumerate(self.losses):
            if loss.target_concentration > target_concentration:
                target_concentration = loss.target_concentration
            weighted_loss = loss(logits_list[i], labels_list[i]) * self.mixing_params[i]
            total_loss.append(weighted_loss / target_concentration)
        total_loss = torch.stack(total_loss, dim=0)
        # Normalize by target concentration, so that loss  magnitude is constant wrt lr and other losses
        return torch.sum(total_loss)


class DirichletKLLoss:
    """
    Can be applied to any model which returns logits

    """

    def __init__(
        self, target_concentration=1e3, concentration=1.0, reverse=True, alpha_fix=False
    ):
        """
        :param target_concentration: The concentration parameter for the
        target class (if provided)
        :param concentration: The 'base' concentration parameters for
        non-target classes.
        :param alpha_fix: If set to true make the minimum output alpha for the dirichlet 1
        fixing the issue of degenerate parameterization, i.e., concentration parameters < 1.
        """
        self.target_concentration = torch.tensor(
            target_concentration, dtype=torch.float32
        )
        self.concentration = concentration
        self.reverse = reverse
        self.alpha_fix = alpha_fix

    def __call__(self, logits, labels: Optional[torch.tensor] = None, reduction="mean"):
        """
        :param alphas: The alpha parameter outputs from the model
        :param labels: Optional. The target labels indicating the correct
        class.

        The loss creates a set of target alpha (concentration) parameters
        with all values set to self.concentration, except for the correct
        class (if provided), which is set to self.target_concentration
        :return: an array of per example loss
        """
        alphas = torch.exp(logits)
        if self.alpha_fix:
            alphas = alphas + 1

        target_alphas = torch.ones_like(alphas) * self.concentration
        if labels is not None:
            target_alphas += torch.zeros_like(alphas).scatter_(
                1, labels[:, None], self.target_concentration.item()
            )

        if self.reverse:
            loss = dirichlet_reverse_kl_divergence(
                alphas=alphas, target_alphas=target_alphas
            )
        else:
            loss = dirichlet_kl_divergence(alphas=alphas, target_alphas=target_alphas)

        if reduction == "mean":
            return torch.mean(loss)
        elif reduction == "none":
            return loss
        else:
            raise NotImplementedError


class UnfixedDirichletKLLoss:
    def __init__(
        self,
        concentration,
        target_concentration=None,
        entropy_reg=1e-4,
        reverse_kl=True,
        alpha_fix=True,
    ):
        self.entropy_reg = entropy_reg
        self.reverse_kl = reverse_kl
        self.alpha_fix = alpha_fix
        self.concentration = concentration
        self.target_concentration = target_concentration

    def __call__(self, logits, label=None, reduction="mean"):
        alphas = torch.exp(logits)

        target_alphas = torch.empty_like(alphas, requires_grad=False).fill_(
            self.concentration
        )
        if label is not None:
            if self.target_concentration is None:
                target_concentration = torch.sum(alphas, 1)
            else:
                target_concentration = self.target_concentration

            target_alphas[torch.arange(len(label)), label] = target_concentration

        if self.alpha_fix:
            output_alphas = alphas + 1
        else:
            output_alphas = alphas

        if self.reverse_kl:
            loss = dirichlet_kl_divergence(target_alphas, output_alphas)
        else:
            loss = dirichlet_kl_divergence(output_alphas, target_alphas)

        loss += (
            self.entropy_reg * -torch.distributions.Dirichlet(output_alphas).entropy()
        )
        if reduction == "mean":
            return loss.mean()
        return loss


def dirichlet_kl_divergence(
    alphas, target_alphas, precision=None, target_precision=None, epsilon=1e-8
):
    """
    This function computes the Forward KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentation parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentation parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
    :param epsilon: Smoothing factor for numercal stability. Default value is 1e-8
    :return: Tensor for Batchsize X 1 of forward KL divergences between target Dirichlet and model
    """
    if not precision:
        precision = torch.sum(alphas, dim=1, keepdim=True)
    if not target_precision:
        target_precision = torch.sum(target_alphas, dim=1, keepdim=True)

    precision_term = torch.lgamma(target_precision) - torch.lgamma(precision)
    assert torch.all(torch.isfinite(precision_term)).item()
    alphas_term = torch.sum(
        torch.lgamma(alphas + epsilon)
        - torch.lgamma(target_alphas + epsilon)
        + (target_alphas - alphas)
        * (
            torch.digamma(target_alphas + epsilon)
            - torch.digamma(target_precision + epsilon)
        ),
        dim=1,
        keepdim=True,
    )
    assert torch.all(torch.isfinite(alphas_term)).item()

    cost = torch.squeeze(precision_term + alphas_term)
    return cost


def dirichlet_reverse_kl_divergence(
    alphas, target_alphas, precision=None, target_precision=None, epsilon=1e-8
):
    """
    This function computes the Reverse KL divergence between a model Dirichlet distribution
    and a target Dirichlet distribution based on the concentration (alpha) parameters of each.

    :param alphas: Tensor containing concentation parameters of model. Expected shape is batchsize X num_classes.
    :param target_alphas: Tensor containing target concentation parameters. Expected shape is batchsize X num_classes.
    :param precision: Optional argument. Can pass in precision of model. Expected shape is batchsize X 1
    :param target_precision: Optional argument. Can pass in target precision. Expected shape is batchsize X 1
    :param epsilon: Smoothing factor for numercal stability. Default value is 1e-8
    :return: Tensor for Batchsize X 1 of reverse KL divergences between target Dirichlet and model
    """
    return dirichlet_kl_divergence(
        alphas=target_alphas,
        target_alphas=alphas,
        precision=target_precision,
        target_precision=precision,
        epsilon=epsilon,
    )
