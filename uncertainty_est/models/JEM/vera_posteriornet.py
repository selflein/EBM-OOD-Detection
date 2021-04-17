import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.distributions import Dirichlet

from uncertainty_est.models.JEM.vera import VERA
from uncertainty_est.models.priornet.dpn_losses import dirichlet_kl_divergence
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class VERAPosteriorNet(VERA):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        n_classes,
        gen_learning_rate,
        ebm_iters,
        generator_iters,
        entropy_weight,
        generator_type,
        generator_arch_name,
        generator_arch_config,
        generator_config,
        min_sigma,
        max_sigma,
        p_control,
        n_control,
        pg_control,
        clf_ent_weight,
        ebm_type,
        clf_weight,
        warmup_steps,
        no_g_batch_norm,
        batch_size,
        lr_decay,
        lr_decay_epochs,
        vis_every=-1,
        alpha_fix=True,
        entropy_reg=0.0,
        is_toy_dataset=False,
        toy_dataset_dim=2,
        alpha_0_control=0.0,
        **kwargs,
    ):
        if n_control is None:
            n_control = p_control

        super().__init__(
            arch_name,
            arch_config,
            learning_rate,
            beta1,
            beta2,
            weight_decay,
            n_classes,
            gen_learning_rate,
            ebm_iters,
            generator_iters,
            entropy_weight,
            generator_type,
            generator_arch_name,
            generator_arch_config,
            generator_config,
            min_sigma,
            max_sigma,
            p_control,
            n_control,
            pg_control,
            clf_ent_weight,
            ebm_type,
            clf_weight,
            warmup_steps,
            no_g_batch_norm,
            batch_size,
            lr_decay,
            lr_decay_epochs,
            is_toy_dataset,
            toy_dataset_dim,
            vis_every,
            **kwargs,
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()

    def classifier_loss(self, ld_logits, y_l):
        alpha = torch.exp(ld_logits)  # / self.p_y.unsqueeze(0).to(self.device)
        # Multiply by class counts for Bayesian update

        soft_output = F.one_hot(y_l, self.n_classes)
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.n_classes)
        entropy_reg = Dirichlet(alpha).entropy()
        UCE_loss = torch.sum(
            soft_output * (torch.digamma(alpha_0) - torch.digamma(alpha))
        ) - self.clf_ent_weight * torch.sum(
            entropy_reg
        )  # alpha = self.class_counts.unsqueeze(0).to(self.device) * alpha

        return UCE_loss

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)
        alphas = torch.exp(outputs[0]).reshape(-1) + 1 if self.alpha_fix else 0
        self.logger.experiment.add_histogram("alphas", alphas, self.current_epoch)

    def ood_detect(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        logits = []
        for x, _ in tqdm(loader):
            x = x.to(self.device)
            logits.append(self.model.classify(x).cpu())
        logits = torch.cat(logits)
        scores = logits.exp().sum(1)

        uncert = {}
        # exp(-E(x)) ~ p(x)
        uncert["p(x)-epistemic_uncert"] = scores.numpy()
        uncert["log p(x)"] = scores.log().numpy()
        dirichlet_uncerts = dirichlet_prior_network_uncertainty(
            logits.numpy(), alpha_correction=self.alpha_fix
        )
        uncert = {**uncert, **dirichlet_uncerts}
        return uncert
