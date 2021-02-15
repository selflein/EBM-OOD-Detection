import torch
from tqdm import tqdm

from uncertainty_est.models.JEM.vera import VERA
from uncertainty_est.models.priornet.dpn_losses import dirichlet_kl_divergence
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class VERAPriorNet(VERA):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        beta1,
        beta2,
        weight_decay,
        n_classes,
        uncond,
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
        concentration=1.0,
        target_concentration=None,
        entropy_reg=0.0,
        reverse_kl=True,
        is_toy_dataset=False,
        toy_dataset_dim=2,
        **kwargs,
    ):
        super().__init__(
            arch_name,
            arch_config,
            learning_rate,
            beta1,
            beta2,
            weight_decay,
            n_classes,
            uncond,
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
            vis_every,
            **kwargs,
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()

    def classifier_loss(self, ld_logits, y_l):
        p_xy = torch.exp(ld_logits)
        p_x = torch.sum(p_xy, 1)

        # Update prior with evidence
        alphas = self.concentration + p_xy

        if torch.isnan(alphas).any() or not torch.isfinite(alphas).any():
            raise ValueError()

        if self.target_concentration is None:
            target_concentration = p_x + self.concentration
        else:
            target_concentration = (
                torch.empty(len(alphas))
                .fill_(self.target_concentration)
                .to(self.device)
            )

        target_alphas = torch.empty_like(alphas).fill_(self.concentration)
        target_alphas[torch.arange(len(y_l)), y_l] = target_concentration

        if self.reverse_kl:
            kl_term = dirichlet_kl_divergence(target_alphas, alphas)
        else:
            kl_term = dirichlet_kl_divergence(alphas, target_alphas)

        clf_loss = self.clf_weight * kl_term.mean()
        clf_loss += (
            self.entropy_reg * -torch.distributions.Dirichlet(alphas).entropy().mean()
        )
        self.log("train/clf_loss", clf_loss)
        return clf_loss

    def validation_epoch_end(self, outputs):
        super().validation_epoch_end(outputs)
        alphas = torch.exp(outputs[0]).reshape(-1) + self.concentration
        self.logger.experiment.add_histogram("alphas", alphas, self.current_epoch)

    def ood_detect(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        logits = []
        for x, _ in tqdm(loader):
            x = x.to(self.device)
            logits.append(self(x).cpu())
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
