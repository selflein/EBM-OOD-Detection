import torch
from tqdm import tqdm

from uncertainty_est.models.JEM.vera import VERA
from uncertainty_est.models.priornet.dpn_losses import UnfixedDirichletKLLoss
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

        self.clf_loss = UnfixedDirichletKLLoss(
            concentration, target_concentration, entropy_reg, reverse_kl, alpha_fix
        )

    def classifier_loss(self, ld_logits, y_l):
        loss = self.clf_loss(ld_logits, y_l)
        self.log("train/clf_loss", loss)
        return loss

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
        scores = logits.logsumexp(1)

        uncert = {}
        # exp(-E(x)) ~ p(x)
        uncert["p(x)"] = scores.numpy()
        dirichlet_uncerts = dirichlet_prior_network_uncertainty(
            logits.numpy(), alpha_correction=self.alpha_fix
        )
        uncert = {**uncert, **dirichlet_uncerts}
        return uncert
