import torch
from tqdm import tqdm
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
        entropy_reg=0.0,
        is_toy_dataset=False,
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
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()

    def setup(self, stage):
        train_loader = self.train_dataloader()

        class_counts = torch.zeros(self.n_classes)
        for (_, y), (_, _) in train_loader.dataset:
            class_counts[y] += 1

        self.class_counts = class_counts
        self.p_y = class_counts / len(train_loader.dataset)

    def classifier_loss(self, ld_logits, y_l):
        p_x_given_y = torch.exp(ld_logits) / self.p_y.unsqueeze(0).to(self.device)
        alpha = self.class_counts.unsqueeze(0).to(self.device) * p_x_given_y

        if self.alpha_fix:
            alpha = alpha + 1

        alpha_0 = alpha.sum(1)
        UCE_loss = self.clf_weight * torch.mean(
            torch.digamma(alpha_0) - torch.digamma(alpha[torch.arange(len(y_l)), y_l])
        )
        entropy_reg = self.entropy_reg * -Dirichlet(alpha).entropy().mean()
        UCE_loss += entropy_reg
        self.log("train/uce_loss", UCE_loss)
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
