import torch
from tqdm import tqdm

from uncertainty_est.models.EBM.nce import NoiseContrastiveEstimation
from uncertainty_est.models.priornet.dpn_losses import UnfixedDirichletKLLoss
from uncertainty_est.models.priornet.uncertainties import (
    dirichlet_prior_network_uncertainty,
)


class NCEPriorNet(NoiseContrastiveEstimation):
    """Implementation of Noise Contrastive Estimation http://proceedings.mlr.press/v9/gutmann10a.html"""

    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        noise_distribution="uniform",
        noise_distribution_kwargs={"low": 0, "high": 1},
        is_toy_dataset=False,
        toy_dataset_dim=2,
        test_ood_dataloaders=[],
        clf_loss_weight=1.0,
        concentration=1.0,
        target_concentration=None,
        entropy_reg=1e-4,
        reverse_kl=True,
        alpha_fix=True,
    ):
        super().__init__(
            arch_name,
            arch_config,
            learning_rate,
            momentum,
            weight_decay,
            noise_distribution,
            noise_distribution_kwargs,
            is_toy_dataset,
            toy_dataset_dim,
            test_ood_dataloaders,
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()
        self.clf_loss = UnfixedDirichletKLLoss(
            concentration, target_concentration, entropy_reg, reverse_kl, alpha_fix
        )

    def training_step(self, batch, batch_idx):
        _, y = batch
        torch.autograd.set_detect_anomaly(True)

        ebm_loss, logits = self.compute_ebm_loss(batch, return_outputs=True)
        classifier_loss = self.clf_loss(logits, y)
        loss = ebm_loss + self.clf_loss_weight * classifier_loss

        self.log("train/loss", loss)
        self.log("train/ebm_loss", ebm_loss)
        return loss

    def validation_step(self, batch, batch_idx):
        return

    def test_step(self, batch, batch_idx):
        self.to(torch.float32)
        x, y = batch
        y_hat = self.model(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)
        return y_hat

    def ood_detect(self, loader):
        self.eval()
        torch.set_grad_enabled(False)
        self.model.to(torch.float64)
        logits = []
        for x, _ in tqdm(loader):
            x = x.to(self.device).double()
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
