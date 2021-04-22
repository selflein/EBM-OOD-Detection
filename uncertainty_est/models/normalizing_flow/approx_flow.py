from uncertainty_est.models.normalizing_flow.norm_flow import NormalizingFlow


class ApproxNormalizingFlow(NormalizingFlow):
    def __init__(
        self,
        density_type,
        latent_dim,
        n_density,
        learning_rate,
        momentum,
        weight_decay,
        weight_penalty_weight=1.0,
    ):
        super().__init__(
            density_type,
            latent_dim,
            n_density,
            learning_rate,
            momentum,
            weight_decay,
        )
        assert density_type in ("orthogonal_flow", "reparameterized_flow")
        self.__dict__.update(locals())
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, _ = batch
        log_p = self.density_estimation.log_prob(x)

        loss = -log_p.mean()
        self.log("train/ml_loss", loss, on_epoch=True)

        weight_penalty = 0.0
        transforms = self.density_estimation.transforms
        for t in transforms:
            weight_penalty += t.compute_weight_penalty()
        weight_penalty /= len(transforms)
        self.log("train/weight_penalty", weight_penalty, on_epoch=True)
        loss += self.weight_penalty_weight * weight_penalty

        return loss
