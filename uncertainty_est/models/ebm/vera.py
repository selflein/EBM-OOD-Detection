import torch
from torch import distributions

from uncertainty_est.models.ebm.utils.model import JEM
from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.models.ood_detection_model import OODDetectionModel
from uncertainty_est.models.ebm.utils.vera_utils import (
    VERADiscreteGenerator,
    VERAGenerator,
    VERAHMCGenerator,
    set_bn_to_eval,
    set_bn_to_train,
)


class VERA(OODDetectionModel):
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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.__dict__.update(locals())
        self.save_hyperparameters()
        self.automatic_optimization = False

        arch = get_arch(arch_name, arch_config)
        self.model = JEM(arch)

        g = get_arch(generator_arch_name, generator_arch_config)
        if generator_type == "verahmc":
            self.generator = VERAHMCGenerator(g, **generator_config)
        elif generator_type == "vera":
            self.generator = VERAGenerator(g, **generator_config)
        elif generator_type == "vera_discrete":
            self.generator = VERADiscreteGenerator(g, **generator_config)
        else:
            raise NotImplementedError(f"Generator '{generator_type}' not implemented!")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        opt_e, opt_g = self.optimizers()
        (x_l, y_l), (x_d, _) = batch

        x_l.requires_grad_()
        x_d.requires_grad_()

        # sample from q(x, h)
        x_g, h_g = self.generator.sample(x_l.size(0), requires_grad=True)

        # ebm (contrastive divergence) objective
        if batch_idx % self.ebm_iters == 0:
            ebm_loss = self.ebm_step(x_d, x_l, x_g, y_l)

            self.log("train/ebm_loss", ebm_loss, prog_bar=True)

            opt_e.zero_grad()
            self.manual_backward(ebm_loss, opt_e)
            opt_e.step()

        # gen obj
        if batch_idx % self.generator_iters == 0:
            gen_loss = self.generator_step(x_g, h_g)

            self.log("train/gen_loss", gen_loss, prog_bar=True)

            opt_g.zero_grad()
            self.manual_backward(gen_loss, opt_g)
            opt_g.step()

        # clamp sigma to (.01, max_sigma) for generators
        if self.generator_type in ["verahmc", "vera"]:
            self.generator.clamp_sigma(self.max_sigma, sigma_min=self.min_sigma)

    def ebm_step(self, x_d, x_l, x_g, y_l):
        x_g_detach = x_g.detach().requires_grad_()

        if self.no_g_batch_norm:
            self.model.apply(set_bn_to_eval)
            lg_detach, lg_logits = self.model(x_g_detach, return_logits=True)
            self.model.apply(set_bn_to_train)
        else:
            lg_detach, lg_logits = self.model(x_g_detach, return_logits=True)

        unsup_ent = torch.tensor(0.0)
        if self.ebm_type == "ssl":
            ld, unsup_logits = self.model(x_d, return_logits=True)
            _, ld_logits = self.model(x_l, return_logits=True)
            unsup_ent = distributions.Categorical(logits=unsup_logits).entropy()
        elif self.ebm_type == "jem":
            ld, ld_logits = self.model(x_l, return_logits=True)
            self.log("train/acc", (ld_logits.argmax(1) == y_l).float().mean(0))
        elif self.ebm_type == "p_x":
            ld, ld_logits = self.model(x_l).squeeze(), torch.tensor(0.0).to(self.device)
        else:
            raise NotImplementedError(f"EBM type '{self.ebm_type}' not implemented!")

        logp_obj = ld.mean() - lg_detach.mean()
        e_loss = (
            -logp_obj
            + self.p_control * (ld ** 2).mean()
            + self.n_control * (lg_detach ** 2).mean()
            + self.clf_ent_weight * unsup_ent.mean()
        )

        if self.pg_control > 0:
            grad_ld = (
                torch.autograd.grad(ld.mean(), x_l, create_graph=True)[0]
                .flatten(start_dim=1)
                .norm(2, 1)
            )
            e_loss += self.pg_control * (grad_ld ** 2.0 / 2.0).mean()

        self.log("train/e_loss", e_loss.item())

        if self.clf_weight > 0:
            clf_loss = self.clf_weight * self.classifier_loss(ld_logits, y_l, lg_logits)
            self.log("train/clf_loss", clf_loss)
            e_loss += clf_loss

        return e_loss

    def classifier_loss(self, ld_logits, y_l, lg_logits):
        return torch.nn.CrossEntropyLoss()(ld_logits, y_l)

    def generator_step(self, x_g, h_g):
        lg = self.model(x_g).squeeze()
        grad = torch.autograd.grad(lg.sum(), x_g, retain_graph=True)[0]
        ebm_gn = grad.norm(2, 1).mean()

        if self.entropy_weight != 0.0:
            entropy_obj, ent_gn = self.generator.entropy_obj(x_g, h_g)

        logq_obj = lg.mean() + self.entropy_weight * entropy_obj
        return -logq_obj

    def validation_step(self, batch, batch_idx):
        (x_l, y_l), _ = batch
        ld, ld_logits = self.model(x_l, return_logits=True)

        self.log("val/loss", -ld.mean())

        # Performing density estimation only
        if ld_logits.shape[1] < 2:
            return

        acc = (y_l == ld_logits.argmax(1)).float().mean(0)
        self.log("val/acc", acc)
        return ld_logits

    def test_step(self, batch, batch_idx):
        x, y = batch
        _, y_hat = self.model(x, return_logits=True)

        if self.n_classes < 2:
            return

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("acc", acc)

        return y_hat

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            betas=(self.beta1, self.beta2),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        gen_optim = torch.optim.AdamW(
            self.generator.parameters(),
            betas=(self.beta1, self.beta2),
            lr=self.gen_learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim, gamma=self.lr_decay, milestones=self.lr_decay_epochs
        )
        gen_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            gen_optim, gamma=self.lr_decay, milestones=self.lr_decay_epochs
        )
        return [optim, gen_optim], [scheduler, gen_scheduler]

    def classify(self, x):
        return self.model.classify(x).softmax(-1)

    def get_ood_scores(self, x):
        return {"p(x)": self.model(x)}
