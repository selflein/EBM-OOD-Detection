"""
From https://github.com/wgrathwohl/VERA
"""

import torch
import torch.nn as nn
import torch.distributions as distributions
import torch.nn.functional as F
import numpy as np


def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.eval()


def set_bn_to_train(m):
    classname = m.__class__.__name__
    if classname.find("BatchNorm") != -1:
        m.train()


def _gen_post_helper(netG, x_tilde, eps, sigma):
    eps = eps.clone().detach().requires_grad_(True)
    with torch.no_grad():
        G_eps = netG(eps)
    bsz = eps.size(0)
    log_prob_eps = (eps ** 2).view(bsz, -1).sum(1).view(-1, 1)
    log_prob_x = (x_tilde - G_eps) ** 2 / sigma ** 2
    log_prob_x = log_prob_x.view(bsz, -1)
    log_prob_x = torch.sum(log_prob_x, dim=1).view(-1, 1)
    logjoint_vect = -0.5 * (log_prob_eps + log_prob_x)
    logjoint_vect = logjoint_vect.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = eps.grad
    return logjoint_vect, logjoint, grad_logjoint


def get_gen_posterior_samples(
    netG,
    x_tilde,
    eps_init,
    sigma,
    burn_in,
    num_samples_posterior,
    leapfrog_steps,
    stepsize,
    flag_adapt,
    hmc_learning_rate,
    hmc_opt_accept,
):
    device = eps_init.device
    bsz, eps_dim = eps_init.size(0), eps_init.size(1)
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz * num_samples_posterior, eps_dim).to(device)
    current_eps = eps_init
    cnt = 0
    for i in range(n_steps):
        eps = current_eps
        p = torch.randn_like(current_eps)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _gen_post_helper(
            netG, x_tilde, current_eps, sigma
        )
        current_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            eps = eps + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _gen_post_helper(
                    netG, x_tilde, eps, sigma
                )
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _gen_post_helper(
            netG, x_tilde, eps, sigma
        )
        proposed_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p ** 2).sum(dim=1)
        current_K = current_K.view(-1, 1)  # should be size of B x 1
        proposed_K = 0.5 * (p ** 2).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1)  # should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K))
        accept = accept.float().squeeze()  # should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try:
            len(ind) > 0
            current_eps[ind, :] = eps[ind, :]
            current_U[ind] = proposed_U[ind]
        except:
            print("Samples were all rejected...skipping")
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = (
                stepsize
                + hmc_learning_rate
                * (accept.float().mean() - hmc_opt_accept)
                * stepsize
            )
        else:
            if eps_dim == 1:
                samples[cnt * bsz : (cnt + 1) * bsz, :] = current_eps
            else:
                samples[cnt * bsz : (cnt + 1) * bsz, :] = current_eps.squeeze()
            cnt += 1

        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize


def _ebm_helper(netEBM, x):
    x = x.clone().detach().requires_grad_(True)
    E_x = netEBM(x)
    logjoint_vect = E_x.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint = x.grad
    return logjoint_vect, logjoint, grad_logjoint


def get_ebm_samples(
    netEBM,
    x_init,
    burn_in,
    num_samples_posterior,
    leapfrog_steps,
    stepsize,
    flag_adapt,
    hmc_learning_rate,
    hmc_opt_accept,
):
    device = x_init.device
    bsz, x_size = x_init.size(0), x_init.size()[1:]
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)
    samples = torch.zeros(bsz * num_samples_posterior, *x_size).to(device)
    current_x = x_init
    cnt = 0
    for i in range(n_steps):
        x = current_x
        p = torch.randn_like(current_x)
        current_p = p
        logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, current_x)
        current_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        for j in range(leapfrog_steps):
            x = x + stepsize * p
            if j < leapfrog_steps - 1:
                logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, x)
                proposed_U = -logjoint_vect
                grad_U = -grad_logjoint
                p = p - stepsize * grad_U
        logjoint_vect, logjoint, grad_logjoint = _ebm_helper(netEBM, x)
        proposed_U = -logjoint_vect.view(-1, 1)
        grad_U = -grad_logjoint
        p = p - stepsize * grad_U / 2.0
        p = -p
        current_K = 0.5 * (current_p ** 2).flatten(start_dim=1).sum(dim=1)
        current_K = current_K.view(-1, 1)  # should be size of B x 1
        proposed_K = 0.5 * (p ** 2).flatten(start_dim=1).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1)  # should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K))
        accept = accept.float().squeeze()  # should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try:
            len(ind) > 0
            current_x[ind, :] = x[ind, :]
            current_U[ind] = proposed_U[ind]
        except:
            print("Samples were all rejected...skipping")
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = (
                stepsize
                + hmc_learning_rate
                * (accept.float().mean() - hmc_opt_accept)
                * stepsize
            )
        else:
            samples[cnt * bsz : (cnt + 1) * bsz, :] = current_x
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples, acceptRate, stepsize


def _ebm_latent_helper(netEBM, netG, z, eps, sigma):
    z = z.clone().detach().requires_grad_(True)
    eps = eps.clone().detach().requires_grad_(True)
    x = netG(z) + eps * sigma
    E_x = netEBM(x)
    logjoint_vect = E_x.squeeze()
    logjoint = torch.sum(logjoint_vect)
    logjoint.backward()
    grad_logjoint_z = z.grad
    grad_logjoint_eps = eps.grad
    return logjoint_vect, logjoint, (grad_logjoint_z, grad_logjoint_eps)


def get_ebm_latent_samples(
    netEBM,
    netG,
    z_init,
    eps_init,
    sigma,
    burn_in,
    num_samples_posterior,
    leapfrog_steps,
    stepsize,
    flag_adapt,
    hmc_learning_rate,
    hmc_opt_accept,
):
    device = z_init.device
    bsz, z_size, eps_size = z_init.size(0), z_init.size()[1:], eps_init.size()[1:]
    n_steps = burn_in + num_samples_posterior
    acceptHist = torch.zeros(bsz, n_steps).to(device)
    logjointHist = torch.zeros(bsz, n_steps).to(device)

    samples_z = torch.zeros(bsz * num_samples_posterior, *z_size).to(device)
    samples_eps = torch.zeros(bsz * num_samples_posterior, *eps_size).to(device)

    current_z = z_init
    current_eps = eps_init
    cnt = 0
    for i in range(n_steps):
        z = current_z
        eps = current_eps

        p_z = torch.randn_like(current_z)
        p_eps = torch.randn_like(current_eps)

        current_p_z = p_z
        current_p_eps = p_eps

        (
            logjoint_vect,
            logjoint,
            (grad_logjoint_z, grad_logjoint_eps),
        ) = _ebm_latent_helper(netEBM, netG, current_z, current_eps, sigma)
        current_U = -logjoint_vect.view(-1, 1)

        grad_U_z = -grad_logjoint_z
        grad_U_eps = -grad_logjoint_eps

        p_z = p_z - stepsize * grad_U_z / 2.0
        p_eps = p_eps - stepsize * grad_U_eps / 2.0

        for j in range(leapfrog_steps):

            z = z + stepsize * p_z
            eps = eps + stepsize * p_eps

            if j < leapfrog_steps - 1:
                (
                    logjoint_vect,
                    logjoint,
                    (grad_logjoint_z, grad_logjoint_eps),
                ) = _ebm_latent_helper(netEBM, netG, z, eps, sigma)
                proposed_U = -logjoint_vect

                grad_U_z = -grad_logjoint_z
                grad_U_eps = -grad_logjoint_eps

                p_z = p_z - stepsize * grad_U_z
                p_eps = p_eps - stepsize * grad_U_eps

        (
            logjoint_vect,
            logjoint,
            (grad_logjoint_z, grad_logjoint_eps),
        ) = _ebm_latent_helper(netEBM, netG, z, eps, sigma)
        proposed_U = -logjoint_vect.view(-1, 1)

        grad_U_z = -grad_logjoint_z
        grad_U_eps = -grad_logjoint_eps

        p_z = p_z - stepsize * grad_U_z / 2.0
        p_z = -p_z

        p_eps = p_eps - stepsize * grad_U_eps / 2.0
        p_eps = -p_eps

        current_K = 0.5 * (current_p_z ** 2).flatten(start_dim=1).sum(dim=1)
        current_K += 0.5 * (current_p_eps ** 2).flatten(start_dim=1).sum(dim=1)
        current_K = current_K.view(-1, 1)  # should be size of B x 1

        proposed_K = 0.5 * (p_z ** 2).flatten(start_dim=1).sum(dim=1)
        proposed_K += 0.5 * (p_eps ** 2).flatten(start_dim=1).sum(dim=1)
        proposed_K = proposed_K.view(-1, 1)  # should be size of B x 1
        unif = torch.rand(bsz).view(-1, 1).to(device)
        accept = unif.lt(torch.exp(current_U - proposed_U + current_K - proposed_K))
        accept = accept.float().squeeze()  # should be B x 1
        acceptHist[:, i] = accept
        ind = accept.nonzero().squeeze()
        try:
            len(ind) > 0
            current_z[ind, :] = z[ind, :]
            current_eps[ind, :] = eps[ind, :]
            current_U[ind] = proposed_U[ind]
        except:
            print("Samples were all rejected...skipping")
            continue
        if i < burn_in and flag_adapt == 1:
            stepsize = (
                stepsize
                + hmc_learning_rate
                * (accept.float().mean() - hmc_opt_accept)
                * stepsize
            )
        else:
            samples_z[cnt * bsz : (cnt + 1) * bsz, :] = current_z.squeeze()
            samples_eps[cnt * bsz : (cnt + 1) * bsz, :] = current_eps.squeeze()
            cnt += 1
        logjointHist[:, i] = -current_U.squeeze()
    acceptRate = acceptHist.mean(dim=1)
    return samples_z, samples_eps, acceptRate, stepsize


def sgld_sample(logp_fn, x_init, l=1.0, e=0.01, n_steps=100):
    x_k = torch.autograd.Variable(x_init, requires_grad=True)
    # sgld
    lrs = [l for _ in range(n_steps)]
    for this_lr in lrs:
        f_prime = torch.autograd.grad(logp_fn(x_k).sum(), [x_k], retain_graph=True)[0]
        x_k.data += this_lr * f_prime + torch.randn_like(x_k) * e
    final_samples = x_k.detach()
    return final_samples


def update_logp(u, u_mu, std):
    return distributions.Normal(u_mu, std).log_prob(u).flatten(start_dim=1).sum(1)


def MALA(vars, logp_fn, step_lr):
    """
    Metropolis-Adjusted Langevin Algorithm.
    """
    step_std = (2 * step_lr) ** 0.5
    logp_vars = logp_fn(*vars)
    grads = torch.autograd.grad(logp_vars.sum(), vars)
    updates_mu = [v + step_lr * g for v, g in zip(vars, grads)]
    updates = [u_mu + step_std * torch.randn_like(u_mu) for u_mu in updates_mu]
    logp_updates = logp_fn(*updates)
    reverse_grads = torch.autograd.grad(logp_updates.sum(), updates)
    reverse_updates_mu = [v + step_lr * g for v, g in zip(updates, reverse_grads)]

    logp_forward = sum(
        [update_logp(u, u_mu, step_std) for u, u_mu in zip(updates, updates_mu)]
    )
    logp_backward = sum(
        [update_logp(v, ru_mu, step_std) for v, ru_mu in zip(vars, reverse_updates_mu)]
    )
    logp_accept = logp_updates + logp_backward - logp_vars - logp_forward
    p_accept = logp_accept.exp()
    accept = (torch.rand_like(p_accept) < p_accept).float()

    next_vars = []
    for u_v, v in zip(updates, vars):
        if len(u_v.size()) == 4:
            next_vars.append(
                accept[:, None, None, None] * u_v
                + (1 - accept[:, None, None, None]) * v
            )
        else:
            next_vars.append(accept[:, None] * u_v + (1 - accept[:, None]) * v)
    return next_vars, accept.mean()


class VERAHMCGenerator(nn.Module):
    """
    VERA Generator with HMC estimator.
    """

    def __init__(self, g, noise_dim, mcmc_lr=0.02):
        super().__init__()
        self.g = g
        self.logsigma = nn.Parameter(
            (
                torch.ones(
                    1,
                )
                * 0.01
            ).log()
        )
        self.noise_dim = noise_dim
        self.stepsize = nn.Parameter(torch.tensor(1.0 / noise_dim), requires_grad=False)
        self.mcmc_lr = mcmc_lr
        self.ar = 0.0

    def sample(self, n, requires_grad=False, return_mu=False, return_both=False):
        """sample x, h ~ q(x, h)"""
        h = torch.randn((n, self.noise_dim)).to(next(self.parameters()).device)
        if requires_grad:
            h.requires_grad_()
        x_mu = self.g(h)
        x = x_mu + torch.randn_like(x_mu) * self.logsigma.exp()
        if return_both:
            return x_mu, x, h
        if return_mu:
            return x_mu, h
        else:
            return x, h

    def logq_joint(self, x, h, return_mu=False):
        """
        Join distribution of data and latent.
        """
        logph = distributions.Normal(0, 1).log_prob(h).sum(1)
        gmu = self.g(h)
        px_given_h = distributions.Normal(gmu, self.logsigma.exp())
        logpx_given_h = px_given_h.log_prob(x).flatten(start_dim=1).sum(1)
        if return_mu:
            return logpx_given_h + logph, gmu
        else:
            return logpx_given_h + logph

    def entropy_obj(
        self,
        x,
        h,
        burn_in=2,
        num_samples_posterior=2,
        return_score=False,
        return_accept=False,
    ):
        """
        Entropy estimator using HMC samples.
        """
        h_given_x, self.ar, self.stepsize.data = hmc.get_gen_posterior_samples(
            netG=self.g,  # function to do HMC on
            x_tilde=x.detach(),  # variable to condition on
            eps_init=h.clone(),  # initialized at stationarity
            sigma=self.logsigma.exp().detach(),
            burn_in=burn_in,
            num_samples_posterior=num_samples_posterior,
            leapfrog_steps=5,
            stepsize=self.stepsize,
            flag_adapt=1,
            hmc_learning_rate=self.mcmc_lr,
            hmc_opt_accept=0.67,
        )  # target acceptance rate, for tuning the LR

        mean_output_summed = torch.zeros_like(x)
        mean_output = self.g(h_given_x)
        for cnt in range(num_samples_posterior):
            mean_output_summed = (
                mean_output_summed
                + mean_output[cnt * x.size(0) : (cnt + 1) * x.size(0)]
            )
        mean_output_summed /= num_samples_posterior

        c = ((x - mean_output_summed) / self.logsigma.exp() ** 2).detach()
        mgn = c.norm(2, 1).mean()
        g_error_entropy = torch.mul(c, x).mean(0).sum()
        if return_score:
            return g_error_entropy, mgn, c
        elif return_accept:
            return g_error_entropy, mgn, acceptRate
        else:
            return g_error_entropy, mgn

    def clamp_sigma(self, sigma, sigma_min=0.01):
        """
        Sigma clamping used for entropy estimator.
        """
        self.logsigma.data.clamp_(np.log(sigma_min), np.log(sigma))


class VERAGenerator(VERAHMCGenerator):
    """
    VERA generator.
    """

    def __init__(self, g, noise_dim, post_lr=0.001, init_post_logsigma=0.1):
        super().__init__(g, noise_dim, post_lr)
        self.post_logsigma = nn.Parameter(
            (
                torch.ones(
                    noise_dim,
                )
                * init_post_logsigma
            ).log()
        )
        self.post_optimizer = torch.optim.Adam([self.post_logsigma], lr=post_lr)

    def entropy_obj(
        self, x, h, num_samples_posterior=20, return_score=False, learn_post_sigma=True
    ):
        """
        Entropy objective using variational approximation with importance sampling.
        """
        inf_dist = distributions.Normal(h, self.post_logsigma.detach().exp())
        h_given_x = inf_dist.sample((num_samples_posterior,))
        if len(x.size()) == 4:
            inf_logprob = inf_dist.log_prob(h_given_x).sum(2)
            xr = x[None].repeat(num_samples_posterior, 1, 1, 1, 1)
            xr = xr.view(
                x.size(0) * num_samples_posterior, x.size(1), x.size(2), x.size(3)
            )
            logq, mean_output = self.logq_joint(
                xr, h_given_x.view(-1, h.size(1)), return_mu=True
            )
            mean_output = mean_output.view(
                num_samples_posterior, x.size(0), x.size(1), x.size(2), x.size(3)
            )
            logq = logq.view(num_samples_posterior, x.size(0))
            w = (logq - inf_logprob).softmax(dim=0)
            fvals = (x[None] - mean_output) / (self.logsigma.exp() ** 2)
            weighted_fvals = (fvals * w[:, :, None, None, None]).sum(0).detach()
            c = weighted_fvals
        else:
            inf_logprob = inf_dist.log_prob(h_given_x).sum(2)
            xr = x[None].repeat(num_samples_posterior, 1, 1)
            xr = xr.view(x.size(0) * num_samples_posterior, x.size(1))
            logq, mean_output = self.logq_joint(
                xr, h_given_x.view(-1, h.size(1)), return_mu=True
            )
            mean_output = mean_output.view(num_samples_posterior, x.size(0), x.size(1))
            logq = logq.view(num_samples_posterior, x.size(0))
            w = (logq - inf_logprob).softmax(dim=0)
            fvals = (x[None] - mean_output) / (self.logsigma.exp() ** 2)
            weighted_fvals = (fvals * w[:, :, None]).sum(0).detach()
            c = weighted_fvals

        mgn = c.norm(2, 1).mean()
        g_error_entropy = torch.mul(c, x).mean(0).sum()

        post = distributions.Normal(h.detach(), self.post_logsigma.exp())
        h_g_post = post.rsample()
        joint = self.logq_joint(x.detach(), h_g_post)
        post_ent = post.entropy().sum(1)

        elbo = joint + post_ent
        post_loss = -elbo.mean()

        if learn_post_sigma:
            self.post_optimizer.zero_grad()
            post_loss.backward()
            self.post_optimizer.step()

        if return_score:
            return g_error_entropy, mgn, c
        else:
            return g_error_entropy, mgn


class VERADiscreteGenerator(nn.Module):
    def __init__(
        self, g, noise_dim, post_lr=0.001, init_post_logsigma=0.1, temp=0.1, eps=1e-4
    ):
        super().__init__()
        self.g = g
        self.noise_dim = noise_dim
        self.post_logsigma = nn.Parameter(
            (
                torch.ones(
                    noise_dim,
                )
                * init_post_logsigma
            ).log()
        )
        self.temp = temp
        self.eps = eps
        self.post_optimizer = torch.optim.Adam([self.post_logsigma], lr=post_lr)

    def sample(self, n, requires_grad=False):
        """sample x, h ~ q(x, h)"""
        h = torch.randn((n, self.noise_dim)).to(next(self.parameters()).device)
        if requires_grad:
            h.requires_grad_()
        x = self.g.sample(h)
        return x, h

    def logq_joint(self, x, h, return_mu=False):
        """
        Join distribution of data and latent.
        """
        logph = distributions.Normal(0, 1).log_prob(h).sum(1)
        logits = self.g(h)
        log_px_given_h = F.log_softmax(logits, -1)
        log_px_given_h = torch.gather(
            log_px_given_h, -1, torch.argmax(x, -1).unsqueeze(-1)
        ).sum((1, 2))

        if return_mu:
            return log_px_given_h + logph, logits
        else:
            return log_px_given_h + logph

    def entropy_obj(
        self, x, h, num_samples_posterior=20, return_score=False, learn_post_sigma=True
    ):
        """
        Entropy objective using variational approximation with importance sampling.
        """
        inf_dist = distributions.Normal(h, self.post_logsigma.detach().exp())
        h_given_x = inf_dist.sample((num_samples_posterior,))
        inf_logprob = inf_dist.log_prob(h_given_x).sum(2)

        xr = x[None].repeat(num_samples_posterior, 1, 1, 1)
        xr = xr.view(x.size(0) * num_samples_posterior, x.size(1), x.size(2))
        xr.requires_grad_()

        logq, logits = self.logq_joint(
            xr, h_given_x.view(-1, h.size(1)), return_mu=True
        )
        logits = logits.view(num_samples_posterior, x.size(0), x.size(1), x.size(2))
        logq = logq.view(num_samples_posterior, x.size(0))

        w = (logq - inf_logprob).softmax(dim=0)

        # Grad_x q(x|z) = Grad_x Cat(x; f(z))
        # Use Gumbel Softmax Distribution as smooth approximation
        # x_eps = x[None] + self.eps
        # first_term = 1 / (logits.exp() / (x_eps ** self.temp)).sum(-1, keepdim=True)
        # fvals = first_term - logits.exp() / (x_eps ** 2) - (self.temp + 1) / x_eps

        # Use parameterization of Categorical log p(x) = W^T x - log Z where x is not restricted to x in {0, ..., k}^D but continuous
        fvals = logits.softmax(-1)

        weighted_fvals = (fvals * w[:, :, None, None]).sum(0).detach()
        c = weighted_fvals

        mgn = c.norm(2, 1).mean()
        g_error_entropy = torch.mul(c, x).mean(0).sum()

        post = distributions.Normal(h.detach(), self.post_logsigma.exp())
        h_g_post = post.rsample()
        joint = self.logq_joint(x.detach(), h_g_post)
        post_ent = post.entropy().sum(1)

        elbo = joint + post_ent
        post_loss = -elbo.mean()

        if learn_post_sigma:
            self.post_optimizer.zero_grad()
            post_loss.backward()
            self.post_optimizer.step()

        if return_score:
            return g_error_entropy, mgn, c
        else:
            return g_error_entropy, mgn

    def clamp_sigma(self, sigma, sigma_min=0.01):
        """
        Sigma clamping used for entropy estimator.
        """
        pass
