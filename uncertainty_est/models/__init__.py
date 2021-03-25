from uncertainty_est import models

MODELS = {
    "HDGE": models.JEM.hdge.HDGEModel,
    "JEM": models.JEM.jem.JEM,
    "CEBaseline": models.ce_baseline.CEBaseline,
    "EnergyOOD": models.energy_finetuning.EnergyFinetune,
    "PriorNet": models.priornet.priornet.PriorNet,
    "JEMPriorNet": models.JEM.jem_priornet.JEMPriorNet,
    "VERA": models.JEM.vera.VERA,
    "VERAPriorNet": models.JEM.vera_priornet.VERAPriorNet,
    "VERAPosteriorNet": models.JEM.vera_posteriornet.VERAPosteriorNet,
    "NormalizingFlow": models.normalizing_flow.norm_flow.NormalizingFlow,
    "ApproxNormalizingFlow": models.normalizing_flow.approx_flow.ApproxNormalizingFlow,
    "RegularizedEBM": models.EBM.regularized_ebm.RegularizedEBM,
    "PerSampleNCE": models.EBM.per_sample_nce.PerSampleNCE,
    "IResNet": models.normalizing_flow.iresnet.IResNetFlow,
    "RealNVP": models.normalizing_flow.real_nvp.RealNVPModel,
    "VAE": models.vae.VAE,
    "NCE": models.EBM.nce.NoiseContrastiveEstimation,
}
