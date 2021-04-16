import yaml
from pathlib import Path

from uncertainty_est.models.JEM.hdge import HDGEModel
from uncertainty_est.models.JEM.mcmc import MCMC
from uncertainty_est.models.ce_baseline import CEBaseline
from uncertainty_est.models.energy_finetuning import EnergyFinetune
from uncertainty_est.models.priornet.priornet import PriorNet
from uncertainty_est.models.JEM.vera import VERA
from uncertainty_est.models.JEM.vera_priornet import VERAPriorNet
from uncertainty_est.models.JEM.vera_posteriornet import VERAPosteriorNet
from uncertainty_est.models.normalizing_flow.norm_flow import NormalizingFlow
from uncertainty_est.models.normalizing_flow.approx_flow import ApproxNormalizingFlow
from uncertainty_est.models.EBM.regularized_ebm import RegularizedEBM
from uncertainty_est.models.EBM.per_sample_nce import PerSampleNCE
from uncertainty_est.models.normalizing_flow.iresnet import IResNetFlow
from .normalizing_flow.real_nvp import RealNVPModel
from .vae import VAE
from .EBM.nce import NoiseContrastiveEstimation
from .EBM.flow_contrastive_estimation import FlowContrastiveEstimation
from .JEM.nce_priornet import NCEPriorNet


MODELS = {
    "HDGE": HDGEModel,
    "JEM": MCMC,
    "CEBaseline": CEBaseline,
    "EnergyOOD": EnergyFinetune,
    "PriorNet": PriorNet,
    "VERA": VERA,
    "VERAPriorNet": VERAPriorNet,
    "VERAPosteriorNet": VERAPosteriorNet,
    "NormalizingFlow": NormalizingFlow,
    "ApproxNormalizingFlow": ApproxNormalizingFlow,
    "RegularizedEBM": RegularizedEBM,
    "PerSampleNCE": PerSampleNCE,
    "IResNet": IResNetFlow,
    "RealNVP": RealNVPModel,
    "VAE": VAE,
    "NCE": NoiseContrastiveEstimation,
    "NCEPriorNet": NCEPriorNet,
    "FlowCE": FlowContrastiveEstimation,
}


def load_model(model_folder: Path, last=False, *args, **kwargs):
    ckpts = [file for file in model_folder.iterdir() if file.suffix == ".ckpt"]
    ckpts = [ckpt for ckpt in ckpts if not "last" in ckpt.stem == last]
    assert len(ckpts) > 0

    checkpoint_path = sorted(ckpts)[-1]
    return load_checkpoint(checkpoint_path, *args, **kwargs)


def load_checkpoint(checkpoint_path: Path, *args, **kwargs):
    with (checkpoint_path.parent / "config.yaml").open("r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = MODELS[config["model_name"]].load_from_checkpoint(
        checkpoint_path, *args, **kwargs
    )
    return model, config
