from uncertainty_est.models.JEM.hdge import HDGEModel
from uncertainty_est.models.JEM.jem import JEM
from uncertainty_est.models.JEM.jem_priornet import JEMPriorNet
from uncertainty_est.models.JEM.hdge_priornet import HDGEPriorNetModel
from uncertainty_est.models.ce_baseline import CEBaseline
from uncertainty_est.models.energy_finetuning import EnergyFinetune
from uncertainty_est.models.priornet.priornet import PriorNet
from uncertainty_est.models.JEM.vera import VERA
from uncertainty_est.models.JEM.vera_priornet import VERAPriorNet
from uncertainty_est.models.JEM.vera_posteriornet import VERAPosteriorNet
from uncertainty_est.models.normalizing_flow.norm_flow import NormalizingFlow
from uncertainty_est.models.normalizing_flow.approx_flow import ApproxNormalizingFlow
from uncertainty_est.models.regularized_ebm import RegularizedEBM

MODELS = {
    "HDGE": HDGEModel,
    "JEM": JEM,
    "CEBaseline": CEBaseline,
    "EnergyOOD": EnergyFinetune,
    "PriorNet": PriorNet,
    "JEMPriorNet": JEMPriorNet,
    "HDGEPriorNet": HDGEPriorNetModel,
    "VERA": VERA,
    "VERAPriorNet": VERAPriorNet,
    "VERAPosteriorNet": VERAPosteriorNet,
    "NormalizingFlow": NormalizingFlow,
    "ApproxNormalizingFlow": ApproxNormalizingFlow,
    "RegularizedEBM": RegularizedEBM,
}
