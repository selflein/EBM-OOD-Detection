from uncertainty_est.archs.wrn import WideResNet
from torchvision.models import vgg16


def get_arch(name: str, config_dict: dict):
    if name == "wrn":
        return WideResNet(**config_dict)
    elif name == "vgg16":
        return vgg16()
    else:
        raise ValueError(f'Architecture "{name}" not implemented!')
