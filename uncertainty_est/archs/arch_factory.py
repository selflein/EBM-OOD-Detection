from uncertainty_est.archs.wrn import WideResNet
from uncertainty_est.archs.fc import SynthModel
from uncertainty_est.archs.resnet import ResNetGenerator
from torchvision.models import vgg16


def get_arch(name, config_dict: dict):
    if name == "wrn":
        return WideResNet(**config_dict)
    elif name == "vgg16":
        return vgg16(**config_dict)
    elif name == "fc":
        return SynthModel(**config_dict)
    elif name == "resnetgenerator":
        return ResNetGenerator(**config_dict)
    else:
        raise ValueError(f'Architecture "{name}" not implemented!')
