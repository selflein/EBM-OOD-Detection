from uncertainty_est.archs.wrn import WideResNet


def get_arch(name: str, config_dict: dict):
    if name == "wrn":
        return WideResNet(**config_dict)

    else:
        raise ValueError(f'Architecture "{name}" not implemented!')
