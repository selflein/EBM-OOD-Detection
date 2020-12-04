import torch


def to_np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()
