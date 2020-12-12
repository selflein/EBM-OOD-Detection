import torch


def accuracy(y: torch.Tensor, y_hat: torch.Tensor):
    return (y == y_hat.argmax(dim=1)).float().mean(0).item()
