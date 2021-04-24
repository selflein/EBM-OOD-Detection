import torch
from torch import nn


class KHotCrossEntropyLoss(nn.Module):
    def __init__(self, dim=-1):
        super(KHotCrossEntropyLoss, self).__init__()
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        return torch.mean(torch.sum(-target * pred, dim=self.dim))


def smooth_one_hot(labels, classes, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    label_shape = torch.Size((labels.size(0), classes))
    with torch.no_grad():
        dist = torch.empty(size=label_shape, device=labels.device)
        dist.fill_(smoothing / (classes - 1))
        dist.scatter_(1, labels.data.unsqueeze(-1), 1.0 - smoothing)
    return dist
