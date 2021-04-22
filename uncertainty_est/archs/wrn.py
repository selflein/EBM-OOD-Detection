"""
WideResnet architecture adapted from https://github.com/meliketoy/wide-resnet.pytorch
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def conv3x3(in_planes, out_planes, stride=1):
    """
    Convolution with 3x3 kernels.
    """
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )


def conv_init(m):
    """
    Initializing convolution layers.
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find("BatchNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class Identity(nn.Module):
    """
    Identity norm as a stand in for no BN.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of model.
        """
        return x


class wide_basic(nn.Module):
    """
    One block in the Wide resnet.
    """

    def __init__(
        self,
        in_planes,
        planes,
        dropout_rate,
        stride=1,
        norm=None,
        leak=0.2,
        first=False,
    ):
        super(wide_basic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=True
        )

        self.first = first

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        """
        Forward pass of block.
        """
        if (
            self.first
        ):  # if it's the first block, don't apply the first batchnorm to the data
            out = self.dropout(self.conv1(self.lrelu(x)))
        else:
            out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)

        return out


def get_norm(n_filters, norm):
    """
    Get batchnorm or other.
    """
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)
    elif norm == "group":
        return nn.GroupNorm(32, n_filters)


class WideResNet(nn.Module):
    """
    Wide resnet model.
    """

    def __init__(
        self,
        depth,
        widen_factor,
        num_classes=10,
        input_channels=3,
        sum_pool=False,
        norm=None,
        leak=0.2,
        dropout=0.0,
        strides=(1, 2, 2),
        bottleneck_dim=None,
        bottleneck_channels_factor=None,
    ):
        super(WideResNet, self).__init__()
        self.leak = leak
        self.in_planes = 16
        self.sum_pool = sum_pool
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)
        self.bottleneck_dim = bottleneck_dim
        self.bottleneck_channels_factor = bottleneck_channels_factor

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen_factor

        print("| Wide-Resnet %dx%d" % (depth, k))
        nStages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(input_channels, nStages[0])
        self.layer1 = self._wide_layer(
            wide_basic, nStages[1], n, dropout, stride=strides[0], first=True
        )
        self.layer2 = self._wide_layer(
            wide_basic, nStages[2], n, dropout, stride=strides[1]
        )
        self.layer3 = self._wide_layer(
            wide_basic, nStages[3], n, dropout, stride=strides[2]
        )
        self.bn1 = get_norm(nStages[3], self.norm)
        self.last_dim = nStages[3]
        self.linear = nn.Linear(nStages[3], num_classes)

        if self.bottleneck_dim is not None:
            self.bottleneck = nn.Sequential(
                nn.Linear(nStages[3], nStages[3] // 2),
                nn.ReLU(True),
                nn.Linear(nStages[3] // 2, self.bottleneck_dim),
                nn.ReLU(True),
                nn.Linear(self.bottleneck_dim, nStages[3] // 2),
                nn.ReLU(True),
                nn.Linear(nStages[3] // 2, nStages[3]),
            )

        self.apply(conv_init)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, first=False):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for i, stride in enumerate(strides):
            if first and i == 0:  # first block of first layer has no BN
                layers.append(
                    block(
                        self.in_planes,
                        planes,
                        dropout_rate,
                        stride,
                        norm=self.norm,
                        first=True,
                    )
                )
            else:
                layers.append(
                    block(self.in_planes, planes, dropout_rate, stride, norm=self.norm)
                )
            self.in_planes = planes

        if self.bottleneck_channels_factor is not None:
            bottleneck_channels = int(self.in_planes * self.bottleneck_channels_factor)
            layers.extend(
                [
                    nn.Conv2d(self.in_planes, bottleneck_channels, kernel_size=1),
                    nn.Conv2d(bottleneck_channels, self.in_planes, kernel_size=1),
                ]
            )

        return nn.Sequential(*layers)

    def encode(self, x, vx=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        if self.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)
        else:
            out = F.avg_pool2d(out, out.shape[2:])
        out = out.view(out.size(0), -1)

        if self.bottleneck_dim is not None:
            out = self.bottleneck(out)
        return out

    def forward(self, x, vx=None):
        """
        Forward pass. TODO: purpose of vx?
        """
        out = self.encode(x, vx)

        return self.linear(out)


if __name__ == "__main__":
    import torch

    for strides in [(1, 2, 2), (1, 1, 2), (1, 1, 1)]:
        wrn = WideResNet(28, 10, strides=strides, bottleneck_channels_factor=0.1)
        print(wrn)
        out = wrn(torch.zeros(1, 3, 32, 32))
        print(strides, out.shape)
