from torch import nn as nn
from torch.nn import init as nninit


class GeneratorBlock(nn.Module):
    """ResNet-style block for the generator model."""

    def __init__(self, in_chans, out_chans, upsample=False):
        super().__init__()

        self.upsample = upsample

        if in_chans != out_chans:
            self.shortcut_conv = nn.Conv2d(in_chans, out_chans, kernel_size=1)
        else:
            self.shortcut_conv = None
        self.bn1 = nn.BatchNorm2d(in_chans)
        self.conv1 = nn.Conv2d(in_chans, in_chans, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_chans)
        self.conv2 = nn.Conv2d(in_chans, out_chans, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, *inputs):
        x = inputs[0]

        if self.upsample:
            shortcut = nn.functional.upsample(x, scale_factor=2, mode="nearest")
        else:
            shortcut = x

        if self.shortcut_conv is not None:
            shortcut = self.shortcut_conv(shortcut)

        x = self.bn1(x)
        x = self.relu(x)
        if self.upsample:
            x = nn.functional.upsample(x, scale_factor=2, mode="nearest")
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x + shortcut


class ResNetGenerator(nn.Module):
    """The generator model."""

    def __init__(self, unit_interval, feats=128):
        super().__init__()

        self.input_linear = nn.Linear(128, 4 * 4 * feats)
        self.block1 = GeneratorBlock(feats, feats, upsample=True)
        self.block2 = GeneratorBlock(feats, feats, upsample=True)
        self.block3 = GeneratorBlock(feats, feats, upsample=True)
        self.output_bn = nn.BatchNorm2d(feats)
        self.output_conv = nn.Conv2d(feats, 3, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.feats = feats

        # Apply Xavier initialisation to the weights
        relu_gain = nninit.calculate_gain("relu")
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                gain = relu_gain if module != self.input_linear else 1.0
                nninit.xavier_uniform(module.weight.data, gain=gain)
                module.bias.data.zero_()

        if unit_interval:
            self.final_act = nn.functional.sigmoid
        else:
            self.final_act = nn.functional.tanh

        self.last_output = None

    def forward(self, *inputs):
        x = inputs[0]

        x = self.input_linear(x)
        x = x.view(-1, self.feats, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output_bn(x)
        x = self.relu(x)
        x = self.output_conv(x)
        x = self.final_act(x)

        self.last_output = x

        return x
