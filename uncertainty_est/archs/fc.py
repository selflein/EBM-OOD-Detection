from torch import nn


class SynthModel(nn.Module):
    def __init__(self, inp_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp_dim, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 50),
            nn.LeakyReLU(),
            nn.Linear(50, num_classes),
        )

    def forward(self, inp):
        return self.net(inp)
