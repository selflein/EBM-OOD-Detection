from torch import nn


def make_mlp(dim_list, activation="relu", batch_norm=False, dropout=0):
    layers = []
    if len(dim_list) > 2:
        for dim_in, dim_out in zip(dim_list[:-2], dim_list[1:-1]):
            layers.append(nn.Linear(dim_in, dim_out))

            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))

            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            else:
                raise NotImplementedError(f"Activation '{activation}' not implemented!")

            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
    layers.append(nn.Linear(dim_list[-2], dim_list[-1]))
    model = nn.Sequential(*layers)
    return model


class SynthModel(nn.Module):
    def __init__(
        self,
        inp_dim,
        num_classes,
        hidden_dims=(50,),
        activation="leaky_relu",
        batch_norm=False,
        dropout=0.0,
    ):
        super().__init__()
        self.net = make_mlp(
            [
                inp_dim,
            ]
            + hidden_dims
            + [
                num_classes,
            ],
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def forward(self, inp):
        return self.net(inp)
