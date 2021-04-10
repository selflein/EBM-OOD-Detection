import torch
from torch import nn
import torch.nn.functional as F


class SequenceClassifier(nn.Module):
    def __init__(
        self, in_channels, num_filters, kernel_size, fc_hidden_size, num_classes
    ):
        super().__init__()
        self.input_size = in_channels

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, num_filters // 4, kernel_size=kernel_size),
            nn.ReLU(True),
            nn.Conv1d(num_filters // 4, num_filters // 2, kernel_size=kernel_size),
            nn.ReLU(True),
            nn.Conv1d(num_filters // 2, num_filters, kernel_size=kernel_size),
            nn.ReLU(True),
        )
        self.fc = nn.Sequential(
            nn.Linear(num_filters, fc_hidden_size),
            nn.ReLU(True),
            nn.Linear(fc_hidden_size, num_classes),
        )

    def forward(self, inp):
        inp = F.one_hot(inp.long(), self.input_size).float().transpose(1, 2)

        out = self.conv(inp)
        out, _ = out.max(dim=-1)
        out = self.fc(out)
        return out


class SequenceGenerativeModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.input_size = input_size

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        one_hots = F.one_hot(x.long(), self.input_size).float()

        out, _ = self.lstm(one_hots)
        logits = self.classifier(out)
        return logits

    def log_prob(self, x):
        logits = self.forward(x)
        log_prob = torch.gather(
            F.log_softmax(logits, dim=-1)[:, :-1], -1, x[:, 1:].long().unsqueeze(-1)
        ).sum((1, 2))
        return log_prob
