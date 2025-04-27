import torch
import torch.nn as nn

from config import TrainConfig


class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.linear = nn.Linear(1152, TrainConfig.num_classes)

    def forward(self, inp):
        out = self.conv1(inp)
        out = self.conv2(out)
        out = torch.flatten(out, start_dim=1)
        return self.linear(out)
