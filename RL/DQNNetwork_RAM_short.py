import torch.nn as nn
import torch.nn.functional as F


class DQNNetworkRAM(nn.Module):
    def __init__(self, inputs, out_actions):
        super().__init__()

        self.linear1 = nn.Linear(inputs, 1)
        self.linear2 = nn.Linear(1, out_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
