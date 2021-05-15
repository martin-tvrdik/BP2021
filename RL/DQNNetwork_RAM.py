import torch.nn as nn
import torch.nn.functional as F


class DQNNetworkRAM(nn.Module):
    def __init__(self, inputs, out_actions):
        super().__init__()

        self.linear1 = nn.Linear(inputs, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, out_actions)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x
