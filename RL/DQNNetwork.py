import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Convolution network taken from nature DQN paper 4x84x84 input.
    """

    def __init__(self, input_channels, out_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.linear1 = nn.Linear(64 * 7 * 7, 512)
        self.linear2 = nn.Linear(512, out_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 64 * 7 * 7)  # flatten
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x
