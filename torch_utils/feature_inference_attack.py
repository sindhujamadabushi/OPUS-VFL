import torch
from torch import nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim=100, target_width=11):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.target_width = target_width
        self.input_channels = 3
        self.height = 32

        self.input_dim = self.input_channels * self.height * (32 - target_width) + self.noise_dim
        self.output_dim = self.input_channels * self.height * target_width

        self.fc1 = nn.Linear(self.input_dim, 1024)
        self.fc2 = nn.Linear(1024, self.output_dim)

    def forward(self, x_adv, r_t):
        B = x_adv.size(0)
        x_adv_flat = x_adv.view(B, -1)
        x = torch.cat([x_adv_flat, r_t], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(B, self.input_channels, self.height, self.target_width)
