import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torchvision.models as models


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(101)

import torch
import torch.nn as nn
import torch.nn.functional as F


class torch_organization_model(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128, 128], out_dim = 64):
        super(torch_organization_model, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
    

class torch_top_model_cifar10(nn.Module):
    def __init__(self, input_dim=190, hidden_units=[64, 64], num_classes=10):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_units[0])  # ✅ Corrected to use dynamic input_dim
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.dropout1 = nn.Dropout(0.5)
        
        self.hidden_layer = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.dropout2 = nn.Dropout(0.5)

        self.output_layer = nn.Linear(hidden_units[1], num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden_layer(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        return x


class surrogate_model_for_backdoor_cifar10(nn.Module):
    def __init__(self, input_dim=256, hidden_units=[64, 64], num_classes=10):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_units[0])  # ✅ Corrected to use dynamic input_dim
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.dropout1 = nn.Dropout(0.5)
        
        self.hidden_layer = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.dropout2 = nn.Dropout(0.5)

        self.output_layer = nn.Linear(hidden_units[1], num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.hidden_layer(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.output_layer(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class torch_organization_model_cifar10(nn.Module):
    def __init__(self, out_dim=64, gpu=True):
        super(torch_organization_model_cifar10, self).__init__()
        self.gpu = gpu
        self.in_channels = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # ResNet-20 consists of 3 stages, each with 3 residual blocks
        self.layer1 = self._make_layer(16, 3, stride=1)
        self.layer2 = self._make_layer(32, 3, stride=2)
        self.layer3 = self._make_layer(64, 3, stride=2)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # Output size (batch, 64, 1, 1)
        self.fc = nn.Linear(64, out_dim)  # Final output layer with out_dim=64

        if gpu:
            self.cuda()

    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # First block handles stride, others are stride=1
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        if self.gpu:
            x = x.cuda()

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.global_avg_pool(out)
        out = torch.flatten(out, 1)  # Flatten to (batch_size, 64)
        out = self.fc(out)  # Output dimension = 64
        return out



class torch_top_model(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128, 128], num_classes=2):
        super(torch_top_model, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        # x = torch.sigmoid(x).squeeze()
        # x = self.softmax(x)

        return x

class MlpModel(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128], num_classes=10):
        super(MlpModel, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        x = torch.sigmoid(x)

        return x
