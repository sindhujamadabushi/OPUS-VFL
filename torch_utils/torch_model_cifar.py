import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torchvision.models as models
import torch.nn.init as init


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

def weights_init(m):
    # classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, kernel_size, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant",
                                                  0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, kernel_size, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], kernel_size, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], kernel_size, stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(weights_init)

    def _make_layer(self, block, planes, num_blocks, kernel_size, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, kernel_size, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        # [bs,3,32,16]
        out = F.relu(self.bn1(self.conv1(x)))
        # [bs,16,32,16]
        out = self.layer1(out)
        # [bs,16,32,16]
        out = self.layer2(out)
        # [bs,32,16,8]
        out = self.layer3(out)
        # [bs,64,8,4]
        out = F.avg_pool2d(out, out.size()[2:])
        # [bs,64,1,1]
        out = out.view(out.size(0), -1)
        # [bs,64]
        out = self.linear(out)
        # [bs,10]
        return out


def resnet20(kernel_size=(3, 3), num_classes=10):
    return ResNet(block=BasicBlock, num_blocks=[3, 3, 3], kernel_size=kernel_size, num_classes=num_classes)

class BottomModelForCifar10(nn.Module):
    def __init__(self):
        super(BottomModelForCifar10, self).__init__()
        self.resnet20 = resnet20(num_classes=10)

    def forward(self, x):
        x = self.resnet20(x)
        return x
    # def forward(self, x):
    #     # Forward pass through the layers up to the pooling layer.
    #     out = F.relu(self.resnet20.bn1(self.resnet20.conv1(x)))
    #     out = self.resnet20.layer1(out)
    #     out = self.resnet20.layer2(out)
    #     out = self.resnet20.layer3(out)
    #     # Average pooling gives shape [batch_size, 64, 1, 1]
    #     out = F.avg_pool2d(out, out.size()[2:])
    #     # Flatten to [batch_size, 64] - this is your embedding.
    #     embedding = out.view(out.size(0), -1)
    #     return embedding
    
class SurrogateModelForBackdoor(nn.Module):
    def __init__(self, bottom_output_size):
        super(TopModelForCifar10, self).__init__()
        self.bottom_output_size = bottom_output_size
        # Total input dimension is number of clients multiplied by the size of each bottom output.
        input_dim =bottom_output_size

        # Define layers using input_dim. You can adjust hidden dimensions as needed.
        self.fc1top = nn.Linear(input_dim, 20)
        self.bn0top = nn.BatchNorm1d(input_dim)  # Updated to match the input dimension
        self.fc2top = nn.Linear(20, 10)
        self.bn1top = nn.BatchNorm1d(20)
        self.fc3top = nn.Linear(10, 10)
        self.bn2top = nn.BatchNorm1d(10)
        self.fc4top = nn.Linear(10, 10)
        self.bn3top = nn.BatchNorm1d(10)

    def forward(self, *bottom_outputs):
        if len(bottom_outputs) != self.num_clients:
            raise ValueError(f"Expected {self.num_clients} bottom outputs, but got {len(bottom_outputs)}")
        # Concatenate all bottom model outputs along the feature dimension.
        output_bottom_models = torch.cat(bottom_outputs, dim=1)
        x = self.fc1top(F.relu(self.bn0top(output_bottom_models)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)


class TopModelForCifar10(nn.Module):
    def __init__(self, num_clients, bottom_output_size):
        super(TopModelForCifar10, self).__init__()
        self.num_clients = num_clients
        self.bottom_output_size = bottom_output_size
        # Total input dimension is number of clients multiplied by the size of each bottom output.
        input_dim = num_clients * bottom_output_size

        # Define layers using input_dim. You can adjust hidden dimensions as needed.
        self.fc1top = nn.Linear(input_dim, 20)
        self.bn0top = nn.BatchNorm1d(input_dim)  # Updated to match the input dimension
        self.fc2top = nn.Linear(20, 10)
        self.bn1top = nn.BatchNorm1d(20)
        self.fc3top = nn.Linear(10, 10)
        self.bn2top = nn.BatchNorm1d(10)
        self.fc4top = nn.Linear(10, 10)
        self.bn3top = nn.BatchNorm1d(10)

    def forward(self, *bottom_outputs):
        if len(bottom_outputs) != self.num_clients:
            raise ValueError(f"Expected {self.num_clients} bottom outputs, but got {len(bottom_outputs)}")
        # Concatenate all bottom model outputs along the feature dimension.
        output_bottom_models = torch.cat(bottom_outputs, dim=1)
        x = self.fc1top(F.relu(self.bn0top(output_bottom_models)))
        x = self.fc2top(F.relu(self.bn1top(x)))
        x = self.fc3top(F.relu(self.bn2top(x)))
        x = self.fc4top(F.relu(self.bn3top(x)))
        return F.log_softmax(x, dim=1)