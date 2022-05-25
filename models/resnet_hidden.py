'''
A ResNet (He, Zhang, Ren and Sun, 2015) in PyTorch.
It does not contain pre-activation.

Reference:
He, K., Zhang, X., Ren, S. and Sun, J., 2015.
Deep Residual Learning for Image Recognition.
[Paper] Available at: <https://arxiv.org/pdf/1512.03385.pdf>
[Accessed 24 December 2021].
'''

import torch.nn as nn
import torch.nn.functional as F


# Basically the BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1):
        super(BasicBlock, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # Second layer
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # Decide if an explicit shortcut is needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * out_planes))

    def forward(self, x):
        input = x

        # First layer
        output = self.conv1(input)
        output = self.bn1(output)
        output = F.relu(output)

        # Second layer
        output = self.conv2(output)
        output = self.bn2(output)

        # Residual network shortcut
        output += self.shortcut(input)
        output = F.relu(output)

        return output


# Deeper variants require a more complex building block to avoid saturation (hence the name)
# Basically the Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, out_planes, stride=1):
        super(Bottleneck, self).__init__()

        # First layer
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)

        # Second layer
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

        # Third layer
        self.conv3 = nn.Conv2d(out_planes, self.expansion * out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * out_planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(self.expansion * out_planes))

    def forward(self, x):
        input = x

        # First layer
        output = self.conv1(input)
        output = self.bn1(output)
        output = F.relu(output)

        # Second layer
        output = self.conv2(output)
        output = self.bn2(output)
        output = F.relu(output)

        # Third layer
        output = self.conv3(output)
        output = self.bn3(output)

        # Residual network shortcut
        output += self.shortcut(input)
        output = F.relu(output)

        return output


# The actual implementation of the network
class ResNet(nn.Module):
    # Assume by default that we are using CIFAR-10, which is why there are 10 classes
    def __init__(self, block, num_blocks_per_layer, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # First part is to put everything in the right shapes and batch
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # The residual network layers constructed with the above components
        self.layer1 = self._construct_layer_from_block(block, 64, num_blocks_per_layer[0], stride=1)
        self.layer2 = self._construct_layer_from_block(block, 128, num_blocks_per_layer[1], stride=2)
        self.layer3 = self._construct_layer_from_block(block, 256, num_blocks_per_layer[2], stride=2)
        self.layer4 = self._construct_layer_from_block(block, 512, num_blocks_per_layer[3], stride=2)

        # Final layer which is matched with the 10 classes
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _construct_layer_from_block(self, block, out_planes, num_blocks, stride):
        # Construct the required strides
        strides = [stride] + [1] * (num_blocks - 1)

        # Generate the layers per block
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_planes, stride))
            self.in_planes = out_planes * block.expansion

        # Cast everything into an actual Pytorch layer (wrapper)
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x

        # Put the input in the initial layer
        output = self.conv1(input)
        output = self.bn1(output)
        output = F.relu(output)

        # Go through the residual layers
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)

        # Average and then get one line
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        # Final return the 10 classes predictions
        return output

    def feature_list(self, x):
        input = x
        out_list = []

        # Put the input in the initial layer
        output = self.conv1(input)
        output = self.bn1(output)
        output = F.relu(output)
        out_list.append(output)

        # Go through the residual layers
        output = self.layer1(output)
        out_list.append(output)
        output = self.layer2(output)
        out_list.append(output)
        output = self.layer3(output)
        out_list.append(output)
        output = self.layer4(output)
        out_list.append(output)

        # Average and then get one line
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.linear(output)

        # Final return the 10 classes predictions
        return output, out_list

    # Returns a feature extracted at a specific layer
    def intermediate_forward(self, x, layer_index):
        out = F.relu(self.bn1(self.conv1(x)))

        if layer_index == 1:
            out = self.layer1(out)

        elif layer_index == 2:
            out = self.layer1(out)
            out = self.layer2(out)

        elif layer_index == 3:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)

        elif layer_index == 4:
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

        return out

    # Returns features of penultimate layer
    def penultimate_forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        penultimate = self.layer4(out)
        out = F.avg_pool2d(penultimate, 4)
        out = out.view(out.size(0), -1)

        y = self.linear(out)
        return y, penultimate


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])