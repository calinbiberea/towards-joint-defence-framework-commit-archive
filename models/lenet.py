'''
A LeNet-5 in PyTorch.

Reference:
Yann LeCun et al, 1998.
Gradient-Based Learning Applied to Document Recognition.
[Paper] Available at: <http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf> 
[Accessed 26 December 2021].
'''

import torch.nn as nn
import torch.nn.functional as F


# Defining the Network (LeNet-5)
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # A convolutional layer (in LeNet-5, 32x32 images are given as input,
        # so we need padding of 2 for MNIST), followed by max-pooling
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        # Second layer
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=True,
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        # Third layer (fully connected layer)
        self.fc1 = nn.Linear(
            64 * 7 * 7, 1024
        )

        # Output layer
        self.fc2 = nn.Linear(
            1024, 10
        )

    def forward(self, x):
        input = x

        # First layer (ReLU activation), max-pool with 2x2 grid
        output = self.conv1(input)
        output = F.relu(output)
        output = self.max_pool_1(output)

        # Second Layer
        output = self.conv2(output)
        output = F.relu(output)
        output = self.max_pool_2(output)

        # Flatten to match network (16 * 5 * 5), given
        # https://stackoverflow.com/a/42482819/7551231
        output = output.view(-1, 64 * 7 * 7)

        # Third layer
        output = self.fc1(output)
        output = F.relu(output)

        # Output
        output = self.fc2(output)
        return output

    # Returns an array of mostly the shape of the features
    def feature_list(self, x):
        input = x

        out_list = []

        # First layer (ReLU activation), max-pool with 2x2 grid
        output = self.conv1(input)
        output = F.relu(output)
        output = self.max_pool_1(output)
        out_list.append(output)

        # Second Layer
        output = self.conv2(output)
        output = F.relu(output)
        output = self.max_pool_2(output)
        out_list.append(output)

        # Flatten to match network (16 * 5 * 5), given
        # https://stackoverflow.com/a/42482819/7551231
        output = output.view(-1, 64 * 7 * 7)

        # Third layer
        output = self.fc1(output)
        output = F.relu(output)
        out_list.append(output)

        # Output
        output = self.fc2(output)

        return output, out_list

    # Returns a feature extracted at a specific layer
    def intermediate_forward(self, x, layer_index):
        input = x

        if (layer_index == 0):
            # First layer (ReLU activation), max-pool with 2x2 grid
            output = self.conv1(input)
            output = F.relu(output)
            output = self.max_pool_1(output)

            return output
        elif (layer_index == 1):
            # First layer (ReLU activation), max-pool with 2x2 grid
            output = self.conv1(input)
            output = F.relu(output)
            output = self.max_pool_1(output)

            # Second Layer
            output = self.conv2(output)
            output = F.relu(output)
            output = self.max_pool_2(output)

            return output
        elif (layer_index == 2):
            # First layer (ReLU activation), max-pool with 2x2 grid
            output = self.conv1(input)
            output = F.relu(output)
            output = self.max_pool_1(output)

            # Second Layer
            output = self.conv2(output)
            output = F.relu(output)
            output = self.max_pool_2(output)

            # Flatten to match network (16 * 5 * 5), given
            # https://stackoverflow.com/a/42482819/7551231
            output = output.view(-1, 64 * 7 * 7)

            # Third layer
            output = self.fc1(output)
            output = F.relu(output)

            return output

        raise Exception('Invalid layer index')

    # Returns features of penultimate layer (should coincide with 3rd layer for lenet)
    def penultimate_forward(self, x):
        input = x

        # First layer (ReLU activation), max-pool with 2x2 grid
        output = self.conv1(input)
        output = F.relu(output)
        output = self.max_pool_1(output)

        # Second Layer
        output = self.conv2(output)
        output = F.relu(output)
        output = self.max_pool_2(output)

        # Flatten to match network (16 * 5 * 5), given
        # https://stackoverflow.com/a/42482819/7551231
        output = output.view(-1, 64 * 7 * 7)

        # Third layer
        output = self.fc1(output)
        output = F.relu(output)

        return output
