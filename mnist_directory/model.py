import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

# LinearNet


class LinearNet(nn.Module):
    def __init__(self, input_size, num_layers, layers_size, output_size):
        super(LinearNet, self).__init__()

        self.linears = nn.ModuleList([nn.Linear(input_size, layers_size[0])])
        self.linears.extend([nn.Linear(layers_size[i-1], layers_size[i]) for i in range(1, num_layers-1)])
        self.linears.append(nn.Linear(layers_size[num_layers-2], output_size))

    def forward(self, x):
        return x


# CNN
class CNN(nn.Module):

    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.dropout = None
        self.linear_array = LinearNet(input_size=9216, num_layers=args.layers, layers_size=args.layer_features, output_size= 10)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        self.dropout = nn.Dropout(self.args.layer_dropout[0])
        x = self.dropout(x)
        x = torch.flatten(x, 1)

        for l in range(self.args.layers):

            x = (self.linear_array.linears[l])(x)
            if l < self.args.layers-1 :
                x = F.relu(x)

        output = F.log_softmax(x, dim=1)

        return output
