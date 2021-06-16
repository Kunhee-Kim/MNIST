import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np

class dataloader():

    def __init__(self, args):
        self.args = args
        transform = transforms.ToTensor()
        self.train_mnist = dsets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.test_mnist = dsets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.train_dataloader = DataLoader(self.train_mnist, batch_size=args.batch_size, shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.test_mnist, batch_size=args.batch_size, shuffle=True, drop_last=True)
