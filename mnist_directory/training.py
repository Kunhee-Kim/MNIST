import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import numpy as np


def training(tr_dataloader, cnn_model, args):

    training_batch = len(tr_dataloader)
    tr_criterion = nn.CrossEntropyLoss()
    tr_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=args.learning_rate)

    for epoch in range(args.training_epochs):
        avg_loss = 0

        for inputs, labels in tr_dataloader:

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            tr_optimizer.zero_grad()

            outputs = cnn_model(inputs)
            loss = tr_criterion(outputs, labels)
            loss.backward()
            tr_optimizer.step()

            avg_loss += loss/training_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_loss))
