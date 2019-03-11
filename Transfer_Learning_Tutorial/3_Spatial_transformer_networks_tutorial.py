#spatial transformer networks

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#download MNIST dataset

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=True, download=True,
                   transform = transforms.Compose(
                       [
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ]
                   )), batch_size = 64, shuffle=True, num_workers=4
)



test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='.', train=False,
                   transform = transforms.Compose(
                       [
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,),(0.3081,))
                       ]
                   )), batch_size = 64, shuffle=True, num_workers=4
)
