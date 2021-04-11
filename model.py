# https://github.com/eriklindernoren/PyTorch-GAN

import torchvision.transforms as transforms
from torch.autograd import Variable

import torch.nn as nn
import torch


class ReshapeLayer(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class ImageEmbedding(nn.Module):
    def __init__(self, image_dimensions=(405, 720), time_window=5):
        super().__init__()

        self.layers = nn.Sequential(
            ReshapeLayer((1, time_window * 3, *image_dimensions)),
            nn.Conv2d(time_window * 3, 16, 5),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 16, 5),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 32, 3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, 3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(4),
            nn.Conv2d(64, 256, 3),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),
            ReshapeLayer((1, -1)),
            nn.Linear(51200, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(1024, 256),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, image_dimensions=(405, 720)):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, image_dimensions[0] * image_dimensions[1]),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(inplace=True),
            ReshapeLayer((1, 1, *image_dimensions)),
            nn.Conv2d(1, 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(3, 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(3, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.layers(z)


class Predictor(nn.Module):
    def __init__(self, embedding, generator):
        super().__init__()
        self.embedding = embedding
        self.generator = generator

    def forward(self, x):
        return self.generator(self.embedding(x))
