import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torchvision
import torchvision.transforms as transforms
from dataset_classes import *
from utils import ImageCaptionDataset, train_for_classification, train_for_retrieval

import matplotlib.pyplot as plt

import torchvision.models as models
resnet18 = models.resnet18()
resnet18.fc.out_features = 10
class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule,self).__init__()
        self.net = resnet18
        #for param in self.net.features.parameters():
          #param.requires_grad = False

    def forward(self,x):
        y = self.net(x)
        # Adaptamos la salida a la que utiliza utils.py
        return {'hidden': y, 'logits': y}


def run_it():
  transform = transforms.Compose(
      [transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
  trainloader = DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
  testloader = DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  print(f'Training')
  # Definamos algunos hiper-parámetros
  BATCH_SIZE = 32//2
  LR = 0.005
  EPOCHS = 5
  REPORTS_EVERY = 1

  net = ClassifierModule() # tu modelo de CNN (para clasificar en 10 clases)
  optimizer = optim.Adam(net.parameters()) # optimizador, e.g., optim.SGD, optim.Adam, ...
  criterion = nn.CrossEntropyLoss() # función de pérdida
  scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1) # (opcional) optim.lr_scheduler proporciona varios métodos para ajustar el lr según el número de épocas

  train_loader = DataLoader(trainset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)
  test_loader = DataLoader(testset, batch_size=4*BATCH_SIZE,
                          shuffle=False, num_workers=2)

  print(f'Before train for classification')
  train_loss, acc = train_for_classification(net, train_loader, 
                                            test_loader, optimizer, 
                                            criterion, lr_scheduler=scheduler, 
                                            epochs=EPOCHS, reports_every=REPORTS_EVERY)

  #plot_results(train_loss, acc)

if __name__ == '__main__':
    run_it()