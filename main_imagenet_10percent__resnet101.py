import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from persist_results import *

import torchvision
import torchvision.transforms as transforms
from dataset_classes import *
from utilsmark6 import ImageCaptionDataset, train_for_classification, train_for_retrieval
from transformations_graph import generate_graph

import matplotlib.pyplot as plt

# Importar modelo
import torchvision.models as models
resnet101 = models.resnet101()

# Wrapper del modelo
class ClassifierModule(nn.Module):
    def __init__(self):
        super(ClassifierModule,self).__init__()
        self.net = resnet101

    def forward(self,x):
        y = self.net(x)
        # Adaptamos la salida a la que utiliza utils.py
        return {'hidden': y, 'logits': y}


def run_it():
  transform = transforms.Compose(
      [transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),])

  print(f'Training')
  # Definamos algunos hiper-parámetros
  BATCH_SIZE = 32
  LR = 0.005
  EPOCHS = 5
  REPORTS_EVERY = 1

  # Generar Grafo de Reducciones
  l_redux = generate_graph(0.1, 0.1, 0.2, 3)

  net = ClassifierModule() # tu modelo de CNN (para clasificar en 10 clases)
  optimizer = optim.Adam(net.parameters()) # optimizador, e.g., optim.SGD, optim.Adam, ...
  criterion = nn.CrossEntropyLoss() # función de pérdida
  scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.1) # (opcional) optim.lr_scheduler proporciona varios métodos para ajustar el lr según el número de épocas

  train_loader = DataLoader(TrainingDataset(transform=transform, list='training_data_list_10percent.json'), batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=2)
  test_loader = DataLoader(ValidationDataset(transform=transform), batch_size=4*BATCH_SIZE,
                          shuffle=False, num_workers=2)

  print(f'Classic epochs')
  train_loss, acc = train_for_classification(net, train_loader, 
                                            test_loader, optimizer, 
                                            criterion, lista_reducciones=l_redux, reducir=False, lr_scheduler=scheduler, 
                                            epochs=EPOCHS, reports_every=REPORTS_EVERY)

  persist_tuple_prefix((train_loss, acc), "resnet101_classic__")

  print("Almacenando punto de control")
  PATH = "./resnet101__checkpoint_epoch_5_classic"
  torch.save({
              'epoch': 5,
              'model_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, PATH)

  print(f'Entropy Reductions')
  train_loss, acc = train_for_classification(net, train_loader, 
                                           test_loader, optimizer, 
                                           criterion, lista_reducciones=l_redux, reducir=True, lr_scheduler=scheduler, 
                                           epochs=1, reports_every=REPORTS_EVERY)

  persist_tuple_prefix((train_loss, acc), "resnet101_reduced__")

  print("Almacenando punto de control")
  PATH = "./resnet101__checkpoint_epoch_6_reduction"
  torch.save({
              'epoch': 6,
              'model_state_dict': net.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              }, PATH)
  #plot_results(train_loss, acc)

if __name__ == '__main__':
    run_it()