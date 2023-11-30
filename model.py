# ELEC475 Lab 5
# Nicholas Chivaran - 18nc34
# Samantha Hawco - 18srh5

# imports
import torch.nn as nn


class PetNet(nn.Module):

    def __init__(self, resnet):
        super(PetNet, self).__init__()
        self.resnet = resnet

        # freeze resnet weights
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc_layers = nn.Sequential(
            nn.Linear(1000, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 2),   #ouput of 2 for x and y coordinates
            nn.ReLU()
        )

    def forward(self, X):
        return self.fc_layers(self.resnet(X))