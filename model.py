# ELEC475 Lab 5
# Nicholas Chivaran - 18nc34
# Samantha Hawco - 18srh5

# imports
import torch.nn as nn


class PetNet(nn.Module):

    def __init__(self, resnet, fc=None, pretrained=False):
        super(PetNet, self).__init__()
        self.resnet = resnet

        if pretrained:  # freeze ResNet weights if we do not want to train ResNet from scratch
            for param in self.resnet.parameters():
                param.requires_grad = False
        else:  # randomize starting weights for non-pretrained ResNet model
            self.init_resnet_weights(mean=0.0, std=0.01)

        self.fc_layers = fc
        if self.fc_layers is None:
            self.fc_layers = nn.Sequential(
                nn.Linear(1000, 256),
                nn.ReLU(),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 16),
                nn.ReLU(),
                nn.Linear(16, 2),  # output of 2 for x and y coordinates
                nn.ReLU()
            )
            self.init_fc_weights(mean=0.0, std=0.01)

    def forward(self, X):
        return self.fc_layers(self.resnet(X))

    def init_resnet_weights(self, mean, std):
        for param in self.resnet.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def init_fc_weights(self, mean, std):
        for param in self.fc_layers.parameters():
            nn.init.normal_(param, mean=mean, std=std)


