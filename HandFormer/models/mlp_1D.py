import torch
import torch.nn as nn
import torch.nn.functional as F

from models.activation import activation_factory


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu', dropout=0, add_bn_layer=True):
        super().__init__()
        channels = [in_channels] + out_channels
        self.layers = nn.ModuleList()
        for i in range(1, len(channels)):
            if dropout > 0.001:
                self.layers.append(nn.Dropout(p=dropout))
            self.layers.append(nn.Linear(channels[i-1], channels[i]))
            if add_bn_layer:
                self.layers.append(nn.BatchNorm1d(channels[i]))
            self.layers.append(activation_factory(activation))

    def forward(self, x):
        # Input shape: (N,C,T,V)
        for layer in self.layers:
            x = layer(x)
        return x

