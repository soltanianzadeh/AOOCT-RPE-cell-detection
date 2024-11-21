'''
Code snippets from: https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F



BN_MOMENTUM = 0.1
activations = {
               'relu': nn.ReLU(inplace=True),
               'leakyrelu': nn.LeakyReLU(inplace=True),
               'celu': nn.CELU(inplace=True),
               'elu': nn.ELU(inplace=True),
               }

def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding
    """

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class TransitionLayer(nn.Module):
    def __init__(self, inplanes):
        super(TransitionLayer, self).__init__()
        self.norm = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(inplanes, inplanes//2, kernel_size=1, bias=False)
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        return self.downsample(self.conv(self.relu(self.norm(x))))


class DenseLayer(nn.Module):
    def __init__(self, inplanes, growth_rate, bn_size, drop_rate, activation='relu'):
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(inplanes)
        self.relu1 = activations[activation]
        self.conv1 = nn.Conv2d(inplanes, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = activations[activation]
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output               

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input
    
        bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class DenseBlock(nn.ModuleDict):

    def __init__(self, inplanes, num_layers, bn_size, growth_rate, drop_rate,conv_type = 'conv', activation = 'relu'):
        super(DenseBlock, self).__init__()
        
        layer0 = DenseLayer

        for i in range(num_layers):
            layer =layer0(
                inplanes + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                activation = activation,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, x):
        features = [x]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


