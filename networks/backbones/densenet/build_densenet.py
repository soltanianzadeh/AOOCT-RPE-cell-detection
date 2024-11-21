

"""
pieces of code from:
https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py#L136
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .basic_module import DenseBlock,TransitionLayer, BN_MOMENTUM

import torch.nn as nn


class build_densenet(nn.Module):
    def __init__(self, 
                 growth_rate = 32,
                 block_config = (6,12,24,16),
                 num_init_features = 64,
                 drop_rate = 0.1,
                 channel_in = 3,
                 conv_type = 'conv'):
        super(build_densenet, self).__init__()
        
        self.conv_type = conv_type
        if conv_type=='conv':
            self.conv1 = nn.Conv2d(channel_in, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            KeyError

        self.bn1 = nn.BatchNorm2d(num_init_features, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inplanes = num_init_features
        self.growth_rate = growth_rate
        self.drop_rate = drop_rate

        # ENCODER
        self.layer1 = self._make_layers(block_config[0])
        self.layer2 = self._make_layers(block_config[1])
        self.layer3 = self._make_layers(block_config[2])
        self.layer4 = nn.Conv2d(self.inplanes, 512, kernel_size = 3, stride = 2, padding=1, bias = False)


    def _make_layers(self, blocks, add_transition = True):
        layers = []
        layers.append(DenseBlock(self.inplanes, blocks, 1, self.growth_rate, self.drop_rate, self.conv_type))
        self.inplanes += self.growth_rate*blocks
        
        if add_transition:
            layers.append(TransitionLayer(self.inplanes))
            self.inplanes = self.inplanes//2
        
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.relu(self.bn2(self.layer4(x)))

        return x


def get_densenet_121(channel_in = 3, conv_type = 'conv'):
    model = build_densenet(channel_in = channel_in, conv_type = conv_type)

    return model


if __name__ == '__main__':
    model = get_densenet_121(channel_in = 3)
    print(model)