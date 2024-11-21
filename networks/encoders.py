
import torch.nn as nn
import torch.nn.functional as F

from .backbones.densenet.densenet_factory import get_densenet_backbone


from functools import partial

nonlinearity = partial(F.relu, inplace=True)

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x


class densenet_enc(nn.Module):
    def __init__(self,num_channels=1, conv_type = 'conv'):
        super(densenet_enc, self).__init__()
      
        self.growth_rate = 32
        self.drop_rate = 0.1
        self.inplanes= 64
        densenet = get_densenet_backbone('densenet121')(channel_in = num_channels, conv_type = conv_type)
        self.firstconv = densenet.conv1
 
        self.encoder1 = densenet.layer1
        self.encoder2 = densenet.layer2
        self.encoder3 = densenet.layer3
        self.encoder4 = densenet.layer4
    
    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
      
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        return [e4, e3, e2, e1, x]
