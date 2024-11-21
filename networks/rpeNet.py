import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.encoders import densenet_enc
from .backbones.densenet.basic_module import DenseBlock


from functools import partial

nonlinearity = partial(F.relu, inplace=True)
activations = {
               'relu': nn.ReLU(inplace=True),
               'leakyrelu': nn.LeakyReLU(inplace=True),
               }

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation = 'relu', upsample = 'conv'):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity if activation=='relu' else activations[activation]

        if upsample =='conv':
            self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 4, stride=2, padding=1, output_padding=0)
        else:
            self.deconv2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity if activation=='relu' else activations[activation]

        self.conv3 = nn.Conv2d(in_channels // 4, out_channels, 1)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nonlinearity if activation=='relu' else activations[activation]

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x
    
    

class Densenet_LinkNet(nn.Module):
    def __init__(self, conv_type = 'conv', activation = 'relu', upsample = 'conv'):
        super(Densenet_LinkNet, self).__init__()
        
        self.conv_type = conv_type
        self.num_classes = 1
        self.growth_rate = 32
        self.drop_rate = 0.1
        block_config = (6,12,24,16)
        self.inplanes= 64
        filters = [64,128,256,512]
        self.enc = densenet_enc()

        self.firstbn = nn.BatchNorm2d(filters[-1])
        self.firstrelu = nonlinearity if activation=='relu' else activations[activation]
        if upsample == 'conv':
            self.firstdeconv = nn.ConvTranspose2d(filters[-1],filters[-1], 3, stride = 2, padding=1, output_padding=1)
        else:
            self.firstdeconv = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.firstdrop = nn.Dropout2d(p=self.drop_rate)    

        self.norm1 = nn.BatchNorm2d(320)
        self.relu1 = nonlinearity  if activation=='relu' else activations[activation]
        if upsample =='conv':
            self.deconv1 = nn.ConvTranspose2d(320, filters[0], 3, stride = 2, padding=1, output_padding=1)
        else:
            self.deconv1 = nn.Sequential(
                                   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                   nn.Conv2d(320,filters[0], 3, padding = 1)
                            )
        self.drop1 = nn.Dropout2d(p=self.drop_rate)
        self.decoder1 = self._make_layers(filters[1], block_config[0])

        self.norm2 = nn.BatchNorm2d(640)
        self.relu2 = nonlinearity if activation=='relu' else activations[activation]
        if upsample =='conv':
            self.deconv2 = nn.ConvTranspose2d(640,filters[1], 3, stride = 2, padding=1, output_padding=1)
        else:
            self.deconv2 = nn.Sequential(
                                   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                   nn.Conv2d(640,filters[1], 3, padding = 1)
                            )
        self.drop2 = nn.Dropout2d(p=self.drop_rate)       
        self.decoder2 = self._make_layers(filters[2],block_config[1])

        self.norm3 = nn.BatchNorm2d(1280)
        self.relu3 = nonlinearity if activation=='relu' else activations[activation]
        if upsample == 'conv':
            self.deconv3 = nn.ConvTranspose2d(1280,filters[2], 3, stride = 2, padding=1, output_padding=1)
        else:
            self.deconv3 = nn.Sequential(
                                   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                   nn.Conv2d(1280,filters[2], 3, padding = 1)
                            )
        self.drop3 = nn.Dropout2d(p=self.drop_rate)        
        self.decoder3 = self._make_layers(filters[3], block_config[2])
     
        self.finalbn =  nn.BatchNorm2d(filters[0])
        self.finalrelu = nonlinearity if activation=='relu' else activations[activation]
        self.finaldeconv = nn.Sequential(
                                   nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                                   nn.Conv2d(filters[0], self.num_classes, 3, padding = 1)
                            )                
        


    def _make_layers(self, inplanes, blocks):
        layers = []

        layers.append(DenseBlock(inplanes, blocks, 1, self.growth_rate, self.drop_rate, self.conv_type))

                
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder
        e = self.enc(x.contiguous())

        # Decoder       
        d4 = self.firstdrop(self.firstdeconv(self.firstrelu(self.firstbn(e[0])))) + e[1]
        
        d3 = self.decoder3(d4)
        d3 = self.drop3(self.deconv3(self.relu3(self.norm3(d3)))) + e[2]
        
        d2 = self.decoder2(d3)
        d2 = self.drop2(self.deconv2(self.relu2(self.norm2(d2)))) + e[3]
        
        d1 = self.decoder1(d2)
        d1 = self.drop1(self.deconv1(self.relu1(self.norm1(d1)))) + e[4]
        
        out = self.finalrelu(self.finalbn(d1))
        out = self.finaldeconv(out)

        
        return torch.sigmoid(out)
    
