# -*- coding: utf-8 -*-
"""
Copyright © 2024, Authored by Somayyeh Soltanian-Zadeh.

If you use any part of this code, please cite our work:

S. Soltanian-Zadeh et al., "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy," Biomedical Optics Express, 2024.
          
"""

import torch
import torch.nn as nn
import numbers

from torch.nn import functional as F


### Helper Gaussian smoothing
class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-((mgrid - mean) / std) ** 2 / 2)


        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel.type(torch.cuda.FloatTensor))
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding='same')
    
#%% centroid regression losses
class regLoss_d(nn.Module):
    __name__ = 'regression_loss'
    def __init__(self, type = 'mse', sigma = -1):
        super(regLoss_d, self).__init__()  
        self.loss = regression_loss(type)
        self.sigma = sigma
        # sigma<=0 means do nothing
        # else y_pred will be smoothed with gaussian filtering

    def __call__(self, y_pred, y_true):
        if isinstance(y_pred,dict):
            y_pred = y_pred['center']
        if self.sigma>0:
            y_pred = GaussianSmoothing(1,5,self.sigma)(y_pred)
        return self.loss(y_true['center'].float(), y_pred.squeeze(dim=1).float())  

class regression_loss(nn.Module):
    def __init__(self, type = 'mse'):
        super(regression_loss, self).__init__()
        if type=='mse' or type =='l2':
            self.reg_loss = nn.MSELoss()
        elif type =='l1':
            self.reg_loss = nn.L1Loss()
        else:
            raise Exception("Loss type not supported")
        
    def __call__(self, y_true, y_pred):
        return self.reg_loss(y_pred, y_true)
        

