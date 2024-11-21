# -*- coding: utf-8 -*-
"""
Copyright © 2024, Authored by Somayyeh Soltanian-Zadeh.

If you use any part of this code, please cite our work:

S. Soltanian-Zadeh et al., "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy," Biomedical Optics Express, 2024.
          
"""

import torch
import glob
import os
import random
import numpy as np
import cv2
import albumentations as albu


from pathlib import Path
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.fft import ifft2, fft2

import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import v2



def density_map(loc, imgSize, sigma):
    map = np.zeros(imgSize)
    loc = loc.astype(np.uint8)  
    map[loc[:,1],loc[:,0]] = 1
    if sigma>0:
        map = gaussian_filter(map, sigma)
    return map

class Collate_F():
    @staticmethod
    def train_collate(batch):
        batch = list(zip(*batch))
        images = torch.stack(batch[0], 0)
        chfs = torch.stack(batch[1], 0)
        return images, chfs

class RawDataset(BaseDataset):
    def __init__(
            self, 
            data_dir, 
            sigma = 1.5,
            crop_size = 128,
            augmentation=None, 
            mode = 'train',
            loss_type = 'l1'
            ):
        self.totensor = ToTensor()
        self.data_fps = []
        self.sigma = sigma
        self.crop_size = crop_size
        self.mode = mode
        self.loss_type = loss_type
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

        assert self.loss_type in ['l1','l2','mse']

        for dir_i in data_dir:
            ids = glob.glob(os.path.join(dir_i,'*.tif'))
            self.data_fps += [data_id for data_id in ids]
        
        self.augmentation  = augmentation
    
    def __getitem__(self, i):
        
        # read data: Image and cell coordinates
        image = Image.open(self.data_fps[i])
        image = np.array(image).astype(np.float64)
        image =  (image-image.min())/(image.max()-image.min())

        if os.path.isfile(self.data_fps[i][:-4]+'_cellCenter.mat'):
            points = loadmat(self.data_fps[i][:-4]+'_cellCenter.mat')
        else:
            cellfile = glob.glob(os.path.join(os.path.dirname(self.data_fps[i]),'*_cellCenter.mat'))
            points = loadmat(cellfile[0])

        loc = points['cellXY'] - 1 #matlab to python coordinates
        

        if self.crop_size:
            # make sure image is larger than crop_size
            image = np.pad(image, 
                           ((0,max(0,self.crop_size-image.shape[0])),(0,max(0,self.crop_size-image.shape[1]))), 
                           mode = 'reflect')
            # now crop
            x = random.randint(0, image.shape[1] - self.crop_size)
            y = random.randint(0, image.shape[0] - self.crop_size)
            image = image[y:y+self.crop_size, x:x+self.crop_size]
            mask = (loc[:,0]>x) & (loc[:,0]<x+self.crop_size) & (
                loc[:,1]>y) & (loc[:,1]<y+self.crop_size)
            loc = loc[mask]
            loc = loc - [x,y]
            
        # apply augmentations
        if self.augmentation:
            out = self.augmentation({'img':image, 'loc':loc})
            image, loc = out['img'], out['loc']

        image = self.totensor(image)

        cen_mask = density_map(loc,np.shape(image), self.sigma)
        cen_mask = self.totensor((cen_mask-cen_mask.min())/(cen_mask.max()-cen_mask.min()) )
        if self.mode == 'train':
            return image.unsqueeze(0), {'center':cen_mask}
        else:
            FolderName = Path(os.path.dirname(self.data_fps[i])).stem
            return image.unsqueeze(0), {'center':cen_mask}, loc, FolderName, Path(self.data_fps[i]).stem
        
              

        
    def __len__(self):
        return len(self.data_fps)
                

    
class ToTensor(object):
    def __call__(self, data):
        return torch.Tensor(data.copy())


### Other utility functions
def sliding_window(image, stepSize, windowSize, op = None):
  outImage = np.zeros(image.shape)
  # pad image for calculations at border areas
  padY = (int(windowSize[1]/2),int(windowSize[1]/2))
  padX = (int(windowSize[0]/2),int(windowSize[0]/2))
  image = np.pad(image,(padY,padX),'symmetric')
  for y in range(0, outImage.shape[0], stepSize):
    for x in range(0, outImage.shape[1], stepSize):
          outImage[y,x] = extractFeatures(image[y+padY[0]:y +padY[0]+ windowSize[1], x+padX[0]:x+padX[0] + windowSize[0]], op)
  
  return outImage

def extractFeatures(data, op):

        assert op in ['min','max','mean']
        if op =='min':
            return data.min()
        if op == 'max':
            return data.max()
        if op == 'mean':
            return np.mean(data)
        


#%% Augmentation
def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),      
    ]   
    return albu.Compose(train_transform)    
  

def get_RawTest_aug():
    t = [Identity(),
         RandomHFlip(0),
         RandomVFlip(0)]

    return t    


def get_RawTraining_aug(intensity_aug = 'None'):
    int_aug_options = {'clahe': [ClaheAug(p=0.3)],
                       'fourier': [FourierAug(p=0.5, limits = (0.5, 1))],
                       'both': [ FourierAug(p=0.3, limits = (0.3, 1))] if np.random.rand(1)>0.5 else [ClaheAug(p=0.6)]
                       }
    t = [RandomHFlip(0.3),
        RandomVFlip(0.3),
        RandomTranspose(0.3)]
    if intensity_aug != 'None':
        assert intensity_aug in ['clahe','fourier','both']

        transforms = v2.Compose( t + int_aug_options[intensity_aug])
        return transforms
    else:
        return v2.Compose(t)

class Identity(torch.nn.Module):
    __name__ = 'identity'
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input):
        return input
    
class RandomVFlip(torch.nn.Module):
    __name__ = 'vflip'
    def __init__(self, p = 0.5):
        super(RandomVFlip, self).__init__()
        self.p = p

    def forward(self, input):
        img = input['img']
        loc = input['loc']
        w, h = np.shape(img)
        if np.random.rand(1) >self.p:
            img = np.fliplr(img)
            if loc.shape[0] > 0:
                loc[:, 0] = w - loc[:, 0]       
        return {'img':img, 'loc':loc} 
    
class RandomHFlip(torch.nn.Module):
    __name__ = 'hflip'
    def __init__(self, p = 0.5):
        super(RandomHFlip, self).__init__()
        self.p = p

    def forward(self, input):
        img = input['img']
        loc = input['loc']
        w, h = np.shape(img)
        if np.random.rand(1) > self.p:
            img = np.flipud(img)
            if loc.shape[0] > 0:
                loc[:, 1] = h - loc[:, 1]       
        return {'img':img, 'loc':loc}     
    
class RandomTranspose(torch.nn.Module):
    __name__ = 'transpose'
    def __init__(self, p = 0.5):
        super(RandomTranspose, self).__init__()
        self.p = p

    def forward(self, input):
        img = input['img']
        loc = input['loc']
        if np.random.rand(1) > self.p:
            img = np.transpose(img, [1,0])
            if loc.shape[0] > 0:
                loc=loc[:, [1,0]]       
        return {'img':img, 'loc':loc}   

class ClaheAug(torch.nn.Module):
    __name__ = 'clahe'
    def __init__(self, limit = (1., 3.), p = 0.5):
        super(ClaheAug, self).__init__()
        self.p = p
        self.limit = limit

    def forward(self,input):
        img = input['img']
        loc = input['loc']
        if np.random.rand(1) > self.p:
            clahe = cv2.createCLAHE(clipLimit= np.random.uniform(self.limit[0],self.limit[1]) ,tileGridSize=(4,4))
            img_np = np.array(img)
            img_np = np.uint8(cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX))
            img_np = clahe.apply(img_np.squeeze())
            img_np = (img_np-img_np.min())/(img_np.max()-img_np.min())
        else:
            img_np = np.array(img)
       
        return {'img':img_np, 'loc':loc}
    
class FourierAug(torch.nn.Module):
    # Fourier-based intensity augmentation of image
    def __init__(self, limits = (0.5,1), p =0.5):
        super(FourierAug, self).__init__()
        self.p = p
        self.limits = limits

    def forward(self, input):
        img = input['img']
        if np.random.rand(1)>self.p:
            Fmag, Fangle = abs(fft2(img)), np.angle(fft2(img))
            # augment amplitude by multiplying by randomly uniform noise
            Fmag = np.multiply(Fmag, np.random.uniform(low=self.limits[0], high = self.limits[1],size=np.shape(img)))
            F = np.multiply(Fmag,np.exp(1j*Fangle))
            img = np.real(ifft2(F))
            img = (img-img.min())/(img.max()-img.min())
        return {'img': img, 'loc': input['loc']}
