import torch
import glob
import os
import numpy as np

from torch.utils.data import Dataset as BaseDataset
from PIL import Image
from pathlib import Path

from vessel_module import add_vessel



#### For vessel segmentation task
class Dataset(BaseDataset):
    
    def __init__(
            self, 
            data_dir, 
            augment = True,
            add_vessel = True,
            resize = (160,160),
            keep_last = False,
            ):
        
        self.totensor = ToTensor()
        self.augment = augment
        self.add_vessel = add_vessel
        self.H, self.W = resize
        self.data_fps = []
        for dir_i in data_dir:
            ids = glob.glob(os.path.join(dir_i,'*.tif'))
            if keep_last: ids = [ids[-1]]
            self.data_fps += [data_id for data_id in ids]
            
    def __getitem__(self, i):
        
        # read data
        image = Image.open(self.data_fps[i])
        image = np.array(image).astype(np.float64)
        image =  (image-image.min())/(image.max()-image.min())

        # crop image if larger than window size
        H, W = np.shape(image)
        if H>self.H: image = image[:self.H,:]
        if W>self.W: image = image[:,:self.W]


        # pad images if smaller than window size        
        postH = max(0,self.H-H)
        postW = max(0,self.W-W)
        image = np.pad(image, ((0,postH),(0, postW)), mode='symmetric')

        # parameters
        if self.augment:
            th, y0 = np.random.uniform(0.15,0.3), np.random.uniform(-1,1)
            Scale = 1-th
        else:
            th, y0, Scale = 0.25, 0.3, 0.6

        if self.add_vessel:
            image, label = add_vessel(image,
                                  th = th, y0 = y0, 
                                  augment = self.augment, alpha = 1, 
                                  sigma = 20, alpha_affine = 30,
                                  Scale = Scale)
        else:
            label = image.copy()
        
        image, label = self.totensor(image), self.totensor(label)
       
        image = image.expand(3,image.shape[0],image.shape[1])
        FolderName = os.path.basename(os.path.dirname(self.data_fps[i]))
        Filename = Path(self.data_fps[i]).stem
        return image, label.unsqueeze(0), [FolderName,Filename]
        
    def __len__(self):
        return len(self.data_fps)
    
    
class ToTensor(object):
    def __call__(self, data):
        return torch.Tensor(data.copy())
    