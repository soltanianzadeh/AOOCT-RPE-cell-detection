# -*- coding: utf-8 -*-

import cv2
import scipy
import random as py_random
import numpy as np

from scipy.ndimage.filters import gaussian_filter
from typing import Optional, Sequence, Union, Any

import albumentations as A

#%% Vessel simulation functions 
def add_vessel(img, th = 0.1, y0 = 0, 
               augment = True, alpha = 1, 
               sigma = 20, alpha_affine = 30,
               Scale = 0.75):
    
    V, V_s = create_vessel(img.shape, th, y0)
    
    if augment:
        V, matrix, map_x, map_y = elastic_transform(
                                                    V,
                                                    alpha,
                                                    sigma,
                                                    alpha_affine,
                                                    interpolation=cv2.INTER_NEAREST)
        ## Apply the same transofmration
        height, width = V_s.shape
        V_s = cv2.warpAffine(V_s, M=matrix, dsize=(width, height), 
                             flags=cv2.INTER_LINEAR, 
                             borderMode=cv2.BORDER_REFLECT_101, 
                             borderValue=None)
        V_s = cv2.remap(V_s, map1=map_x, map2=map_y, 
                        interpolation=cv2.INTER_LINEAR, 
                        borderMode=cv2.BORDER_REFLECT_101, 
                        borderValue=None)
        
    img = img * (1- Scale*V_s)
    
    return img, V

def create_vessel(SZ, th = 0.1, y0 = 0):
    """
    y0 is the y offset of the sine function used to represent vessel.
    """
    x = np.linspace(0,np.pi, SZ[0])
    y = np.linspace(-1,1,SZ[1])
    xx, yy = np.meshgrid(x,y)
    V = (abs(yy-np.cos(xx)+y0) <= th).astype('float32')
    
    # smooth the map and normalize
    V_smooth = scipy.ndimage.gaussian_filter(V, 2)
    V_smooth = (V_smooth - np.min(V_smooth))/(np.max(V_smooth) - np.min(V_smooth))
    
    return V, V_smooth

"""
 Elastic_transform from:
https://github.com/albumentations-team/albumentations/blob/89a675cbfb2b76f6be90e7049cd5211cb08169a5/albumentations/augmentations/geometric/functional.py#L234
"""
NumType = Union[int, float, np.ndarray]
Size = Union[int, Sequence[int]]

def get_random_state() -> np.random.RandomState:
    return np.random.RandomState(py_random.randint(0, (1 << 32) - 1))

def rand(d0: NumType, d1: NumType, *more, random_state: Optional[np.random.RandomState] = None, **kwargs) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.randn(d0, d1, *more, **kwargs)  # type: ignore

def uniform(
    low: NumType = 0.0,
    high: NumType = 1.0,
    size: Optional[Size] = None,
    random_state: Optional[np.random.RandomState] = None,
) -> Any:
    if random_state is None:
        random_state = get_random_state()
    return random_state.uniform(low, high, size)

def elastic_transform(
    img,
    alpha,
    sigma,
    alpha_affine,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
    random_state=None,
    approximate=False,
    same_dxdy=False,
):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/ernestum/601cdf56d2b424757de5
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    height, width = img.shape

    # Random affine
    center_square = np.float32((height, width)) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + uniform(-alpha_affine, alpha_affine, size=pts1.shape, random_state=random_state).astype(
        np.float32
    )
    matrix = cv2.getAffineTransform(pts1, pts2)
    img = cv2.warpAffine(img, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value)

    dx = np.float32(
        gaussian_filter((rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
    )
    if same_dxdy:
        # Speed up
        dy = dx
    else:
        dy = np.float32(
            gaussian_filter((rand(height, width, random_state=random_state) * 2 - 1), sigma) * alpha
        )

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    img = cv2.remap(img, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value)

    return img, matrix, map_x, map_y
