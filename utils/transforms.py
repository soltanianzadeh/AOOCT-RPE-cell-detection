"""
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.

Accessed 6/3/2024
Modified functions for flow-independent augmentations (channels are handled the same)
"""

import numpy as np
import warnings
import cv2
import torch
from scipy.ndimage import gaussian_filter1d

import logging

transforms_logger = logging.getLogger(__name__)


def _taper_mask(ly=224, lx=224, sig=7.5):
    """
    Generate a taper mask.

    Args:
        ly (int): The height of the mask. Default is 224.
        lx (int): The width of the mask. Default is 224.
        sig (float): The sigma value for the tapering function. Default is 7.5.

    Returns:
        numpy.ndarray: The taper mask.

    """
    bsize = max(224, max(ly, lx))
    xm = np.arange(bsize)
    xm = np.abs(xm - xm.mean())
    mask = 1 / (1 + np.exp((xm - (bsize / 2 - 20)) / sig))
    mask = mask * mask[:, np.newaxis]
    mask = mask[bsize // 2 - ly // 2:bsize // 2 + ly // 2 + ly % 2,
                bsize // 2 - lx // 2:bsize // 2 + lx // 2 + lx % 2]
    return mask


def unaugment_tiles(y):
    """Reverse test-time augmentations for averaging (includes flipping of flowsY and flowsX).

    Args:
        y (float32): Array of shape (ntiles_y, ntiles_x, chan, Ly, Lx) 

    Returns:
        float32: Array of shape (ntiles_y, ntiles_x, chan, Ly, Lx).

    """
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j % 2 == 0 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, :]
                
            elif j % 2 == 1 and i % 2 == 0:
                y[j, i] = y[j, i, :, :, ::-1]
                
                
            elif j % 2 == 1 and i % 2 == 1:
                y[j, i] = y[j, i, :, ::-1, ::-1]

    return y


def average_tiles(y, ysub, xsub, Ly, Lx):
    """
    Average the results of the network over tiles.

    Args:
        y (float): Output of cellpose network for each tile. Shape: [ntiles x nclasses x bsize x bsize]
        ysub (list): List of arrays with start and end of tiles in Y of length ntiles
        xsub (list): List of arrays with start and end of tiles in X of length ntiles
        Ly (int): Size of pre-tiled image in Y (may be larger than original image if image size is less than bsize)
        Lx (int): Size of pre-tiled image in X (may be larger than original image if image size is less than bsize)

    Returns:
        yf (float32): Network output averaged over tiles. Shape: [nclasses x Ly x Lx]
    """
    Navg = np.zeros((Ly, Lx))
    yf = np.zeros((y.shape[1], Ly, Lx), np.float32)
    # taper edges of tiles
    mask = _taper_mask(ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0]:ysub[j][1], xsub[j][0]:xsub[j][1]] += mask
    yf /= Navg
    return yf


def make_tiles(imgi, bsize=224, augment=False, tile_overlap=0.1):
    """Make tiles of image to run at test-time.

    Args:
        imgi (np.ndarray): Array of shape (nchan, Ly, Lx) representing the input image.
        bsize (int, optional): Size of tiles. Defaults to 224.
        augment (bool, optional): Whether to flip tiles and set tile_overlap=2. Defaults to False.
        tile_overlap (float, optional): Fraction of overlap of tiles. Defaults to 0.1.

    Returns:
        tuple containing
            - IMG (np.ndarray): Array of shape (ntiles, nchan, bsize, bsize) representing the tiles.
            - ysub (list): List of arrays with start and end of tiles in Y of length ntiles.
            - xsub (list): List of arrays with start and end of tiles in X of length ntiles.
            - Ly (int): Height of the input image.
            - Lx (int): Width of the input image.
    """
    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = np.int32(bsize)
        # pad if image smaller than bsize
        if Ly < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, bsize - Ly, Lx))), axis=1)
            Ly = bsize
        if Lx < bsize:
            imgi = np.concatenate((imgi, np.zeros((nchan, Ly, bsize - Lx))), axis=2)
        Ly, Lx = imgi.shape[-2:]
        # tiles overlap by half of tile size
        ny = max(2, int(np.ceil(2. * Ly / bsize)))
        nx = max(2, int(np.ceil(2. * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)

        ysub = []
        xsub = []

        # flip tiles so that overlapping segments are processed in rotation
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]
                # flip tiles to allow for augmentation of overlapping segments
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, :]
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = IMG[j, i, :, :, ::-1]
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = IMG[j, i, :, ::-1, ::-1]
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = min(bsize, Ly), min(bsize, Lx)
        bsizeY = np.int32(bsizeY)
        bsizeX = np.int32(bsizeX)
        # tiles overlap by 10% tile size
        ny = 1 if Ly <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1. + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)

        ysub = []
        xsub = []
        IMG = np.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[:, ysub[-1][0]:ysub[-1][1], xsub[-1][0]:xsub[-1][1]]

    return IMG, ysub, xsub, Ly, Lx


def normalize99(Y, lower=1, upper=99, copy=True):
    """
    Normalize the image so that 0.0 corresponds to the 1st percentile and 1.0 corresponds to the 99th percentile.

    Args:
        Y (ndarray): The input image.
        lower (int, optional): The lower percentile. Defaults to 1.
        upper (int, optional): The upper percentile. Defaults to 99.
        copy (bool, optional): Whether to create a copy of the input image. Defaults to True.

    Returns:
        ndarray: The normalized image.
    """
    X = Y.copy() if copy else Y
    x01 = np.percentile(X, lower)
    x99 = np.percentile(X, upper)
    if x99 - x01 > 1e-3:
        X = (X - x01) / (x99 - x01)
    else:
        X[:] = 0
    return X


def normalize99_tile(img, blocksize=100, lower=1., upper=99., tile_overlap=0.1,
                     norm3D=False, smooth3D=1, is3D=False):
    """Compute normalization like normalize99 function but in tiles.

    Args:
        img (numpy.ndarray): Array of shape (Lz x) Ly x Lx (x nchan) containing the image.
        blocksize (float, optional): Size of tiles. Defaults to 100.
        lower (float, optional): Lower percentile for normalization. Defaults to 1.0.
        upper (float, optional): Upper percentile for normalization. Defaults to 99.0.
        tile_overlap (float, optional): Fraction of overlap of tiles. Defaults to 0.1.
        norm3D (bool, optional): Use same tiled normalization for each z-plane. Defaults to False.
        smooth3D (int, optional): Smoothing factor for 3D normalization. Defaults to 1.
        is3D (bool, optional): Set to True if image is a 3D stack. Defaults to False.

    Returns:
        numpy.ndarray: Normalized image array of shape (Lz x) Ly x Lx (x nchan).
    """
    shape = img.shape
    is1c = True if img.ndim == 2 or (is3D and img.ndim == 3) else False
    is3D = True if img.ndim > 3 or (is3D and img.ndim == 3) else False
    img = img[..., np.newaxis] if is1c else img
    img = img[np.newaxis, ...] if img.ndim == 3 else img
    Lz, Ly, Lx, nchan = img.shape

    tile_overlap = min(0.5, max(0.05, tile_overlap))
    blocksizeY, blocksizeX = min(blocksize, Ly), min(blocksize, Lx)
    blocksizeY = np.int32(blocksizeY)
    blocksizeX = np.int32(blocksizeX)
    # tiles overlap by 10% tile size
    ny = 1 if Ly <= blocksize else int(np.ceil(
        (1. + 2 * tile_overlap) * Ly / blocksize))
    nx = 1 if Lx <= blocksize else int(np.ceil(
        (1. + 2 * tile_overlap) * Lx / blocksize))
    ystart = np.linspace(0, Ly - blocksizeY, ny).astype(int)
    xstart = np.linspace(0, Lx - blocksizeX, nx).astype(int)
    ysub = []
    xsub = []
    for j in range(len(ystart)):
        for i in range(len(xstart)):
            ysub.append([ystart[j], ystart[j] + blocksizeY])
            xsub.append([xstart[i], xstart[i] + blocksizeX])

    x01_tiles_z = []
    x99_tiles_z = []
    for z in range(Lz):
        IMG = np.zeros((len(ystart), len(xstart), blocksizeY, blocksizeX, nchan),
                       "float32")
        k = 0
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                IMG[j, i] = img[z, ysub[k][0]:ysub[k][1], xsub[k][0]:xsub[k][1], :]
                k += 1
        x01_tiles = np.percentile(IMG, lower, axis=(-3, -2))
        x99_tiles = np.percentile(IMG, upper, axis=(-3, -2))

        # fill areas with small differences with neighboring squares
        to_fill = np.zeros(x01_tiles.shape[:2], "bool")
        for c in range(nchan):
            to_fill = x99_tiles[:, :, c] - x01_tiles[:, :, c] < +1e-3
            if to_fill.sum() > 0 and to_fill.sum() < x99_tiles[:, :, c].size:
                fill_vals = np.nonzero(to_fill)
                fill_neigh = np.nonzero(~to_fill)
                nearest_neigh = (
                    (fill_vals[0] - fill_neigh[0][:, np.newaxis])**2 +
                    (fill_vals[1] - fill_neigh[1][:, np.newaxis])**2).argmin(axis=0)
                x01_tiles[fill_vals[0], fill_vals[1],
                          c] = x01_tiles[fill_neigh[0][nearest_neigh],
                                         fill_neigh[1][nearest_neigh], c]
                x99_tiles[fill_vals[0], fill_vals[1],
                          c] = x99_tiles[fill_neigh[0][nearest_neigh],
                                         fill_neigh[1][nearest_neigh], c]
            elif to_fill.sum() > 0 and to_fill.sum() == x99_tiles[:, :, c].size:
                x01_tiles[:, :, c] = 0
                x99_tiles[:, :, c] = 1
        x01_tiles_z.append(x01_tiles)
        x99_tiles_z.append(x99_tiles)

    x01_tiles_z = np.array(x01_tiles_z)
    x99_tiles_z = np.array(x99_tiles_z)
    # do not smooth over z-axis if not normalizing separately per plane
    for a in range(2):
        x01_tiles_z = gaussian_filter1d(x01_tiles_z, 1, axis=a)
        x99_tiles_z = gaussian_filter1d(x99_tiles_z, 1, axis=a)
    if norm3D:
        smooth3D = 1 if smooth3D == 0 else smooth3D
        x01_tiles_z = gaussian_filter1d(x01_tiles_z, smooth3D, axis=a)
        x99_tiles_z = gaussian_filter1d(x99_tiles_z, smooth3D, axis=a)

    if not norm3D and Lz > 1:
        x01 = np.zeros((len(x01_tiles_z), Ly, Lx, nchan), "float32")
        x99 = np.zeros((len(x01_tiles_z), Ly, Lx, nchan), "float32")
        for z in range(Lz):
            x01_rsz = cv2.resize(x01_tiles_z[z], (Lx, Ly),
                                 interpolation=cv2.INTER_LINEAR)
            x01[z] = x01_rsz[..., np.newaxis] if nchan == 1 else x01_rsz
            x99_rsz = cv2.resize(x99_tiles_z[z], (Lx, Ly),
                                 interpolation=cv2.INTER_LINEAR)
            x99[z] = x99_rsz[..., np.newaxis] if nchan == 1 else x01_rsz
        if (x99 - x01).min() < 1e-3:
            raise ZeroDivisionError(
                "cannot use norm3D=False with tile_norm, sample is too sparse; set norm3D=True or tile_norm=0"
            )
    else:
        x01 = cv2.resize(x01_tiles_z.mean(axis=0), (Lx, Ly),
                         interpolation=cv2.INTER_LINEAR)
        x99 = cv2.resize(x99_tiles_z.mean(axis=0), (Lx, Ly),
                         interpolation=cv2.INTER_LINEAR)
        if x01.ndim < 3:
            x01 = x01[..., np.newaxis]
            x99 = x99[..., np.newaxis]

    if is1c:
        img, x01, x99 = img.squeeze(), x01.squeeze(), x99.squeeze()
    elif not is3D:
        img, x01, x99 = img[0], x01[0], x99[0]

    return (img - x01) / (x99 - x01)


def gaussian_kernel(sigma, Ly, Lx, device=torch.device("cpu")):
    """
    Generates a 2D Gaussian kernel.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        Ly (int): Number of pixels in the y-axis.
        Lx (int): Number of pixels in the x-axis.
        device (torch.device, optional): Device to store the kernel tensor. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: 2D Gaussian kernel tensor.

    """
    y = torch.linspace(-Ly / 2, Ly / 2 + 1, Ly, device=device)
    x = torch.linspace(-Ly / 2, Ly / 2 + 1, Lx, device=device)
    y, x = torch.meshgrid(y, x, indexing="ij")
    kernel = torch.exp(-(y**2 + x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel
