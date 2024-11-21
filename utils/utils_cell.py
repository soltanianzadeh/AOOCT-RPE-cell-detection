"""
Copyright © 2024, Authored by Somayyeh Soltanian-Zadeh.

If you use any part of this code, please cite our work:

S. Soltanian-Zadeh et al., "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy," Biomedical Optics Express, 2024.
          
"""

import numpy as np

from skimage import measure
from regional import many
from skimage.feature import peak_local_max

from utils.matcher import build_matcher_cells


Hmatcher = build_matcher_cells()

def QuantifyMatch(centers_pred, centers_gt, imgSize, borderDist, invalidMask = [], maxDist = 5):
    indices, _ = Hmatcher.match({'points':centers_pred}, {'points': centers_gt},
                    maxDist = maxDist)  # in pixels

    mask = (centers_gt[...,0]>borderDist[1]) & (centers_gt[...,0]<imgSize[1]-borderDist[1]) & (
            centers_gt[...,1]>borderDist[0]) & (centers_gt[...,1]<imgSize[0]-borderDist[0])

    mask_pred = (centers_pred[...,0]>borderDist[1]) & (centers_pred[...,0]<imgSize[1]-borderDist[1]) & (
            centers_pred[...,1]>borderDist[0]) & (centers_pred[...,1]<imgSize[0]-borderDist[0])

    gt_borders = [idx for idx,v in enumerate(mask) if not v]
    pred_borders = [idx for idx,v in enumerate(mask_pred) if not v]

    # Remove cells overlapping invalid pixels
    if len(invalidMask):        
        mask = [invalidMask[int(centers_gt[i,1]),int(centers_gt[i,0])] for i in range(len(centers_gt))]
        gt_borders += [idx for idx,v in enumerate(mask) if v==1]
        mask_pred = [invalidMask[int(centers_pred[i,1]),int(centers_pred[i,0])] for i in range(len(centers_pred))]
        pred_borders += [idx for idx,v in enumerate(mask_pred) if v==1]


    matched_border_gt = [idx for idx in range(len(indices[0])) if indices[0][idx] in pred_borders]
    ind_remove_gt = sorted(set(gt_borders)|set(indices[1][matched_border_gt]))

    matched_border = [idx for idx in range(len(indices[1])) if indices[1][idx] in gt_borders]
    ind_remove_pred = sorted(set(pred_borders)|set(indices[0][matched_border]))

    indices[1] = np.delete(indices[1],sorted(set(matched_border_gt)|set(matched_border)) )
    indices[0] = np.delete(indices[0],sorted(set(matched_border)|set(matched_border_gt)) )

    n_gt = len(centers_gt) - len(ind_remove_gt)
    n_pred = len(centers_pred) - len(ind_remove_pred)

    r = (len(indices[0]))/n_gt      #Recall
    p = (len(indices[0]))/n_pred if n_pred>0 else -1.0 #Precision

    stat = {'gt_match_ind': indices[1], 'pred_match_ind': indices[0],
            'gt_border_inds': ind_remove_gt, 'pred_border_inds': ind_remove_pred}
    return r, p, stat


def localize_cells(map, thresh, exclude_border = 5):
    # local-maxima detection on network output
    Lmaxima = peak_local_max(map, 
                            min_distance = 3, threshold_rel = thresh,
                            indices=False, exclude_border= exclude_border)
    centers_pred = _mask_to_regional(Lmaxima)  

    if not isinstance(centers_pred,list):    
                    centers_pred = np.fliplr(centers_pred.center) 

    return centers_pred 


def _mask_to_regional(m, scale = 1, tag_z = True):
    mlbl = measure.label(m)
    coords = []
    
    for lbl in range(1, np.max(mlbl) + 1):
        if len(mlbl.shape)==2 or tag_z == False:
            if len(mlbl.shape)==2:
                yy, xx = np.where(mlbl == lbl)
                coords.append([[y*scale, x*scale] for y, x in zip(yy, xx)])
            else:
                assert len(mlbl.shape)==3
                yy, xx,zz = np.where(mlbl == lbl)
                coords.append([[y*scale, x*scale, z] for y, x,z in zip(yy, xx,zz)])
        else:
            yy, xx,zz = np.where(mlbl == lbl)
            coords.append([[y*scale, x*scale, z*scale] for y, x,z in zip(yy, xx,zz)])            
    
    return many(coords) if len(coords) else []

