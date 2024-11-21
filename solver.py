# -*- coding: utf-8 -*-
"""
Copyright © 2024, Authored by Somayyeh Soltanian-Zadeh.

If you use any part of this code, please cite our work:

S. Soltanian-Zadeh et al., "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy," Biomedical Optics Express, 2024.
          
"""

import os
import torch
import imageio
import math
import numpy as np
import scipy.io as sio
from skimage.transform import resize
from scipy.ndimage import zoom

from torchtnt.utils.loggers import TensorBoardLogger
from torch.autograd import Variable


from utils.train_utils import TrainEpoch, ValidEpoch
from utils.utils_cell import localize_cells, QuantifyMatch
from losses.spatial_losses import regLoss_d as regression_loss


class Solver(object):
    def __init__(self, configer):
        self.configer = configer
        self.optim = torch.optim.Adam
        
    def create_directory(self, savedir, exp_id):
        if not os.path.exists(os.path.join(savedir, str(exp_id))):
            os.makedirs(os.path.join(savedir, str(exp_id)))

        #csv = 'results_exp'+str(exp_id)+'.csv'
        #with open(os.path.join(savedir, csv), 'w') as f:
        #    f.write('epoch, total_loss, mse, iou, recall, precision, F1 \n')        
        

    def train(self, model, train_loader, val_loader):
        print('.... training ...')
        
        saveDir = os.path.join(self.configer.get('network','model_dir'))
        self.create_directory(saveDir, self.configer.get('fold'))
        
        optim = self.optim([dict(params=model.parameters(), lr = self.configer.get('lr','base_lr'))])

        self.loss_func = regression_loss(self.configer.get('loss','loss_name'))
        #metrics = [ M.Accuracy(threshold = 0.5)]


        lossname = self.loss_func.__name__

        train_epoch = TrainEpoch(
                            model, loss = self.loss_func, 
                            optimizer = optim,
                            device = 'cuda',
                            verbose = True,
                        )            
        valid_epoch = ValidEpoch(
                            model, loss = self.loss_func, 
                            device = 'cuda',
                            verbose = True,
                        )
        
        # train model for n epochs
        logger = TensorBoardLogger(os.path.join(self.configer.get('logging','log_file'),'train'))
        logger_val = TensorBoardLogger(os.path.join(self.configer.get('logging','log_file'),'val'))
        max_score = 1e3
        
        for i in range(0, self.configer.get('solver','max_epoch')):
            
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader, i+1)
            valid_logs = valid_epoch.run(val_loader, i+1)
            
            # save logs
            logger.log_dict(train_logs, i+1)
            logger_val.log_dict(valid_logs, i+1)

            # Save the model when loss value improves
            if max_score > valid_logs[lossname]:
                max_score = valid_logs[lossname]
                torch.save(model.state_dict(), saveDir+'//best_model.pth')
                print('Best model saved!')
                print('Best validation score: ', str(max_score))
                
            if (i+1) % self.configer.get('solver','save-per-epochs') == 0:
                torch.save(model.state_dict(), saveDir + '//'+str(i+1)+'_model.pth') 
                
        print('... Finished training ...')
        logger.close()
        
    def test(self, model, loader, borderDist = None, maxDist = 5, peakThresh = 0.1, vessel_dir = None, augmentation = None, imgScale = 1, nout = 1):
        if borderDist is None:
            borderDist = 5 # pixels 
        
        if isinstance(borderDist,int):
            borderDist = (borderDist,borderDist)

        # Read previously saved vessel masks
        if vessel_dir: vFiles = os.listdir(vessel_dir)
        
        model.eval()
        pred_out = {'center_map': np.empty((len(loader),), dtype= object),
                    'gt_centers': np.empty((len(loader),), dtype= object),
                    'pred_centers': np.empty((len(loader),), dtype= object),
                    'names':[],
                    'vessel_map':np.empty((len(loader),), dtype= object)}
        
        scores = {'names': [], 'recall': [], 'precision': [], 'matchInfo': []}
      
        with torch.no_grad():
            for j, tst_data in enumerate(loader):
                img = tst_data[0]
                SZ = np.shape(img.squeeze())
                centers_gt = tst_data[2].squeeze()
                nameSubj = tst_data[-2][0]
                Name = tst_data[-1][0]

                # Scale image by factor imgScale
                img = torch.Tensor(zoom(img,(1,1,imgScale,imgScale)).copy())
        
                # pad image if size not divisible by 32
                DIV = 32
                SZ0 = img.size()[2:]
                pad_right = math.ceil((DIV*(math.ceil(SZ0[1]/DIV))-SZ0[1]))
                pad_bottom = math.ceil((DIV*(math.ceil(SZ0[0]/DIV))-SZ0[0]))
                img0 = torch.nn.functional.pad(img, 
                           (0,pad_right,0,pad_bottom), mode = 'reflect')

                if augmentation:
                    pred = np.zeros(SZ+(len(augmentation),)) if nout==1 else np.zeros((nout,)+SZ+(len(augmentation),))
                    for nt,t in enumerate(augmentation):
                        if t.__name__ != 'identity': 
                            out = t({'img': img0.squeeze(), 'loc':np.zeros((1,2))})
                            img_in = out['img']
                            img_in = torch.Tensor(img_in.copy()).unsqueeze(0).unsqueeze(0)
                        else:
                            img_in = img0

                        # pass thorugh network
                        pred_model = model(Variable(img_in).cuda())      
            
                        if isinstance(pred_model,dict):
                            pred_model= pred_model['center']
                    
                        
                        pred_model = pred_model.squeeze().cpu().numpy()
                        
                        if pred_model.ndim ==3:
                            if t.__name__ in ['hflip','vflip']:
                                for irange in range(nout):
                                    pred_model[irange,:] = np.flipud(pred_model[irange,:]) if t.__name__ == 'hflip' else np.fliplr(pred_model[irange,:])
                            # Crop and Scale back
                            if pad_right>0: pred_model = pred_model[:,:,:-pad_right]
                            if pad_bottom>0: pred_model = pred_model[:,:-pad_bottom,:]
                            pred_model = resize(pred_model ,(3,)+SZ)
                        else:

                            # Redo augmentation (if not intensity)
                            if t.__name__ in ['hflip','vflip']:
                                pred_model = np.flipud(pred_model) if t.__name__ == 'hflip' else np.fliplr(pred_model)
                            
                            # Crop and Scale back
                            if pad_right>0: pred_model = pred_model[:,:-pad_right]
                            if pad_bottom>0: pred_model = pred_model[:-pad_bottom,:]
                            pred_model = resize(pred_model, SZ)

                        pred[...,nt]= pred_model

                    pred = np.mean(pred,axis=-1)
                else:
                    # pass thorugh network
                    pred = model(Variable(img0).cuda())      
        
                    if isinstance(pred,dict):
                        pred = pred['center']

                    pred = pred.squeeze().cpu().numpy() 
                    pred = pred[:SZ[0],:SZ[1]]                


                # local-maxima detection on network output
                centers_pred = localize_cells(pred,peakThresh)                             


                #save predictions results         
                pred_out['center_map'][j] = pred
                pred_out['pred_centers'][j] = centers_pred
                
                pred_out['names'].append(Name)

                if len(centers_pred):
                    if vessel_dir:

                        ind = [i for i,subs in enumerate(vFiles) if Name in subs]
                        if len(ind) !=1:
                            ind = [i for i,subs in enumerate(vFiles) if nameSubj in subs]
                          
                        vMap = sio.loadmat(vessel_dir+vFiles[ind[0]])['vessel_map']
                        vMap = vMap[:SZ[0],:SZ[1]]
                    else:
                        vMap  = []

                    recall, precision, stat = QuantifyMatch(centers_pred, 
                                                            centers_gt, 
                                                            SZ, 
                                                            borderDist, 
                                                            invalidMask = vMap, 
                                                            maxDist = maxDist)

                else:
                    print('no cells found')
                    recall, precision = 0., 0.
                    stat = {}
                    vMap = []

                pred_out['gt_centers'][j] = centers_gt.numpy()
                scores['names'].append(Name)
                scores['recall'].append(recall)
                scores['precision'].append(precision)
                scores['matchInfo'].append(stat)
                pred_out['vessel_map'][j] = vMap
        
        return pred_out, scores


    def save_fig(self, pred, gt, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        
        gt, pred = gt*255, pred*255
        imageio.imwrite(path+'/'+name+'_gt.png', gt.cpu().numpy().astype(np.uint8))
        imageio.imwrite(path+'/'+name+'_pred.png', pred.cpu().numpy().astype(np.uint8))

        