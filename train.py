# -*- coding: utf-8 -*-
"""
Copyright © 2024, Authored by Somayyeh Soltanian-Zadeh.

If you use any part of this code, please cite our work:

S. Soltanian-Zadeh et al., "Identifying retinal pigment epithelium cells in adaptive optics–optical coherence tomography images with partial annotations and superhuman accuracy," Biomedical Optics Express, 15(12), 2024.
          
"""

import torch
import argparse
import scipy.io as sio
import numpy as np

from torch.utils.data import DataLoader
import os
thispath = os.getcwd()
import time

from solver import Solver
from networks.rpeNet import Densenet_LinkNet
from utils.dataset import RawDataset, get_RawTraining_aug, get_RawTest_aug
from utils.configer import Configer


torch.cuda.set_device(0) ## GPU id

### GLOBAL Variable: List of folds for training/testing
FOLDS = {
         0:['0571','8195'], 
         1:['1610','0420'],
         2:['2875','7473'],
         3:['5291','3339'],
         4:['5810','0201'],
         5:['7743']
         }

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')
    
def parse_args():
    parser = argparse.ArgumentParser(description='Training RPE cell detection netowrk')
    parser.add_argument('--configs', default= 'config.json', type=str,
                        dest='configs', help='The file of the hyper parameters.')    
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--exp', type = int, default = 0,
                        dest = 'network:experiment',
                        help = 'experiment number for evaluation. Must be provided if only running in evaluation mode.')

    # ****************** dataset ******************
    parser.add_argument('--data-dir', type=str, 
                        default=os.path.join(thispath,'dataset','FDA_ROI_Healthy'), 
                        dest = 'data:data_dir',
                        help='dataset directory')
    
    #****************** network option & hyper-parameters ******************        
    parser.add_argument('--loss', type = str, default = 'l1',
                        dest='loss:loss_name',
                        help = 'loss function (l1 or l2)')    
    parser.add_argument('--sigma', type = float, default = 1.5,
                        dest='loss:sigma',
                        help = 'sigma Gauss kernel for creating the density map')   

               
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        dest = 'train:batch_size',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        dest='solver:max_epoch',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        dest = 'lr:base_lr',
                        help='learning rate (default: 1e-4)')

    parser.add_argument('--fold', type=int, default=0,
                        help='define the K fold number for K-fold validation')

    # ****************** checkpoint and log ******************
    parser.add_argument('--save', default=os.path.join(thispath,'trained_models'),
                        dest = 'network:model_dir',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--save-per-epochs', type=int, default=15,
                        dest = 'solver:save-per-epochs',
                        help='per epochs to save')

    parser.add_argument('--test-save-dir', default=os.path.join(thispath,'TestResults'),
                        dest = 'test:save-dir',
                        help='Directory for saving model prediction results on test data.')
    
    # ****************** augmentation option ******************
    parser.add_argument('--intensity_aug', default='fourier',
                        dest = 'train:intensity_augmentation',
                        help='Type of intensity augmentation during training (None, clahe, fourier, or both)')
    parser.add_argument('--tta', action="store_true")


    parser.add_argument('REMAIN', nargs='*')
    
    args = parser.parse_args()

    if not os.path.isdir(args.__dict__['network:model_dir']):
        os.makedirs(args.__dict__['network:model_dir'])

    return args


def main(args):

    # Config paramters
    config = Configer(args_parser = args)
    
    # update chekcpoint name based on experiment
    if config.get('eval'):
       
        exp = str(config.get('network','experiment'))
    else:
        exp = str(int(time.time()))

    assert (int(exp) >0)

    dir_save = config.get('network','model_dir')
    sigma = config.get('loss','sigma')    
 
    ck_name = 'rpeNet_'+config.get('loss','loss_name')
    ck_name += '_sigma'+str(sigma)
    

    dm = os.path.join(dir_save, ck_name, config.get('train','intensity_augmentation'),str(config.get('fold')), exp)
    dlog = os.path.join(dir_save,'logs',ck_name,config.get('train','intensity_augmentation'), str(config.get('fold')), exp)

    config.update(('network','model_dir'),dm )
    config.update(('network','model_name'),'rpeNet' )
    config.update(('checkpoints','checkpoints_name'), ck_name)
    config.update(('checkpoints','checkpoints_dir'), 
                  os.path.join(dir_save,'checkpoints', str(config.get('fold'))) )
    
    config.update(('logging','log_file'), dlog)
    
    # save configuration file
    if not config.get('eval'):
        config.save_config(ck_name)
    
    # list train subjects based on fold number
    all_subj = os.listdir(config.get('data','data_dir'))

    tst_inds = [all_subj.index(subjName) for subjName in FOLDS[config.get('fold')] ]
    tst_subj = [all_subj[IND] for IND in tst_inds]
    # remove test subjects from list of all subjects before going forward
    for subj in tst_subj:
        all_subj.remove(subj)
    tst_subj = [os.path.join(config.get('data','data_dir'),x) for x in tst_subj] 


    val_subj = all_subj.pop()
    val_subj = [os.path.join(config.get('data','data_dir'), val_subj) ]
    train_subj = [os.path.join(config.get('data','data_dir'),x) for x in all_subj] 
    
    print("Train data: %s" % [os.path.basename(x) for x in train_subj])
    print("Val data: %s" % [os.path.basename(x) for x in val_subj])
    print("Tst data: %s" % [os.path.basename(x) for x in tst_subj])
    

    Dataset = RawDataset
    train_set = Dataset(
                        data_dir = train_subj, 
                        sigma = sigma,
                        crop_size = 128,
                        augmentation = get_RawTraining_aug(config.get('train','intensity_augmentation')),
                        mode = 'train',
                        loss_type = config.get('loss','loss_name'),
                        )
    val_set = Dataset(data_dir = val_subj, 
                      sigma = sigma, 
                      crop_size = 128, 
                      mode = 'train',
                      loss_type = config.get('loss','loss_name'))

    train_loader = DataLoader(dataset = train_set, 
                              collate_fn =  None,
                              batch_size = config.get('train','batch_size'), 
                              shuffle = True, 
                              pin_memory = True, 
                              num_workers = 0)
    val_loader = DataLoader(dataset = val_set, batch_size = 1, 
                            collate_fn = None,
                            shuffle = False, 
                            pin_memory = True, 
                            num_workers = 6)
 
    print("Train batch number: %i" % len(train_loader))
    print("validation batch number: %i" % len(val_loader))

  
    model = Densenet_LinkNet()

    solver = Solver(config)

    if not config.get('eval'):
        solver.train(model, train_loader, val_loader)

    # load best saved checkpoint for prediction on validation data
    modelDir = config.get('network','model_dir')
    model.load_state_dict(torch.load(modelDir+'//best_model.pth'))
    model.cuda()

    # Loop thorugh different values of threshold
    # Threshold range to optimize performance
    ThreshVals = np.linspace(0.0,0.5, num = 10)
    f1 = []
    val_loader = DataLoader(
                             dataset = Dataset(data_dir = val_subj, 
                                                crop_size = None,
                                                sigma = sigma, 
                                                augmentation = None,
                                                mode='test',
                                                loss_type = config.get('loss','loss_name')), 
                             batch_size = 1, 
                             shuffle = False,
                             pin_memory = True, 
                             num_workers = 6)    
    for nt, T in enumerate(ThreshVals):
        _, scores = solver.test(model, val_loader,borderDist = (5,5), 
                                peakThresh = T,
                                vessel_dir = [],
                                augmentation = None,
                                nout = 1 )

    
        d1 = 2*np.array(scores['recall'])*np.array(scores['precision'])
        d2 = np.array(scores['recall'])+np.array(scores['precision'])+1e-14
        f1 += [np.mean(np.divide(d1,d2))]

    # Find max f1 and associated threshold value
    indBestT = f1.index(max(f1))
    BestThresh = ThreshVals[indBestT]

    # Apply best threshold to test data and save results
    tst_loader = DataLoader(
                             dataset = Dataset(data_dir = tst_subj, 
                                                crop_size = None,
                                                sigma = sigma, 
                                                augmentation = None,
                                                mode='test',
                                                loss_type = config.get('loss','loss_name')), 
                             batch_size = 1, 
                             shuffle = False,
                             pin_memory = True, 
                             num_workers = 6)

    preds, scores = solver.test(model, tst_loader,
                                peakThresh = BestThresh, 
                                augmentation = get_RawTest_aug() if config.get('tta') else None, 
                                nout = 1 )

    dir_save = str(config.get('test','save-dir'))
    savepath = os.path.join(dir_save,ck_name,config.get('train','intensity_augmentation'))
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    sio.savemat(savepath+'/Fold'+str(config.get('fold'))+'_exp'+exp+'.mat', 
                mdict = {'predictions': preds, 
                         'image_info': tst_subj, 
                         'scores': scores,
                         'Thresh': BestThresh}
                         )
    
    
if __name__ == '__main__':
    args = parse_args()
    main(args)    
    
    