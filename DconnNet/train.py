
'''
Modified from: https://github.com/Zyun-Y/DconnNet
'''

import numpy as np
import torch
from torch.autograd import Variable
import sys
import os
thispath = os.getcwd()
sys.path.insert(1, os.path.join(thispath, 'model'))

from model.DconnNet import DconnNet
from data_loader import Dataset
from solver import Solver
from connect_loss import Bilateral_voting

import scipy.io as sio
import argparse
import torch.nn.functional as F



torch.cuda.set_device(0) ## GPU id
thispath = os.getcwd()
parentpath = os.path.abspath(os.path.join(thispath, os.pardir))


def parse_args():
    parser = argparse.ArgumentParser(description='DconnNet Training With Pytorch')

    # dataset info
    parser.add_argument('--dataset', type=str, default='rpe',  
                        help='')    
    parser.add_argument('--data_root', type=str, 
                        default=os.path.join(parentpath,'dataset','FDA_ROI_Healthy'),  
                        help='dataset directory')
    parser.add_argument('--resize', type=int, default=[160,160], nargs='+',
                        help='image size: [height, width]')
    
    # network option & hyper-parameters
    parser.add_argument('--num-class', type=int, default=1, metavar='N',
                        help='number of classes for your data')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--lr-update', type=str, default='step',  
                        help='the lr update strategy: poly, step, warm-up-epoch, CosineAnnealingWarmRestarts')
    parser.add_argument('--lr-step', type=int, default=12,  
                        help='define only when you select step lr optimization: what is the step size for reducing your lr')
    parser.add_argument('--gamma', type=float, default=0.5,  
                        help='define only when you select step lr optimization: what is the annealing rate for reducing your lr (lr = lr*gamma)')

    parser.add_argument('--use_SDL', action='store_true', default=False,
                        help='set as True if use SDL loss; only for Retouch dataset in this code. If you use it with other dataset please define your own path of label distribution in solver.py')
    #parser.add_argument('--folds', type=int, default=1,
    #                    help='define folds number K for K-fold validation')

    # checkpoint and log
    parser.add_argument('--pretrained', type=str, 
                        default=os.path.join(parentpath,'DconnNet','models','1','best_model.pth'),
                        help='put the path pretrained model for inference')
    parser.add_argument('--weights', type=str, default='/home/ziyun/Desktop/project/BiconNet_codes/DconnNet/general/data_loader/retouch_weights/',
                        help='path of SDL weights')
    parser.add_argument('--save', default='save',
                        help='Directory for saving checkpoint models')

    parser.add_argument('--save-per-epochs', type=int, default=15,
                        help='per epochs to save')

                        
    # evaluation only
    parser.add_argument('--test_only', action='store_true', default=False,
                        help='test only, please load the pretrained model')
    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    return args


def main(args):
    
    if args.test_only:
        assert args.pretrained

        augmentation = get_aug()
        SavePath  = './vessel_predictions/'
        if not os.path.exists(SavePath):
            os.makedirs(SavePath)        

        # dataset
        all_subj = os.listdir(args.data_root)
        keepLast = True if 'snr' in args.data_root else False
       
       # Modify these values if using with other external dataset. 
        init_size = (512,502) if 'FDA' in args.data_root else (320,320)
        in_size = (512,512) if 'FDA' in args.data_root else (320,320)

        dataset = Dataset(data_dir = [os.path.join(args.data_root,x) for x in all_subj] , 
                          augment = False, 
                          add_vessel = False,
                          resize = in_size,
                          keep_last = keepLast)
        data_loader = torch.utils.data.DataLoader(dataset = dataset, batch_size = 1,                 
                                shuffle = False, pin_memory = True, num_workers = 6)
        
        # load model
        model = DconnNet(num_class=args.num_class).cuda()
        model.load_state_dict(torch.load(args.pretrained,map_location = torch.device('cpu')))
        model = model.cuda()


        # prediction
        model.eval()
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(data_loader):

                batch = 1
                H, W = in_size

                hori_translation = torch.zeros([1,args.num_class,W,W])
                for i in range(W-1):
                    hori_translation[:,:,i,i+1] = torch.tensor(1.0)
                verti_translation = torch.zeros([1,args.num_class,H,H])
                for j in range(H-1):
                    verti_translation[:,:,j,j+1] = torch.tensor(1.0)
                hori_translation = hori_translation.float()
                verti_translation = verti_translation.float()

                hori_translation = hori_translation.repeat(batch,1,1,1).cuda()
                verti_translation = verti_translation.repeat(batch,1,1,1).cuda()

                if augmentation:
                    ext = '_tta'
                    X_test = test_data[0]
                    y_test = test_data[1]

                    class_pred = torch.zeros((args.num_class,8,)+in_size+(len(augmentation),))

                    for nt,t in enumerate(augmentation):
                        if t.__name__ != 'identity': 
                            out = t(X_test.squeeze().cpu().numpy(), y_test.squeeze(0).cpu().numpy())
                            img_in = out[0]
                            img_in = torch.Tensor(img_in.copy()).unsqueeze(0)
                        else:
                            img_in = X_test

                        # pass thorugh network
                        pred_model,_ = model(Variable(img_in).cuda())  
                        pred_model = F.sigmoid(pred_model)    
                       
                        pred_model = pred_model.squeeze().cpu().numpy()
                        # Redo augmentation (if not intensity)
                        if t.__name__ in ['hflip','vflip']:
                            pred_model = pred_model[:,::-1,...] if t.__name__ == 'hflip' else pred_model[...,::-1]
                        if t.__name__ == 'transpose':
                            pred_model = np.transpose(pred_model, [0,2,1])

                        output_test = torch.tensor(pred_model.copy())
                        class_pred[...,nt] = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                        

                    class_pred = torch.mean(class_pred,axis=-1, keepdim =False).view([batch,-1,8,H,W])
                    pred = torch.where(class_pred>0.5,1,0).cuda()   
                else:
                    ext = ''
                    X_test = Variable(test_data[0])
                    X_test= X_test.cuda()

                    output_test,_ = model(X_test)
                    output_test = F.sigmoid(output_test)                
                    class_pred = output_test.view([batch,-1,8,H,W]) #(B, C, 8, H, W)
                    pred = torch.where(class_pred>0.5,1,0)

                
                pred,_ = Bilateral_voting(pred.float(),hori_translation,verti_translation) 
                
                img = pred.squeeze().cpu().numpy()
                img = img[:init_size[0],:init_size[1]]
                sio.savemat(SavePath+''.join(test_data[-1][0])+'_'+''.join(test_data[-1][1])+ext+'.mat', 
                       mdict={'vessel_map': img})            
        

    else:
        ## training ##
        exp_id = 0
       
        all_subj = os.listdir(args.data_root)
        val_subj = all_subj.pop()
        val_subj = [os.path.join(args.data_root, val_subj) ]
        train_subj = [os.path.join(args.data_root,x) for x in all_subj]        

        train_set = Dataset(data_dir = train_subj, augment = True, resize = args.resize)
        val_set = Dataset(data_dir = val_subj, augment = False, resize = args.resize)

        train_loader = torch.utils.data.DataLoader(dataset = train_set, 
                                batch_size = args.batch_size, 
                                shuffle = True, pin_memory = True, num_workers = 6)
        val_loader = torch.utils.data.DataLoader(dataset = val_set, batch_size = 1,                 
                                shuffle = False, pin_memory = True, num_workers = 6)

        print("Train batch number: %i" % len(train_loader))
        print("Test batch number: %i" % len(val_loader))

        #### Above: define how you get the data on your own dataset ######
        model = DconnNet(num_class=args.num_class).cuda()

        if args.pretrained:
            model.load_state_dict(torch.load(args.pretrained,map_location = torch.device('cpu')))
            model = model.cuda()

        solver = Solver(args)

        solver.train(model, train_loader, val_loader,exp_id+1, num_epochs=args.epochs)

########### Augmentation functions
def get_aug():                      
    t = [Identity(),
         RandomHFlip(0),
         RandomVFlip(0),
         RandomTranspose(0)]
    return t

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

    def forward(self, img, label):
        
        if np.random.rand(1) >self.p:
            img = img[...,::-1]
            label = label[...,::-1]
        return img, label
    
class RandomHFlip(torch.nn.Module):
    __name__ = 'hflip'
    def __init__(self, p = 0.5):
        super(RandomHFlip, self).__init__()
        self.p = p

    def forward(self, img, label):
        if np.random.rand(1) > self.p:
            img = img[:,::-1,:]
            label = label[:,::-1,:]
        return img, label
    
class RandomTranspose(torch.nn.Module):
    __name__ = 'transpose'
    def __init__(self, p = 0.5):
        super(RandomTranspose, self).__init__()
        self.p = p

    def forward(self, img, label):
        if np.random.rand(1) > self.p:
            img = np.transpose(img, [0,2,1])
            label = np.transpose(label,[0,2,1])     
        return img, label   
        
if __name__ == '__main__':
    args = parse_args()
    main(args)
    
