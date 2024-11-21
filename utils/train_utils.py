'''
parts from:
    https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/utils/train.py
'''

import sys
import torch
import numpy as np

from tqdm import tqdm as tqdm
from .meter import AverageValueMeter

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    

class Epoch:
    def __init__(self, model, loss, stage_name,  device="cpu", verbose=True):
        self.model = model
        self.loss = loss
       
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if isinstance(self.loss,dict):
            self.loss['losses'] = [l.to(self.device) for l in self.loss['losses']]
        else:
            self.loss.to(self.device)


    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, epoch = None):

        self.on_epoch_start()

        logs = {}
        
        loss_meter = AverageValueMeter()
        

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, label_map in iterator:
                x = x.to(self.device)
                if isinstance(label_map, dict):
                    for k in label_map.keys():
                        label_map[k] = label_map[k].to(self.device)
                else:
                    label_map = label_map.to(self.device)
                    
                loss, y_pred, loss_terms = self.batch_update(x, label_map, epoch)
                if isinstance(y_pred,dict):
                    y_pred = y_pred['center']

                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)

                name = self.loss.__name__
                loss_logs = {name: loss_meter.mean}
                logs.update(loss_logs)


                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss,optimizer, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer


    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, epoch = None):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)
        if isinstance(prediction, dict):
            prediction = prediction['center']

        # multiple loss terms if dictionary. 
        loss_terms = {}

        loss = self.loss(prediction, y)
        loss_terms[self.loss.__name__] = loss

        loss.backward()
        self.optimizer.step()
        return loss, prediction, loss_terms


class ValidEpoch(Epoch):
    def __init__(self, model, loss, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, epoch):
        with torch.no_grad():
            prediction = self.model.forward(x)
        if isinstance(prediction, dict):
            prediction = prediction['center']

        # multiple loss terms if dictionary. First train with chf, then transition to l1
        loss_terms = {}

        loss = self.loss(prediction, y)
        loss_terms[self.loss.__name__] = loss

        return loss, prediction, loss_terms