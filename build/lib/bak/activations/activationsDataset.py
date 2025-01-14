# torch stuff
import torch
from torch.utils.data import TensorDataset 

class ActivationsDataset(TensorDataset):
    def __init__(self, activation_keys):
        self.in_activations = {} 
        self.out_activations = {}

        for key in activation_keys:
            self.in_activations[key] = None 
            self.out_activations[key] = None 
                                             
        self.images = None 
        self.labels = None 
        self.preds = None 
        self.results = None 
        self.n = 0
        return
        
    def __getitem__(self, idx):
        if self.n == 0:
            return {} 

        _i = self.images[idx]
        _l = self.labels[idx]
        _p = self.preds[idx]
        _r = self.resuts[idx]
        
        _ia = {}
        for k in self.in_activations:
            _ia[key] = self.in_activations[key][idx]
        
        _oa = {}
        for k in self.out_activations:
            _oa[key] = self.out_activations[key][idx]

        return {
                'image': _i,
                'label': _l,
                'pred': _p,
                'results': _r,
                'in_activations': _ia,
                'out_activations': _oa
                }

    def add(self, image, label, pred, result, in_activations, out_activations):
        bs = image.shape[0]
        
        if self.n == 0:
            self.images = image
            self.labels = label
            self.preds = pred
            self.results = result
            
            for k in self.in_activations:
                if in_activations[k] != None:
                    self.in_activations[k] = in_activations[k]
            for k in self.out_activations:
                if out_activations[k] != None:
                    self.out_activations[k] = out_activations[k]
        else:
            self.images = torch.vstack((self.images, image))
            self.labels = torch.hstack((self.labels, label))
            self.preds = torch.hstack((self.preds, pred))
            self.results = torch.hstack((self.results, result))
        
            for k in self.in_activations:
                if in_activations[k] != None:
                    self.in_activations[k] = torch.vstack((self.in_activations[k], in_activations[k]))
            
            for k in self.out_activations:
                if out_activations[k] != None: 
                    self.out_activations[k] = torch.vstack((self.out_activations[k], out_activations[k]))

        self.n += bs
        return

    def __len__(self):
        return self.n
