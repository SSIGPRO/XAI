# our stuff
from classifier.classifier_base import ClassifierBase

# torch stuff
import torch
from torch.utils.data import DataLoader

# sklearn stuff
from sklearn.mixture import GaussianMixture

# TODO: check learning models such as https://discuss.pytorch.org/t/fit-gaussian-mixture-model/121826

class GMM(ClassifierBase): # quella buona
    def __init__(self, **kwargs):
        cls_kwargs = kwargs.pop('cls_kwargs') if 'cls_kwargs' in kwargs else {}
        ClassifierBase.__init__(self, **kwargs)
        
        self._classifier = GaussianMixture(n_components=self.nl_class, **cls_kwargs)

    def fit(self, **kwargs):
        '''
        Fit GMM. 
        
        Args:
        '''
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        in_dl = kwargs['dataloader']
        
        if verbose: 
            print('\n ---- GMM classifier\n')
            print('Parsing data')

        # temp dataloader for loading the whole dataset
        _dl = DataLoader(in_dl.dataset, batch_size=len(in_dl.dataset), collate_fn=lambda x: x, shuffle=False) 
        _data = next(iter(_dl))
        data = self.parser(data=_data, **self.parser_kwargs)

        if verbose: print('Fitting GMM')
        self._classifier.fit(data)
        
        self._fit_dl = _dl
        return
    
    def classifier_probabilities(self, **kwargs):
        '''
        Get prediction probabilities based on the fitted modelfor the provided inputs.
        
        Args:
        - batch: data containing data to be parsed with the paser function set on __init__() 
        '''
        
        batch = kwargs['batch']

        data = self.parser(data = batch, **self.parser_kwargs)
        probs = torch.tensor(self._classifier.predict_proba(data), dtype=data.dtype)
        return probs  
            
