# python stuff
from tqdm import tqdm

# our stuff
from classifier.classifier_base import ClassifierBase

# torch stuff
import torch
from torch.utils.data import DataLoader

# sklearn stuff
from sklearn.cluster import MiniBatchKMeans as sklKMeans

class KMeans(ClassifierBase): # quella buona
    def __init__(self, **kwargs):
        cls_kwargs = kwargs.pop('cls_kwargs') if 'cls_kwargs' in kwargs else {}
        ClassifierBase.__init__(self, **kwargs)

        self._classifier = sklKMeans(n_clusters=self.nl_class, **cls_kwargs)

    def fit(self, **kwargs):
        '''
        Fitss clusters.
        Args:
        '''
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False
        in_dl = kwargs['dataloader']

        if verbose: 
            print('\n ---- KMeans classifier\n')

        if verbose: print('Fitting KMeans')
        for _data in tqdm(in_dl, disable=not verbose):
            data = self.parser(data=_data, **self.parser_kwargs)
            self._classifier.partial_fit(data)
        
        self._fit_dl = in_dl
        return
    
    def classifier_probabilities(self, **kwargs):
        '''
        Get prediction probabilities based on the fitted modelfor the provided inputs.
        
        Args:
        - batch: data batch containing data to be parsed with the paser function set on __init__() 
        '''
        
        batch = kwargs['batch']

        data = self.parser(data = batch, **self.parser_kwargs)
        distances = torch.tensor(self._classifier.transform(data), dtype=data.dtype)

        # convert distances to probabilities (soft assignment) Gaussian-like softmax
        # TODO: Check the var in the exponent. Should it be over the training set? Should it be there?
        #probs = torch.exp(-distances ** 2 / (2 * (distances.std() ** 2)))
        #probs = torch.exp(-distances ** 2 / 2 )

        # normalize to probabilities
        #probs /= probs.sum(dim=1, keepdim=True)

        # changing strategy: back to softmin
        probs = torch.nn.functional.softmin(distances, dim=1)
            
        return probs 
