import matplotlib.pyplot as plt
import sys
import os
import warnings
import time
import pickle
import random
import numpy as np
import pandas as pd

from itertools import islice
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Subset, ConcatDataset
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms
from torchvision.models import vgg16, vgg16_bn
from torchvision.models import VGG16_Weights
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

import sklearn.metrics as mtr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import adjusted_rand_score as ARI
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans as KM
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score as AUC
from sklearn.metrics import accuracy_score, top_k_accuracy_score

import scipy
from scipy.special import softmax as SM
from scipy.stats import entropy as H
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde as KDE

import collections
import functools
import types

import yaml

abs_lm = '/srv/newpenny/XAI/LM'
abs_lc = '/srv/newpenny/XAI/LC'
activations_path = os.path.join(abs_lc, 'data', 'dict_activations')

def load_res(name):
    path = os.path.join(abs_lm, 'results', 'paper', name)
    with open(path, 'rb') as fp:
        file = pickle.load(fp)   
    return file

def load_res_lm(name):
    path = os.path.join(abs_lm, 'results', 'paper', name)
    with open(path, 'rb') as fp:
        file = pickle.load(fp)  
    return file

def load_res_lc(name):
    path = os.path.join(abs_lc, 'data', 'control', name)
    with open(path, 'rb') as fp:
        file = pickle.load(fp)  
    return file

def save_res_lc(name, file):
    path = os.path.join(abs_lc,'data', 'control', name)
    with open(path, 'wb') as fp:
        pickle.dump(file, fp)

def save_res(name, file):
    path = os.path.join(abs_lm,'results', 'paper', name)
    with open(path, 'wb') as fp:
        pickle.dump(file, fp)

def load_activations(activations_path, dataset, portion):
    # load dict activations with clas 0 and 3
    layers_dict = {'clas': [0,3]}
    model_name = 'vgg16'
    dataset_name = dataset
    fine_tuned = True
    if fine_tuned and (dataset_name=='CIFAR10' or dataset_name=='CIFAR100'):
        seed = 29
    else:
        seed = 'nd'
    layers = ''
    for key in layers_dict.keys():
        for index in layers_dict[key]:
            layer = key + '_' + str(index) + '&'
            layers += layer

    dir_ = 'in'
    dict_file = f'dict_activations_portion={portion}_model={model_name}_'\
                f'layer={layers}_dir={dir_}_ft={fine_tuned}_seed={seed}_'\
                f'dataset={dataset_name}_.pkl' 

    path = os.path.join(activations_path, dict_file)
    with open(path, mode='rb') as fp:
        dict_activations = pickle.load(fp)

    # load and add things related to conv2d layers
    act_file = {}
    for file in os.listdir(activations_path):
        if file.startswith('flattened_activations'):
            if portion in file:
                layer_index = file.split('-')[3]
                act_file[f'feat-{layer_index}'] = file

    for key, file in act_file.items():
        path = os.path.join(activations_path, file)
        dict_activations[key] = scipy.sparse.load_npz(path)
  
    return dict_activations

class DictDataset(torch.utils.data.Dataset):
    '''Requires keys as (idx, label) '''
    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict) 

    def __getitem__(self, idx):
        key = list(self.data_dict.keys())[idx]
        label = key[1]  
        image = self.data_dict[key]

        if self.transform:
            image = self.transform(image)
        return image, label


def dataloader_from_dict(data_dict, batch_size, shuffle):
    '''Might also include transforms'''
    dataset = DictDataset(data_dict)
    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_size=batch_size, 
                                             shuffle=shuffle
                                             )
    return dataloader
                    
####
def load_data(dataset, 
              portion, 
              batch_size, 
              data_kwargs, 
              seed=42, 
              data_augmentation=False, 
              shuffle_train=True):
    '''
    Load and prepare data for a specified portion of a dataset.

    Args:
    - dataset (str): The name of the dataset ('CIFAR10', 'CIFAR100' or 'imagenet-1k').
    - portion (str): The portion of the dataset to load ('train', 'val', 
    'test', 'classes').
    - batch_size (int): The batch size for DataLoader.
    - data_kwargs (dict): Additional keyword arguments for DataLoader.
    - seed (int): Random seed for reproducibility (default: 42).
    - data_augmentation (bool): Flag indicating whether to apply data 
    augmentation (default: False).

    Returns:
    - DataLoader or dict: Depending on the specified portion, returns 
    either a DataLoader for 'train', 'val', or 'test',
    or a dictionary mapping class indices to class names for 'classes'.
    If the specified portion is invalid, None is returned.

    Example:
    - To load the training data of CIFAR10 with a batch size of 32:
      >>> train_loader = load_data('CIFAR10', 'train', batch_size=32, data_kwargs={})

    - To get a dictionary mapping class indices to names for CIFAR100:
      >>> class_dict = load_data('CIFAR100', 'classes', batch_size=32, data_kwargs={})
    '''
    
    dataset_config = {
        'CIFAR10': {'num_classes': 10, 
                    'input_ch': 3, 
                    'means': (0.424, 0.415, 0.384), 
                    'stds': (0.283, 0.278, 0.284)},

        'CIFAR100': {'num_classes': 100, 
                     'input_ch': 3, 
                     'means': (0.438, 0.418, 0.377), 
                     'stds': (0.300, 0.287, 0.294)},
        
        'imagenet-1k': {'num_classes': 1000, 
                     'input_ch': 3, 
                     'means': (0.485, 0.456, 0.406), 
                     'stds': (0.229, 0.224, 0.225)}
    }
    
    if dataset not in dataset_config:
        print('Dataset not found')
        return

    config = dataset_config[dataset]
    
    if dataset=='CIFAR100':
        data_path = os.path.join('/srv/newpenny/XAI/LM', 'data', dataset)
    else:
        data_path = os.path.join('data', dataset)
    
    torch.manual_seed(seed)
    
    if dataset != 'imagenet-1k':
        # original dataset without augmentation
        original_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(config['means'], config['stds'])
        ])
    
        train_data = torchvision.datasets.__dict__[dataset](
            root=data_path,
            train=True,
            transform=original_transform,
            download=True
        )

        test_data = torchvision.datasets.__dict__[dataset](
            root=data_path,
            train=False,
            transform=original_transform,
            download=True
        )

        # augmented dataset
        if data_augmentation:
 
            autoaugment_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10), 
                # transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
                transforms.ToTensor(),
                transforms.Normalize(config['means'], config['stds'])
            ])

            augmented_data = torchvision.datasets.__dict__[dataset](
                root=data_path,
                train=True,
                transform=autoaugment_train,
                download=True
            )

            # take the augmented version for training
            train_dataset, _ = random_split(
                augmented_data,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(seed)
            )

            # take the data with original_transform for validation
            _, val_dataset = random_split(
                train_data,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(seed)
            )

        else:
            train_dataset, val_dataset = random_split(
                train_data,
                [0.8, 0.2],
                generator=torch.Generator().manual_seed(seed)
            )
            
        loaders = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle_train, **data_kwargs),
            'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **data_kwargs),
            'test': DataLoader(test_data, batch_size=batch_size, shuffle=False, **data_kwargs),
            'classes': {i: class_name for i, class_name in enumerate(test_data.classes)}
        }
        
    else:
        
        print('dataaset not handled')
        # loaders = load_imagenet1k('pytorch',
        #                preprocessing=True,
        #                data_kwargs=data_kwargs, 
        #                batch_size=batch_size,
        #                data_augmentation=data_augmentation
        #                )

    return loaders[portion] if portion in loaders else print('Not a valid portion')

def get_svd(params, layer_type, layer_index, input_shape=None, k=None):

    '''

    Compute Singular Value Decomposition (SVD) of the weights of a neural network layer.
 
    Parameters:

        params (dict): A dictionary containing the parameters of the neural network model.

        layer_type (str): Type of the layer ('features' for convolutional layers, else for fully connected layers).

        layer_index (int): Index of the layer whose weights are to be decomposed.

        input_shape (tuple, optional): Shape of the input to the layer. Required for convolutional layers.

        k (int, optional): Number of singular values to compute. Default is None.
 
    Returns:

        tuple: A tuple containing the decomposed matrices U, s, and Vh.

            U (ndarray): Left singular vectors.

            s (ndarray): Singular values.

            Vh (ndarray): Right singular vectors.
 
    Raises:

        ValueError: If input_shape is not provided for convolutional layers.
 
    Note:

        For convolutional layers ('features' type), the function utilizes a sparse representation of the weight matrix

        and computes the SVD. For fully connected layers, it directly applies the SVD.

    '''

    w_name = f'{layer_type}.{layer_index}.weight'

    b_name = f'{layer_type}.{layer_index}.bias'

    W = params[w_name] # get weight from state_dict

    b = params[b_name] # get bias from state_dict
 
    if layer_type == 'features': # conv layers

        if input_shape is not None:

            coo_mat = conv2d_to_sparse(input_shape, W, b, stride=(1, 1), padding=(1, 1))

            W_csr = coo_mat.to_csr()

            if k is None:

                k = min(W_csr.shape) - 1

                k = 10 # to do not for cifar10

            U, s, Vh = scipy.sparse.linalg.svds(W_csr, k=k)

            U = U[::-1]

            s = s[::-1]

            Vh = Vh[::-1]

        else:

            raise ValueError('input shape not defined')

    else: # FC layers

        b_ = b.reshape((b.shape[0], 1))
    
        W_ = np.concatenate((W, b_), axis=1)
    
        U, s, Vh = scipy.linalg.svd(W_, full_matrices=False)

    return U, s, Vh

def get_svd_sparse(W_csr, k=None, solver='arpack'):
    '''
    Perform Singular Value Decomposition (SVD) on a sparse matrix in 
    Compressed Sparse Row (CSR) format.

    Parameters:
        - W_csr (scipy.sparse.csr_matrix): Input sparse matrix in CSR 
        format containing both weight and bias information.
        - k (int, optional): Number of singular values and vectors to 
        compute. If None, computes min(W_csr.shape) - 1.
        - solver (str, optional): Solver to use for computing the SVD. 
        Default is 'arpack'.

    Returns:
        - U (ndarray): Left singular vectors.
        - s (ndarray): Singular values.
        - Vh (ndarray): Right singular vectors.

    Notes:
        - This function computes the SVD of a sparse matrix using 
        scipy.sparse.linalg.svds().
        - The input matrix W_csr is assumed to already encompass both
        weight and bias information.
        - If solver='propack' is desired, set 
        os.environ['SCIPY_USE_PROPACK'] = "True" before importing scipy.
        - If k is not provided, it defaults to min(W_csr.shape) - 1, 
        which is suitable for 'propack' solver.

    Example:
        U, s, Vh = get_svd_sparse(W_csr, k=10, solver='arpack')
    '''
    # W_csr already encompasses the information of both weight and bias       
    if k is None:
        if solver=='propack':
            k = min(W_csr.shape) - 1 # ok for solver='propack'
        else:
            k = min(W_csr.shape) - 2
            
    U, s, Vh = scipy.sparse.linalg.svds(W_csr, k=k, solver=solver)
    U = U[::-1]
    s = s[::-1]
    Vh = Vh[::-1]

    return U, s, Vh

def get_peepholes_train(h_data, new_Vh, stdze=True):
    """
    Obtain training peephole vectors from extracted features and optionally standardize
    the data.

    Parameters:
    ----------
    h_data : array-like, shape (n_samples, n_features)
        Feature vectors to be processed.

    new_Vh : array-like, shape (n_features, n_peepholes)
        Matrix representing the right eigenvectors of the SVD.

    stdze : bool, optional (default=True)
        Whether to standardize the data. If True, the data will be standardized.

    Returns:
    -------
    X_train : array-like, shape (n_samples, n_peepholes)
        Peephole data ready for training.

    scaler : tuple, optional
        Tuple containing the mean and standard deviation used for standardization,
        returned only if stdze is True.
    """
    one = np.ones((h_data.shape[0], 1))
    new_h = np.concatenate((h_data, one), axis=1)
    new_data = scipy.sparse.csr_array(new_h)
    new_Vh_ = scipy.sparse.csr_matrix(new_Vh)
    
    p_ = new_data.dot(new_Vh_.transpose()).toarray()
    
    # p_vectors = []
    # for _, h_ in enumerate(h_data):
    #     tmp = np.concatenate((h_, [1]))
    #     tmp = new_Vh@tmp
    #     p_vectors.append(tmp)
    # p_ = np.concatenate([p_vectors])
    
    if stdze:
        #print('applying std')
        scaler = (p_.mean(axis=0), p_.std(axis=0))
        p_std = (p_ - scaler[0])/(scaler[1])
    
        if p_std.ndim==1:
            X_train = p_std.reshape(-1, 1)
        else:
            X_train = p_std

        return X_train, scaler
    
    else:
        
        if p_.ndim==1:
            X_train = p_.reshape(-1, 1)
        else:
            X_train = p_
            
        return X_train

def get_peepholes_test(h_data, new_Vh, scaler, stdze=True):
    """
    Obtain test peephole vectors from extracted features and optionally standardize
    the data with mean and std from the training data.
    
    Parameters:
    ----------
    h_data : array-like, shape (n_samples, n_features)
        Feature vectors to be processed.

    new_Vh : array-like, shape (n_features, n_peepholes)
        Matrix representing the right eigenvectors of the SVD.

    scaler : tuple
        Tuple containing the mean and standard deviation used for standardization.

    stdze : bool, optional (default=True)
        Whether to standardize the data. If True, the data will be standardized using 
        the provided scaler.

    Returns:
    -------
    X_test : array-like, shape (n_samples, n_peepholes)
        Peephole data ready for testing.
    """
    
    # p_vectors = []
    # for _, h_ in enumerate(h_data):
    #     tmp = np.concatenate((h_, [1]))
    #     tmp = new_Vh@tmp
    #     p_vectors.append(tmp)
    # p_ = np.concatenate([p_vectors])

    one = np.ones((h_data.shape[0], 1))
    new_h = np.concatenate((h_data, one), axis=1)
    new_data = scipy.sparse.csr_array(new_h)
    new_Vh_ = scipy.sparse.csr_matrix(new_Vh)
    
    p_ = new_data.dot(new_Vh_.transpose()).toarray()
    
    if stdze:
        mean = scaler[0]
        std = scaler[1]
        p_std = (p_ - mean)/(std)
        
    if p_std.ndim==1:
        X_test = p_std.reshape(-1, 1)
    else:
        X_test = p_std
        
    return X_test

def get_dict_peephole_train(dim=None, dict_activations_train=None, n_clusters=None, dict_SVD=None):
    peephole_ = {}
    clustering_config_ = {}
    dict_peephole_train = {}
    for key in n_clusters.keys():
        V = dict_SVD[key][2][0:dim]
        data_ = dict_activations_train[key]
        
        if hasattr(data_, 'format') and data_.format == 'csr':
            h_data = dict_activations_train[key].toarray()
        else:
            h_data = torch.cat(dict_activations_train[key])
        
        X_train, scaler = get_peepholes_train(h_data, new_Vh=V, stdze=True)
        seed = random.randint(0,100)

        km = KM(n_clusters=n_clusters[key], random_state=seed).fit(X_train)
        peephole_[key] = [X_train,scaler]
        clustering_config_[key] = km
        
    dict_peephole_train['peephole'] = peephole_
    dict_peephole_train['clustering_config'] = clustering_config_
    
    return dict_peephole_train

def get_dict_peephole_val(dim=None, dict_activations_val=None, dict_peephole_train=None, n_clusters=None, dict_SVD=None):
    peephole_ = {}
    clustering_config_ = {}
    dict_peephole_val = {}
    for key in n_clusters.keys():
        V = dict_SVD[key][2][0:dim]        
        data_ = dict_activations_val[key]
        
        if hasattr(data_, 'format') and data_.format == 'csr':
            h_data = dict_activations_val[key].toarray()
        else:
            h_data = torch.cat(dict_activations_val[key])
        
        _, scaler = dict_peephole_train['peephole'][key]
        X_val = get_peepholes_test(h_data, new_Vh=V, scaler=scaler, stdze=True)
        peephole_[key] = X_val
        clustering_config_[key] = dict_peephole_train['clustering_config'][key]
        
    dict_peephole_val['peephole'] = peephole_
    dict_peephole_val['clustering_config'] = clustering_config_
    
    return dict_peephole_val


def get_clustering_config(dict_peephole_train=None,n_clusters=None):

    dict_peephole_train_new = {}
    peephole_ = {}
    clustering_config_ = {}
    
    for key in n_clusters.keys():
        X_train, scaler = dict_peephole_train['peephole'][key]
        seed = random.randint(0,100)
        km = KM(n_clusters=n_clusters[key], random_state=seed).fit(X_train)
        peephole_[key] = [X_train,scaler]
        clustering_config_[key] = km
        
    dict_peephole_train_new['peephole'] = peephole_
    dict_peephole_train_new['clustering_config'] = clustering_config_
    
    return dict_peephole_train_new    

class SaveInput:
    def __init__(self):
        self.activations = []
        
    def __call__(self, module, module_in, module_out):
        self.activations.append(module_in)
        
    def clear(self):
        self.activations = []

class SaveOutput:
    def __init__(self):
        self.activations = []
        
    def __call__(self, module, module_in, module_out):
        self.activations.append(module_out)
        
    def clear(self):
        self.activations = []
        
def get_activation_VGG(model, loader, layers_dict, dir, device):
    """
    The following function aims to simplify the procedure to extract the activations generated by a VGG network given a specific input. 
    To do so, the function get_activation_VGG deploys the built-in hooks to extract the required information. The result of this function 
    is a dictionary which contains firstly the images, the corresponding labels, the predictions of the model and then the activations 
    divided in the specific layers.

    Here an example of the structure of VGG-16:

        VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=100, bias=True)
      )
    )

    For the other VGG models, the main structure is the same because it is diveded in the three main sections ('feat','avgpool', 'clas'), the only difference 
    is the number of layers that compose each section.

    So if you want the input of the first FC-layer you call get_activation_VGG(model, loader, string_section='clas', index='0', dir=True)

    If you instead want to obtain the output of the third convolutional layer you call get_activation_VGG(model, loader, string_section='feat', index='5', dir=False)

    Parameters:
    -------
    model: the reference model, which generates the activation we are interested in

    loader: The function takes as input the entire loader of a specfic dataset. Given the loader the function iterate over it and extract for each batch the corresponding activations

    string_section: it is a string that identifies the section within the network, you can choose among: (features, avgpool, classifier)

    index: the index of the layer entered as a string in order to identify which is the specific layer within the specific section we are interested in 

    dir: a string that indicates if the activation we get is either the input or the output of the chosen layer.
        the admissible strings are either 'in' or 'out'.

    Returns:
    -------

    dict_activations: this dictonary contains the images given as input to the model, the corresponding labels, the output pprovided by the network and then all the activations we selected thorugh 
                      the input strings string_section and index. 
                      The keys of the provided dictionary are the following:
                      
                          1. 'images'
                          2. 'labels'
                          3. 'pred'
                          4. 'activations'- the name of the key is defined by the inputs string_section and index
                      
    """

    dict_activations = {}

    dict_activations['image'] = []
    dict_activations['label'] = []
    dict_activations['pred'] = []
    dict_activations['results'] = []
    

    hook_handles = []
    keys_act = []
    
    if dir == 'in':
        save_activation = SaveInput() 
    else:
        save_activation = SaveOutput()

    for key in layers_dict.keys():
        
        for index in layers_dict[key]:
            
            key_activation = key + '-' + str(index)
            keys_act.append(key_activation)
            dict_activations[key_activation] = []

            if key == 'feat':

                    print('sono in feat')
        
                    for name, layer in model.features.named_children():
                            
                        if name == str(index):

                            print(name)
        
                            handle = layer.register_forward_hook(save_activation)
                            
                            hook_handles.append(handle)
    
                            print(hook_handles)
        
            elif key == 'avgpool':
        
                for name, layer in model.avgpool:
                    
                    handle = layer.register_forward_hook(save_activation)
                    
                    hook_handles.append(handle)

                    print(hook_handles)
        
            else:
    
                for name, layer in model.classifier.named_children():
        
                    if name == str(index):
    
                        handle = layer.register_forward_hook(save_activation)
                        
                        hook_handles.append(handle)    

    flatten = nn.Flatten()
    
    for data in loader: 
        
        image, label = data

        dict_activations['image'].append(image)
        dict_activations['label'].append(label)
        
        image = image.to(device)
        label = label.to(device)

        y_predicted = model(image)

        idx = 0

        for key in keys_act:

            # print(key)
            # print(key[0:4])

            if key[0:4] == 'clas':

                if dir == 'in':
                    
                    dict_activations[key].append(save_activation.activations[idx][0].detach().cpu())
                else:
                    
                    dict_activations[key].append(save_activation.activations[idx].detach().cpu())
            else:

                if dir == 'in':
                    
                    dict_activations[key].append(flatten(save_activation.activations[idx][0].detach().cpu()))
                else:
                    
                    dict_activations[key].append(flatten(save_activation.activations[idx].detach().cpu()))
            
            idx += 1
            
        save_activation.clear()
        
        labels_predicted = y_predicted.argmax(axis = 1)
        
        dict_activations['pred'].append(labels_predicted.detach().cpu())

        results = labels_predicted == label

        dict_activations['results'].append(results.detach().cpu())
        
    return dict_activations


def get_InputOutput_layer(model, loader, layer, device):
    """
    The following function aims to simplify the procedure to extract the activations generated by a VGG network given a specific input. 
    To do so, the function get_activation_VGG deploys the built-in hooks to extract the required information. The result of this function 
    is a dictionary which contains firstly the images, the corresponding labels, the predictions of the model and then the activations 
    divided in the specific layers.

    Here an example of the structure of VGG-16:

        VGG(
      (features): Sequential(
        (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (3): ReLU(inplace=True)
        (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): ReLU(inplace=True)
        (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (8): ReLU(inplace=True)
        (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (11): ReLU(inplace=True)
        (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (13): ReLU(inplace=True)
        (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (15): ReLU(inplace=True)
        (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (18): ReLU(inplace=True)
        (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (20): ReLU(inplace=True)
        (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (22): ReLU(inplace=True)
        (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (25): ReLU(inplace=True)
        (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (27): ReLU(inplace=True)
        (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (29): ReLU(inplace=True)
        (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
      (classifier): Sequential(
        (0): Linear(in_features=25088, out_features=4096, bias=True)
        (1): ReLU(inplace=True)
        (2): Dropout(p=0.5, inplace=False)
        (3): Linear(in_features=4096, out_features=4096, bias=True)
        (4): ReLU(inplace=True)
        (5): Dropout(p=0.5, inplace=False)
        (6): Linear(in_features=4096, out_features=100, bias=True)
      )
    )

    For the other VGG models, the main structure is the same because it is diveded in the three main sections ('feat','avgpool', 'clas'), the only difference 
    is the number of layers that compose each section.

    So if you want the input of the first FC-layer you call get_activation_VGG(model, loader, string_section='clas', index='0', dir=True)

    If you instead want to obtain the output of the third convolutional layer you call get_activation_VGG(model, loader, string_section='feat', index='5', dir=False)

    Parameters:
    -------
    model: the reference model, which generates the activation we are interested in

    loader: The function takes as input the entire loader of a specfic dataset. Given the loader the function iterate over it and extract for each batch the corresponding activations

    string_section: it is a string that identifies the section within the network, you can choose among: (features, avgpool, classifier)

    index: the index of the layer entered as a string in order to identify which is the specific layer within the specific section we are interested in 

    dir: a string that indicates if the activation we get is either the input or the output of the chosen layer.
        the admissible strings are either 'in' or 'out'.

    Returns:
    -------

    dict_activations: this dictonary contains the images given as input to the model, the corresponding labels, the output pprovided by the network and then all the activations we selected thorugh 
                      the input strings string_section and index. 
                      The keys of the provided dictionary are the following:
                      
                          1. 'images'
                          2. 'labels'
                          3. 'pred'
                          4. 'activations'- the name of the key is defined by the inputs string_section and index
                      
    """

    dict_activations = {}

    dict_activations['image'] = []
    dict_activations['label'] = []
    dict_activations['pred'] = []
    dict_activations['results'] = []
    

    hook_handles = []
    keys_act = []
    
    save_activation = SaveInput() 

    save_activation = SaveOutput()

    if layer[0:3] == 'feat':

        print('sono in feat')

        for name, layer in model.features.named_children():
                
            if name == str(index):

                print(name)

                handle = layer.register_forward_hook(save_activation)
                
                hook_handles.append(handle)

                print(hook_handles)
        
    elif layer[0:3] == 'avgpool':

        for name, layer in model.avgpool:
            
            handle = layer.register_forward_hook(save_activation)
            
            hook_handles.append(handle)

            print(hook_handles)

    else:

        for name, layer in model.classifier.named_children():

            if name == str(index):

                handle = layer.register_forward_hook(save_activation)
                
                hook_handles.append(handle)    

    flatten = nn.Flatten()
    
    for data in loader: 
        
        image, label = data

        dict_activations['image'].append(image)
        dict_activations['label'].append(label)
        
        image = image.to(device)
        label = label.to(device)

        y_predicted = model(image)

        idx = 0

        for key in keys_act:

            # # print(key)
            # # print(key[0:4])

            if key[0:4] == 'clas':

                if dir == 'in':
                    
                    dict_activations[key].append(save_activation.activations[idx][0].detach().cpu())
                else:
                    
                    dict_activations[key].append(save_activation.activations[idx].detach().cpu())
            else:

                if dir == 'in':
                    
                    dict_activations[key].append(flatten(save_activation.activations[idx][0].detach().cpu()))
                else:
                    
                    dict_activations[key].append(flatten(save_activation.activations[idx].detach().cpu()))
            
            idx += 1
            
        save_activation.clear()
        
        labels_predicted = y_predicted.argmax(axis = 1)
        
        dict_activations['pred'].append(labels_predicted.detach().cpu())

        results = labels_predicted == label

        dict_activations['results'].append(results.detach().cpu())
        
    return dict_activations
    
def fit_empirical_posteriors(dict_activations_train=None, dict_peephole_train=None, n_classes=None):
    """Fits the empirical posteriors.

    The empirical posteriors map clustering models' cluster assignments to
    labels.

    Args:

        dict_activations: a dictionary containing all the activations of the network generated by specific 
                          inputs, their ground-truth labels, the resulting classifications and a boolean 
                          variable that identifies if the classification was correct or incorrect
        
        clustering_config: a dictionary, whose keys are the layers we are analysing through clustering.
                           The element for each key correspond to a list containg the number of clusters
                           desired and the clustering method that has been used to analyse the peephole 
                           at that level of the network
        
        n_classes: the total number of classes associated to the classification algorithm
    
    Returns:

        empirical_posteriors: a list of length equal to the number of layers defined in clustering_config.
                              Each elemnt of the list is a matrix (n_classes, n_clusters), where n_classes 
                              is one of the arguments required by the function fit_empirical_posterior and n_clusters 
                              corresponds to the first element of the list that is stored in the dictionary clustering_config
                              
    """

    # for each activation, empirical_posteriors is a mapping from the cluster
    # assignment to the classes.
    empirical_posteriors = [
        np.zeros((dict_peephole_train['clustering_config'][key].n_clusters, n_classes))
        for key in dict_peephole_train['clustering_config'].keys()
    ]

    print('fitting empirical posteriors')

    for i, key in enumerate(dict_peephole_train['clustering_config'].keys()):
        # cluster assignment of the i-th activation
        clustering_assignment = dict_peephole_train['clustering_config'][key].labels_
        
        # this is a numpy array of shape (n_clusters, n_classes)
        h = empirical_posteriors[i]
        n_clusters = dict_peephole_train['clustering_config'][key].n_clusters
        
        for j in range(n_clusters):
            
            # indices of training examples that got assigned to cluster j
            idx = np.argwhere(clustering_assignment == j)[:, 0]
    
            # get counts of labels among examples in the j-th cluster.
            labels_ = np.concatenate(np.array(dict_activations_train['label']))
            
            counter = collections.Counter(labels_[idx])
            
            h[j] = [counter.get(k, 0.0) for k in range(n_classes)]
            
    
            # normalize row sums
            h[j] /= np.sum(h[j])

    return empirical_posteriors

    # empirical_posteriors_path = os.path.join(self.work_dir,
    #                                          'empirical_posteriors')
    # if not os.path.exists(empirical_posteriors_path):
    #   os.makedirs(empirical_posteriors_path)
    # joblib.dump(self.empirical_posteriors, os.path.join(
    #     empirical_posteriors_path, 'empirical_posteriors.joblib'))
    

def empirical_posterior_computation(dict_activations=None, n_classes=None, dict_SVD=None, dim=None,n_clusters=None):

    """
    Computation of the empirical posterior

    The following function to compute the required empirical posterior given as input the activations, the clustering configuration
    for each layer and the matrices used for the extraction of the peephole

    Args:

    dict_activations : this dictionary contains the TRAINING DATASET, the corresponding labels, the predictions, a boolean variable
                       that is True in the case of concordance and False in case of discordance and all the required activations
    clustering_config : it is a dict that contains for each layer the number of clusters associated to each clustering model
                        that is used to analyse the peephole
    dict_V : it is a dictionary that is indexed by the name of the selected layers and each element of the dict contains the list 
             [U, s, Vh] obtained from the function get_SVD
    keys : a list containing all the layer on which we are performing the analysis
    dim: int that identifies the dimension of the peepholes

    Returns:

        empirical_posterior : a list of length equal to the number of layer we are analysing.
                              each element of the list corresponds to a matrix of dimension 
                              (n_clusters, n_classes) that identifies within each cluster the
                              total number of elements assigned to a specific class.
        dict_peephole_train : the dictionary has two keys: 'peephole', 'label' and 'clustering_config'. Each key contains a dict 
                              keys corresponding to tha layer we are interested in. 
    
    """
    dict_peephole_train = get_dict_peephole_train(dim=dim,
                                               dict_activations_train=dict_activations,
                                               n_clusters=n_clusters,
                                               dict_SVD=dict_SVD
                                              )
    
    
    empirical_posterior = fit_empirical_posteriors(dict_activations_train=dict_activations, 
                                                   dict_peephole_train=dict_peephole_train,  
                                                   n_classes=n_classes)
    
    return empirical_posterior, dict_peephole_train
    
    


def get_params_gm(X, pred_labels):
    '''
    Computes the parameters (centroids, covariances, and weights) of a 
    Gaussian Mixture based on predicted cluster labels.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Input data matrix containing the samples.

    pred_labels : array-like, shape (n_samples,)
        Predicted cluster labels for each sample in X.

    Returns:
    --------
    centroids : dict
        A dictionary containing the centroids of each cluster identified 
        by the cluster labels.

    covariances : dict
        A dictionary containing the covariance matrices of each cluster 
        identified by the cluster labels.

    weights : dict
        A dictionary containing the weights (prior probabilities) of each 
        cluster identified by the cluster labels.
    '''
    centroids = {}
    covariances = {}
    weights = {}

    unique_labels = np.unique(pred_labels)

    # compute centroids
    for cl in unique_labels:
        data = X[pred_labels == cl]
        centroids[cl] = np.mean(data, axis=0)

    # compute covariances
    for cl in unique_labels:
        data = X[pred_labels == cl]
        centered_data = data - centroids[cl]
        covariances[cl] = np.dot(centered_data.T, centered_data) / (len(data) - 1)

    # compute weights
    total_samples = len(X)
    for cl in unique_labels:
        num_samples_in_cluster = np.sum(pred_labels == cl)
        weights[cl] = num_samples_in_cluster / total_samples

    return centroids, covariances, weights

def mahalanobis_distance_(point, mean, covariance_inv):
    """
    Compute the Mahalanobis distance between a point and a multivariate 
    normal distribution.
    
    Parameters:
        point: numpy array, the data point vector.
        mean: numpy array, the mean vector of the distribution.
        covariance: numpy array, the covariance matrix of the distribution.
        
    Returns:
        The Mahalanobis distance.
    """
    deviation = point - mean
    # covariance_inv = np.linalg.inv(covariance)
    mahalanobis_dist = np.sqrt(np.dot(deviation.T, np.dot(covariance_inv, deviation)))
    return mahalanobis_dist

def mahalanobis_distance_V2(point, means, covariances_inv):
    """
    THIS VERSION IS USED IN CASE NO EMPIRICAL POSTERIOR IS USED 
    Compute the Mahalanobis distance between a point and a multivariate 
    normal distribution.
    
    Parameters:
        point: numpy array, the data point vector.
        mean: numpy array, the mean vector of the distribution.
        covariance: numpy array, the covariance matrix of the distribution.
        
    Returns:
        The Mahalanobis distance.
    """
    num_clusters = len(means.keys())
    cov_inv = {k: np.linalg.inv(v) for k, v in covariances_inv.items()}
    diff = np.array([point - means[k] for k in range(num_clusters)])
    
    mahalanobis_dist = np.array([np.dot(diff[k].T, np.dot(cov_inv[k], diff[k])) for k in range(num_clusters)])

    return mahalanobis_dist

def membership_prob(point, means, covariances, weights):
    '''
    Compute the membership probabilities for a given data point across different clusters
    using a Gaussian Mixture Model (GMM).

    Parameters:
    -----------
    point : array-like, shape (n_features,)
        The data point for which to compute the membership probabilities.

    means : dict
        A dictionary where each key is a cluster label and each value is the mean vector 
        of that cluster.

    covariances : dict
        A dictionary where each key is a cluster label and each value is the covariance 
        matrix of that cluster.

    weights : dict
        A dictionary where each key is a cluster label and each value is the weight 
        (prior probability) of that cluster.

    Returns:
    --------
    prob : numpy array, shape (n_clusters,)
        An array of membership probabilities for the given data point, where each element 
        represents the probability that the data point belongs to the corresponding cluster.

    Raises:
    -------
    AssertionError
        If the sum of the membership probabilities does not approximately equal 1, 
        an AssertionError is raised.

    Notes:
    ------
    This function calculates the membership probabilities based on the Gaussian Mixture 
    Model (GMM) parameters using the following steps:
    1. Compute the probability density function (PDF) for the given point in each cluster.
    2. Compute the weighted sum of these PDFs across all clusters.
    3. Normalize the probabilities to ensure they sum up to 1.
    '''
    num_clusters = len(means)
    d = len(point)

    # Precompute the inverse of covariance matrices and determinants for each cluster
    cov_inv = {k: np.linalg.inv(v) for k, v in covariances.items()}
    det_cov = {k: np.linalg.det(v) for k, v in covariances.items()}
    
    
    # Precompute the coefficients

    coeff = {k: 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_cov[k])) for k in range(num_clusters)}
    
    # Compute the exponent part for all clusters in a vectorized manner
    diff = np.array([point - means[k] for k in range(num_clusters)])
    mahalanobis_dist = np.array([np.dot(diff[k].T, np.dot(cov_inv[k], diff[k])) for k in range(num_clusters)])
    
    # Compute the probability density function (PDF) for all clusters
    pdf = np.array([coeff[k] * np.exp(-0.5 * mahalanobis_dist[k]) for k in range(num_clusters)])
    
    # Compute the weighted probabilities
    weighted_pdf = np.array([weights[k] * pdf[k] for k in range(num_clusters)])
    
    # Sum of the weighted PDFs
    sum_weighted_pdf = np.sum(weighted_pdf)
    
    # Normalize to get membership probabilities
    prob = weighted_pdf / sum_weighted_pdf # need sofmin?
    
    # Ensure the probabilities sum up to 1
    assert np.isclose(prob.sum(), 1), "The probabilities should sum up to 1"
    return prob

def get_distances_prob(dict_peephole_val, dict_peephole_train, method, dim):
 
    """
    The following function is able to compute the conditional probability Pr{c=k|p}, 
    which corresponds to the probability to belong to cluter k given the peephole p.
    This probability is computed for each peephole that we obtain for a specific input
    generated from this level of the network.
    The function recalls the custom fuction 'membership_prob' that computes Pr{c=k|p} based
    on the Bayesian rule:
    Pr{c=k|p}=[Pr{p|c=k}*Pr{c=k}]/Pr{p}
    where:

        - Pr{p|c=k} which corresponds to the conditional probability to observe the given 
          peephole if we are in the k-th cluster. Therefore, this factor corresponds to the 
          likelihood of the clustering model
        - Pr{c=k} which is equal to the probability to belong the k-th cluster. This element
          corresponds to the prior probability
        - Pr{p} which is equal to a regularization term

    Args:
        - dict_peephole_val : a dictionary that contains two keys: 'peephole' and 'clustering_config'
                              Each key is associated to a dictionary both sharing the same keys, which 
                              are corresponding to the layer we are considering in the surrogate model
        - method : it is a string that contains the clustering method that we use
        - dim : it's an int that defines the dimensionality of the peepholes

    """

    distances_prob = []

    if dict_peephole_val == None:

        for key in dict_peephole_train['clustering_config'].keys():

            X_train = dict_peephole_train['peephole'][key][0]

            clustering = dict_peephole_train['clustering_config'][key]

            pred_labels = clustering.predict(X_train)

            #print(pred_labels.shape)

            if method == 'GMM':

                centroids, covariances, weights = get_params_gm(X_train, pred_labels)

            else:

                centroids, _, weights = get_params_gm(X_train, pred_labels)

                covariances = {i: np.identity(dim) for i in centroids.keys()}  

            scores = []

            for point in X_train:

                scores.append(membership_prob(point, centroids, covariances, weights))

            scores_ = np.array(scores)

            distances_prob.append(scores_)

    else:

        for key in dict_peephole_val['clustering_config'].keys():

            X_train = dict_peephole_train['peephole'][key][0]

            X_val = dict_peephole_val['peephole'][key]

            clustering = dict_peephole_train['clustering_config'][key]

            pred_labels = clustering.predict(X_train)

            #print(pred_labels.shape)

            if method == 'GMM':

                centroids, covariances, weights = get_params_gm(X_train, pred_labels)

            else:

                centroids, _, weights = get_params_gm(X_train, pred_labels)

                covariances = {i: np.identity(dim) for i in centroids.keys()}  

            scores = []

            for point in X_val:

                scores.append(membership_prob(point, centroids, covariances, weights))

            scores_ = np.array(scores)

            distances_prob.append(scores_)

    return distances_prob

def p(example_score, empirical_posterior=None, weights=None, n_classes=None):
  # example_score is a list of lentgh n_activations, for one data example
  # each element is a vector of shape (n_clusters,)
    
    result = np.zeros(n_classes)
  
    # iterate through activations and sum up the contributions to probability.
    for s, h, w in zip(example_score, empirical_posterior, weights):
      # output probability based on one activation's clustering
        activation_prob = np.matmul(s, h)
        result += np.array([w*a for a in activation_prob])
    return result 

def predict_proba(weights=None, empirical_posterior=None, clustering_labels=None, n_classes=None):
    """Returns surrogate model's predicted probabilities.

    Args:
      features: a batch of input data that the baseline model can process.
      activations_dict: Python dict of cached activations created by calling
        self.get_activations.
      weights: a list of weights (need not be normalized), one for each
        activation.
    """
    # Equal weights are used if not provided by the user.
    if weights is None:
        weights = [1.0] * len(activation_names)

    pred = []
    for one_hot_encoding in clustering_labels:
        # print(one_hot_encoding)
        # print(len(one_hot_encoding))
        pred.append(p(example_score=one_hot_encoding, 
                      empirical_posterior=empirical_posterior, 
                      weights=weights, n_classes=n_classes))
    return np.array(pred)
    


def get_distances(dict_peephole_val, dict_peephole_train, method, dim):
 
    """
    The following function is able to compute the conditional probability Pr{c=k|p}, 
    which corresponds to the probability to belong to cluter k given the peephole p.
    This probability is computed for each peephole that we obtain for a specific input
    generated from this level of the network.
    The function recalls the custom fuction 'membership_prob' that computes Pr{c=k|p} based
    on the Bayesian rule:
    Pr{c=k|p}=[Pr{p|c=k}*Pr{c=k}]/Pr{p}
    where:

        - Pr{p|c=k} which corresponds to the conditional probability to observe the given 
          peephole if we are in the k-th cluster. Therefore, this factor corresponds to the 
          likelihood of the clustering model
        - Pr{c=k} which is equal to the probability to belong the k-th cluster. This element
          corresponds to the prior probability
        - Pr{p} which is equal to a regularization term

    Args:
        - dict_peephole_val : a dictionary that contains two keys: 'peephole' and 'clustering_config'
                              Each key is associated to a dictionary both sharing the same keys, which 
                              are corresponding to the layer we are considering in the surrogate model
        - method : it is a string that contains the clustering method that we use
        - dim : it's an int that defines the dimensionality of the peepholes

    """

    distances_prob = []

    if dict_peephole_val == None:

        for key in dict_peephole_train['clustering_config'].keys():

            X_train = dict_peephole_train['peephole'][key][0]

            clustering = dict_peephole_train['clustering_config'][key]

            pred_labels = clustering.predict(X_train)


            if method == 'GMM':

                centroids, covariances, weights = get_params_gm(X_train, pred_labels)

            else:

                centroids, _, weights = get_params_gm(X_train, pred_labels)

                covariances = {i: np.identity(dim) for i in centroids.keys()} 

            cov_inv = {k: np.linalg.inv(v) for k, v in covariances.items()}

            scores = []

            for point in X_train:

                scores.append(mahalanobis_distance_V2(point, centroids, cov_inv))

            scores_ = np.array(scores)

            distances_prob.append(scores_)

    else:

        for key in dict_peephole_val['clustering_config'].keys():

            X_train = dict_peephole_train['peephole'][key][0]

            X_val = dict_peephole_val['peephole'][key]

            clustering = dict_peephole_train['clustering_config'][key]

            pred_labels = clustering.predict(X_train)

            if method == 'GMM':

                centroids, covariances, weights = get_params_gm(X_train, pred_labels)

            else:

                centroids, _, weights = get_params_gm(X_train, pred_labels)

                covariances = {i: np.identity(dim) for i in centroids.keys()}  

            cov_inv = {k: np.linalg.inv(v) for k, v in covariances.items()}
            
            scores = []

            for point in X_val:

                scores.append(mahalanobis_distance_V2(point, centroids, cov_inv))

            scores_ = np.array(scores)

            distances_prob.append(scores_)

    return distances_prob
