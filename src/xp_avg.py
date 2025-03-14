import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from sklearn import covariance
import numpy as np
from tqdm import tqdm

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv
from peepholelib.classifier.classifier_base import trim_corevectors
from peepholelib.classifier.tkmeans import KMeans as tKMeans 
from peepholelib.classifier.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from tensordict import TensorDict
from tensordict import MemoryMappedTensor as MMT

def sample_estimator(n_classes, target_layers, act_dl, cv_dl, device):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    
    group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(target_layers)
    num_sample_per_class = np.empty(n_classes)
    num_sample_per_class.fill(0)
#     _features = torch.zeros((n_classes, target_layers))
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(n_classes):
            temp_list.append(0)
        list_features.append(temp_list) ## it creats a list of dim num_output, n_classes
    
    for acts, cvs in tqdm(zip(act_dl, cv_dl)): # we have a structure that correspond 
                                               # to a dict whose keys are the layers 
                                               # in target_layers and zip with acivation
        
        # construct the sample matrix
        for act, cv in zip(acts, cvs): # they iterate over the batch
            
            label = int(act['label']) 
            
            if num_sample_per_class[label] == 0:
                out_count = 0
                for i, layer in enumerate(target_layers):## it iterates over the layer
                    list_features[out_count][label] = cv[layer][i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for i, layer in enumerate(target_layers):
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], cv[layer][i].view(1, -1)), 0)
                    out_count += 1                
            num_sample_per_class[label] += 1
            
    sample_class_mean = []
    out_count = 0
    for i, (layer, data) in enumerate(cv_dl.dataset.items()):
        _size = data.size(1)
        temp_list = torch.Tensor(n_classes, _size).cuda()
        for j in range(n_classes):
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1
    precision = []
    for k in range(num_output):
        X = 0
        for i in range(n_classes):
            if i == 0:
                X = list_features[k][i].to(device) - sample_class_mean[k][i].to(device)
                print(X)

            else:
                X = torch.cat((X, list_features[k][i].to(device) - sample_class_mean[k][i]), 0)       
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision.append(temp_precision)
        
    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision


if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_path = Path.cwd()/'../data'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'coreavg'

    act_path = Path.cwd()/'../data/corevectors'
    act_name = 'activations'
    
    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'
    
    verbose = True 
    
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset=dataset
            )
    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    n_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    model = ModelWrap(device=device)
    model.set_model(model=nn, path=model_dir, name=model_name, verbose=verbose)

    target_layers = [
        #     'classifier.0',
            # 'classifier.3',
            #'features.7',
            'features.14',
            'features.28',
            ]
    model.set_target_layers(target_layers=target_layers, verbose=verbose)

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._dss['train'][0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #dss = ds._dss
    dss = random_subsampling(ds._dss, 0.05)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # for each layer we define the function used to perform dimensionality reduction
    reduction_fns = {'features.14': ChannelWiseMean_conv,
                     'features.28': ChannelWiseMean_conv,
                #      'classifier.0': lambda x: x,
                     }
    
    shapes = {'features.14': 256,
              'features.28': 512,
        #       'classifier.0': 25088,
              }
    
    with corevecs as cv: 
        # copy dataset to coreVect dataset

        cv.get_activations(
                batch_size = bs,
                datasets = dss,
                verbose = verbose
                )
        
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                shapes = shapes,
                verbose = verbose
                )

        cv_dl = cv.get_dataloaders(verbose=verbose)
    
        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['train']:
            print(data['features.14'].shape)
            print(data['features.14'][34:56,:])
            i += 1
            if i == 3: break
    ##-----------------------
    # TESTING The adversaries
    ##-----------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    activations = CoreVectors(
                path = act_path,
                name = act_name,
                )

    with corevecs as cv, activations as act:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        cv_dl = cv.get_dataloaders(
                batch_size = bs,
                verbose = True,
                )
        
        act.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        act_dl = act.get_dataloaders(
                 batch_size = bs,
                 verbose = True,
                 )
        sample_class_mean, precision = sample_estimator(n_classes=n_classes, 
                                                        target_layers=target_layers, 
                                                        act_dl=act_dl['train'], 
                                                        cv_dl=cv_dl['train'],
                                                        device=device)
        print(sample_class_mean, precision)