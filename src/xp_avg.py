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

def sample_estimator(n_classes, target_layers, act_ds, cv_ds, device):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    
    group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
    num_output = len(target_layers)
    num_sample_per_class = np.zeros(n_classes)
#     _features = torch.zeros((n_classes, target_layers))
    list_features = {}
    for key in target_layers:
        list_features[key] = []
        for j in range(n_classes):
            list_features[key].append(0)
         ## it creats a list of dim num_output, n_classes
    
    for i, (act, cv) in tqdm(enumerate(zip(act_ds, cv_ds))): # we have a structure that correspond 
                                                            # to a dict whose keys are the layers 
                                                             # in target_layers and zip with acivation
            
        label = int(act['label']) 
        
        if num_sample_per_class[label] == 0:
            # out_count = 0
            for layer in target_layers:## it iterates over the layer
                list_features[layer][label] = cv[layer].view(1, -1)
                # out_count += 1
        else:
            # out_count = 0
            for layer in target_layers:
                list_features[layer][label] \
                = torch.cat((list_features[layer][label], cv[layer].view(1, -1)), 0)
                # out_count += 1                
        num_sample_per_class[label] += 1
            
    sample_class_mean = {}
    # out_count = 0
    for layer in target_layers:
        _size = cv_ds[layer].size(1)
        temp_list = torch.Tensor(n_classes, _size).cuda()
        for j in range(n_classes):
            temp_list[j] = torch.mean(list_features[layer][j], 0)
        sample_class_mean[layer] = temp_list
        # out_count += 1
    precision = {}
    for layer in target_layers:
        X = 0
        for i in range(n_classes):
            if i == 0:
                X = list_features[layer][i].to(device) - sample_class_mean[layer][i].to(device)
                print(X)

            else:
                X = torch.cat((X, list_features[layer][i].to(device) - sample_class_mean[layer][i]), 0)       
        # find inverse            
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        temp_precision = torch.from_numpy(temp_precision).float().cuda()
        precision[layer] = temp_precision


    return sample_class_mean, precision

def get_Mahalanobis_score(model, act_dl, cv_dl, num_classes, outf, out_flag, net_type, sample_mean, precision, layer, magnitude):
    '''
    Compute the proposed Mahalanobis confidence score on input dataset
    return: Mahalanobis score from layer_index It computes the score for a single layer at the time


    sample_mean is a DICT of a number of elemnts that is equal to the number of layers present in target layers. Each element is a tensor 
    of dim (n_classes, dim of the coreavg) in layer features.28 it will be(100,512)
    precision is a DICT of precision matrices one for each layer
    '''
    Mahalanobis = []
    
    if out_flag == True:
        temp_file_name = '%s/confidence_Ga%s_In.txt'%(outf, str(layer))
    else:
        temp_file_name = '%s/confidence_Ga%s_Out.txt'%(outf, str(layer))
        
    g = open(temp_file_name, 'w')
    
    for acts, cvs in tqdm(zip(act_dl, cv_dl)):

        data = acts['image']
        target = acts['label']
        out_features = cvs[layer]
        
        # compute Mahalanobis score
        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)
        
        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer].index_select(0, sample_pred)
        zero_f = out_features - batch_sample_mean
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer]), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()
         
        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2
        if net_type == 'densenet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))
        elif net_type == 'resnet':
            gradient.index_copy_(1, torch.LongTensor([0]).cuda(), gradient.index_select(1, torch.LongTensor([0]).cuda()) / (0.2023))
            gradient.index_copy_(1, torch.LongTensor([1]).cuda(), gradient.index_select(1, torch.LongTensor([1]).cuda()) / (0.1994))
            gradient.index_copy_(1, torch.LongTensor([2]).cuda(), gradient.index_select(1, torch.LongTensor([2]).cuda()) / (0.2010))
        tempInputs = torch.add(data.data, -magnitude, gradient)
 
        noise_out_features = model.intermediate_forward(Variable(tempInputs, volatile=True), layer_index)
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)      

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)
        Mahalanobis.extend(noise_gaussian_score.cpu().numpy())
        
        for i in range(data.size(0)):
            g.write("{}\n".format(noise_gaussian_score[i]))
    g.close()

    return Mahalanobis


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
                                                        act_ds=act_dl['train'].dataset, 
                                                        cv_ds=cv_dl['train'].dataset,
                                                        device=device)
    
