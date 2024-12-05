### EXPERIMENTING WITH DkNN

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ['SCIPY_USE_PROPACK'] = "True"
 
threads = "32"
os.environ["OMP_NUM_THREADS"] = threads
os.environ["OPENBLAS_NUM_THREADS"] = threads
os.environ["MKL_NUM_THREADS"] = threads
os.environ["VECLIB_MAXIMUM_THREADS"] = threads
os.environ["NUMEXPR_NUM_THREADS"] = threads

# python stuff

from pathlib import Path as Path
from numpy.random import randint
from tqdm import tqdm

# Our stuff
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap
from coreVectors.coreVectors import CoreVectors 
from coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn

from credibility.DkNN import NearestNeighbor, DkNN

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")
    
    #--------------------------------
    # Dataset 
    #--------------------------------
    # model parameters
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64
    data_path = f'/srv/newpenny/data/{dataset}'
    
    ds = Cifar(
                data_path = '/srv/newpenny/dataset/CIFAR100',
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
    pretrained = True
    model_dir = '/srv/newpenny/XAI/models'
    model_ = 'vgg16'
    # model_name = f'{model_}_pretrained={pretrained}_dataset={dataset}-'\
    # f'augmented_policy=CIFAR10_bs={bs}_seed={seed}.pth'
    
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    num_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    model = ModelWrap(device=device)
    model.set_model(model=nn, path=model_dir, name=model_name, verbose=True)
    target_layers = ['classifier.0', 
                     'classifier.3', 
                     'features.7',
                     'features.14',
                     # 'features.24', 
                     # 'features.26', 
                     'features.28']
    model.set_target_layers(target_layers=target_layers, verbose=True)
    
    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)
    
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
    cvs_name = 'corevectors'
    cvs_path = Path.cwd()/f'../data/corevectors__/{dataset}'
    
    corevectors = CoreVectors(
                        path = cvs_path,
                        name = cvs_name,
                        model = model
                        )
    
    act_name = 'corevectors.activations'
    act_path = Path.cwd()/f'../data/corevectors__/{dataset}'
    
    activations = CoreVectors(
                        path = act_path,
                        name = act_name,
                        model = model
                        )
    
    loaders = ds.get_dataset_loaders()
    
    #--------------------------------
    # DkNN
    #--------------------------------
    
    nb_classes = ds.config['num_classes']
    neighbors = 75
    percentage = {'train':100,
                   'val':10,
                   'test':1}
    
    verbose = True
    
    #dknn_path = '/srv/newpenny/XAI/generated_data/DkNN'
    dknn_path = Path.cwd()/'../data/DkNN'
    dknn_name = f'DkNN_dnn=_dataset={dataset}' 
    nb_tables = 200
    number_bits = 17
    
    with corevectors as cv, activations as act:
        # copy dataset to coreVect dataset
        cv.get_coreVec_dataset(
                loaders = loaders,
                verbose = True
                ) 
        
        cv.get_activations(
                loaders=loaders,
                verbose=True
                )
        cv.load_only(
                    loaders = ['train', 'test', 'val'],
                    verbose = True,
                    load_type = cvs_name 
                    ) 
    
        cv_dl = cv.get_dataloaders(
                batch_size = bs,
                verbose = True,
                )
        
        act.load_only(
                    loaders = ['train', 'test', 'val'],
                    verbose = True,
                    load_type = act_name 
                    ) 
        
        act_dl = act.get_activations_loaders(verbose=True)
        
        kwargs = {'model' : model,
              'nb_classes' : nb_classes,
              'neighbors' : neighbors,
              'cv_dl' : cv_dl,
              'act_dl' : act_dl,
              'percentage' : percentage, 
              'seed' : seed,
              'device' : device,
              'verbose' : verbose,
              'path' : dknn_path,
              'name' : dknn_name,
              'nearest_neighbor_backend' : NearestNeighbor.BACKEND.FALCONN,
              'nb_tables' : nb_tables,
              'number_bits' : number_bits,
            }
        dknn = DkNN(**kwargs)
        dknn.calibrate() 
        dknn.fprop('all')