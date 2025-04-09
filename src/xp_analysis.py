import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.analyze import evaluate, evaluate_dists 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda


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
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes'
    phs_name = 'peepholes'
    
    verbose = True 
    #--------------------------------
    # Model
    #--------------------------------
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    n_classes = 100
    nn.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    model = ModelWrap(
            model=nn,
            path=model_dir,
            name=model_name,
            device=device
            )
    model.load_checkpoint(verbose=verbose)
                                                                            
    target_layers = [
            'classifier.0',
            'classifier.3',
            'features.24',
            'features.26',
            'features.28',
            ]
    model.set_target_modules(target_modules=target_layers, verbose=verbose)
                                                                            
    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    #dry_img, _ = ds._dss['train'][0]
    #dry_img = dry_img.reshape((1,)+dry_img.shape)
    #model.dry_run(x=dry_img)
                                                                            
    #--------------------------------
    # SVDs 
    #--------------------------------
    print('target modules: ', model.get_target_modules()) 
    t0 = time()
    model.get_svds(
            target_modules = target_layers,
            path = svds_path,
            rank = 300,
            channel_wise = True,
            name = svds_name,
            verbose = verbose
            )

    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100
    n_cluster = 100
    cv_dim = 50
    peep_layers = [
            'classifier.0',
            'classifier.3',
            'features.24',
            'features.26',
            'features.28',
            ]
    
    cls_kwargs = {}#{'batch_size': bs} 

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    cv_parsers = {
            'classifier.0': partial(
                trim_corevectors,
                module = 'classifier.0',
                cv_dim = cv_dim
                ),
            'classifier.3': partial(
                trim_corevectors,
                module = 'classifier.3',
                cv_dim = cv_dim
                ),
            'features.24': partial(
                trim_channelwise_corevectors,
                module = 'features.24',
                cv_dim = cv_dim
                ),
            'features.26': partial(
                trim_channelwise_corevectors,
                module = 'features.26',
                cv_dim = cv_dim
                ),
            'features.28': partial(
                trim_channelwise_corevectors,
                module = 'features.28',
                cv_dim = cv_dim
                ),
            }

    feature_sizes = {
            'classifier.0': cv_dim,
            'classifier.3': cv_dim,
            # for channel_wise corevectors, the size is n_channels * cv_dim
            'features.24': cv_dim*model._svds['features.24']['Vh'].shape[0],
            'features.26': cv_dim*model._svds['features.26']['Vh'].shape[0],
            'features.28': cv_dim*model._svds['features.28']['Vh'].shape[0],
            }
    drillers = {}
    for peep_layer in peep_layers:
        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = cv_parsers[peep_layer],
                device = device
                )

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            driller = drillers,
            target_modules = peep_layers,
            device = device
            )
    
    # fitting classifiers
    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        ph.load_only(
                loaders = ['train', 'test', 'val'],
                batch_size = bs,
                verbose = verbose
                )

        evaluate_dists(
                peepholes = ph,
                activations = cv._actds,
                score_type = 'max',
                bins = 20
                )
        
        evaluate_dists(
                peepholes = ph,
                activations = cv._actds,
                score_type = 'entropy',
                bins = 20
                )

        evaluate(
                peepholes = ph,
                corevectors = cv,
                score_type = 'max',
                bins = 20
                )

        evaluate(
                peepholes = ph,
                corevectors = cv,
                score_type = 'entropy',
                bins = 20
                )
