import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import evaluate, evaluate_dists 

# torch stuff
import torch
from torchvision.models import vgg16
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
    dataset = 'CIFAR100' 
    seed = 29
    bs = 128 
    n_threads = 32

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
    
    # Peepholelib
    target_layers = [
            'classifier.0',
            'classifier.3',
            'features.24',
            'features.26',
            'features.28',
            ]
    
    n_cluster = 10
    cv_dim = 5

    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset = dataset
            )

    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16()
    n_classes = len(ds.get_classes()) 
    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'classifier.6', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )

    model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )
    
    #--------------------------------
    # SVDs 
    #--------------------------------
    t0 = time()
    model.get_svds(
            target_modules = target_layers,
            path = svds_path,
            rank = 10,
            channel_wise = True,
            name = svds_name,
            verbose = verbose
            )
    print('time: ', time()-t0)
    
    '''
    print('\n----------- svds:')
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)
        s = model._svds[k]['s']
        if len(s.shape) == 1:
            plt.figure()
            plt.plot(s, '-')
            plt.xlabel('Rank')
            plt.ylabel('EigenVec')
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            _x = torch.linspace(0, s.shape[1]-1, s.shape[1])
            for r in range(s.shape[0]):
                plt.plot(xs=_x, ys=s[r,:], zs=r, zdir='y')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Channel')
            ax.set_zlabel('EigenVec')
        plt.savefig((svds_path/(svds_name+'/'+k+'.png')).as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
    '''
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #random_subsampling(ds, 0.025)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    reduction_fns = {
            'classifier.0': partial(
                svd_Linear,
                reduct_m=model._svds['classifier.0']['Vh'], 
                device=device
                ),
            'classifier.3': partial(
                svd_Linear,
                reduct_m=model._svds['classifier.3']['Vh'], 
                device=device
                ),
            'features.24': partial(
                svd_Conv2D, 
                reduct_m=model._svds['features.24']['Vh'], 
                layer=model._target_modules['features.24'], 
                device=device
                ),
            'features.26': partial(
                svd_Conv2D, 
                reduct_m=model._svds['features.26']['Vh'], 
                layer=model._target_modules['features.26'], 
                device=device
                ),
            'features.28': partial(
                svd_Conv2D, 
                reduct_m=model._svds['features.28']['Vh'], 
                layer=model._target_modules['features.28'], 
                device=device
                ),
            }
    
    with corevecs as cv: 
        cv.parse_ds(
                batch_size = bs,
                datasets = ds,
                n_threads = n_threads,
                verbose = verbose
                )

        '''
        # This occupies a lot of space. Only do if you need it
        # copy dataset to activatons file
        cv.get_activations(
                batch_size = bs,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                )        
        '''

        # computing the corevectors
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                )

        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'train',
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose=verbose
                    )

    #--------------------------------
    # Peepholes
    #--------------------------------

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
    for peep_layer in target_layers:
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
            device = device
            )

    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        for drill_key, driller in drillers.items():
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key} time = ', time()-t0)
                driller.fit(corevectors = cv._corevds['train'], verbose=verbose)

                driller.compute_empirical_posteriors(
                        dataset = cv._dss['train'],
                        corevectors = cv._corevds['train'],
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        ph.get_peepholes(
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose
                )

        ph.get_scores(
            batch_size = bs,
            verbose=verbose
            )

        evaluate_dists(
                peepholes = ph,
                score_type = 'max',
                dataset = cv._dss,
                bins = 20
                )
