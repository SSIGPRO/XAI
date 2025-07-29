import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.transforms import vgg16_imagenet as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd, conv2d_kernel_svd

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors, trim_kernel_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.utils.samplers import random_subsampling 

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
    ds_path = '/srv/newpenny/dataset/ImageNet_torchvision'

    # model parameters
    dataset = 'ImageNet' 
    seed = 29
    bs = 512 
    n_threads = 1

    # model_dir = '/srv/newpenny/XAI/models'
    # model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_path = Path.cwd()/f'../data/{dataset}'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/f'../data/{dataset}/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/f'../data/{dataset}/drillers'
    drill_name = 'classifier'
    
    verbose = True 

    classifier_svd_rank = 1000
    
    # Peepholelib
    target_layers = [
            # 'features.24',
            # 'features.26',
            # 'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]
    
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = ImageNet(
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
    
    nn = vgg16(weights='IMAGENET1K_V1')
    n_classes = len(ds.get_classes()) 
    model = ModelWrap(
            model = nn,
            device = device
            )

    model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )

    #--------------------------------
    # SVDs 
    #--------------------------------
    svd_fns = {
            'classifier.0': partial(
                linear_svd,
                rank = classifier_svd_rank,
                device = device,
                ),
            'classifier.3': partial(
                linear_svd,
                rank = classifier_svd_rank,
                device = device,
                ),
            'classifier.6': partial(
                linear_svd,
                rank = classifier_svd_rank,
                device = device,
                ),
            }

    t0 = time()
    model.get_svds(
            path = svds_path,
            name = svds_name,
            target_modules = target_layers,
            sample_in = ds._dss['train'][0][0],
            svd_fns = svd_fns,
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
    random_subsampling(ds, 0.3)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    reduction_fns = {
            'classifier.0': partial(
                linear_svd_projection,
                svd = model._svds['classifier.0'], 
                use_s = True,
                device=device
                ),
            'classifier.3': partial(
                linear_svd_projection,
                svd = model._svds['classifier.3'], 
                use_s = True,
                device=device
                ),
            'classifier.6': partial(
                linear_svd_projection,
                svd = model._svds['classifier.6'], 
                use_s = True,
                device=device
                ),
            }
    
    with corevecs as cv: 
        cv.load_only(loaders=['train'], verbose=True)
        plt.imshow(cv._dss['train'][0].detach().cpu().numpy().transpose(1,2,0))
        plt.savefig('prova.png')
        quit()
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
        
        fig, axs = plt.subplots(1,2, figsize=(12, 6))
        axs[0].hist(cv._dss['train']['label'], bins=1000, color='blue', alpha=0.7)
        axs[0].set_title('Train Labels Distribution')
        axs[0].set_xlabel('Labels')
        axs[0].set_ylabel('Frequency')
        axs[1].hist(cv._dss['train']['label'][cv._dss['train']['result']==0], bins=1000, color='red', label='W',alpha=0.7)
        axs[1].hist(cv._dss['train']['label'][cv._dss['train']['result']==1], bins=1000, color='green', label='C',alpha=0.7)
        axs[1].set_title('Correct and Wrong Labels Distribution')
        axs[1].set_xlabel('Labels')
        axs[1].set_ylabel('Frequency')
        axs[1].legend()
        plt.tight_layout()
        plt.savefig((cvs_path/(cvs_name+'.labels_distribution.png')).as_posix(), dpi=300, bbox_inches='tight')
        plt.close() 
       
    #--------------------------------
    # Peepholes
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    classifier_cv_dim = 100
    n_cluster = 500

    cv_parsers = {
            # 'features.24': partial(
            #     trim_kernel_corevectors,
            #     module = 'features.24',
            #     cv_dim = features24_cv_dim
            #     ),
            # 'features.26': partial(
            #     trim_channelwise_corevectors,
            #     module = 'features.26',
            #     cv_dim = features26_cv_dim
            #     ),
            # 'features.28': partial(
            #     trim_corevectors,
            #     module = 'features.28',
            #     cv_dim = features28_cv_dim
            #     ),
            'classifier.0': partial(
                trim_corevectors,
                module = 'classifier.0',
                cv_dim = classifier_cv_dim
                ),
            'classifier.3': partial(
                trim_corevectors,
                module = 'classifier.3',
                cv_dim = classifier_cv_dim
                ),
            'classifier.6': partial(
                trim_corevectors,
                module = 'classifier.6',
                cv_dim = classifier_cv_dim
                ),
            }

    feature_sizes = {
            'classifier.0': classifier_cv_dim,
            'classifier.3': classifier_cv_dim,
            'classifier.6': classifier_cv_dim,
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

    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'val'],
                verbose = True
                ) 

        for drill_key, driller in drillers.items():
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(corevectors = cv._corevds['train'], verbose=verbose)
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                driller.compute_empirical_posteriors(
                        dataset = cv._dss['train'],
                        corevectors = cv._corevds['train'],
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()
    
