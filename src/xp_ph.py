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
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd, conv2d_kernel_svd

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors, trim_kernel_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.scores import conceptogram_protoclass_score as proto_score, model_confidence_score as mconf_score 
from peepholelib.utils.plots import plot_confidence, plot_calibration, plot_ood 
from peepholelib.utils.conceptograms import plot_conceptogram

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
    bs = 512 
    n_threads = 1

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

    plots_path = Path.cwd()/'temp_plots/xp_ph/'
    
    verbose = True 
    
    # Peepholelib
    target_layers = [
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]
    
    features24_svd_rank = 3 
    features26_svd_rank = 6
    features28_svd_rank = 300
    classifier_svd_rank = 300 
    n_cluster = 200
    features24_cv_dim = 2
    features26_cv_dim = 5
    features28_cv_dim = 150
    classifier_cv_dim = 150
    n_conceptograms = 10
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
    svd_fns = {
            'features.24': partial(
                conv2d_kernel_svd, 
                rank = features24_svd_rank,
                device = device,
                ),
            'features.26': partial(
                conv2d_toeplitz_svd, 
                rank = features26_svd_rank,
                channel_wise = True,
                device = device,
                ),
            'features.28': partial(
                conv2d_toeplitz_svd, 
                rank = features26_svd_rank,
                channel_wise = False,
                device = device,
                ),
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
    random_subsampling(ds, 0.025)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    reduction_fns = {
            'features.24': partial(
                conv2d_kernel_svd_projection, 
                svd = model._svds['features.24'], 
                layer = model._target_modules['features.24'], 
                use_s = True,
                device=device
                ),
            'features.26': partial(
                conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.26'], 
                layer = model._target_modules['features.26'], 
                use_s = True,
                device = device
                ),
            'features.28': partial(
                conv2d_toeplitz_svd_projection, 
                svd = model._svds['features.28'], 
                layer = model._target_modules['features.28'], 
                use_s = True,
                device = device
                ),
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
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    #loaders = ['ood-c0', 'ood-c1', 'ood-c2', 'ood-c3', 'ood-c4'],
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
            'features.24': partial(
                trim_kernel_corevectors,
                module = 'features.24',
                cv_dim = features24_cv_dim
                ),
            'features.26': partial(
                trim_channelwise_corevectors,
                module = 'features.26',
                cv_dim = features26_cv_dim
                ),
            'features.28': partial(
                trim_corevectors,
                module = 'features.28',
                cv_dim = features28_cv_dim
                ),
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
            # for channel_wise corevectors, the size is out_size * cv_dim
            # TODO: get 196 from somewhere
            'features.24': features24_cv_dim*196,
            # for channel_wise corevectors, the size is n_channels * cv_dim
            'features.26': features26_cv_dim*model._svds['features.26']['Vh'].shape[0],
            'features.28': features28_cv_dim,
            'classifier.0': classifier_cv_dim,
            'classifier.3': classifier_cv_dim,
            'classifier.6': 100#classifier_cv_dim,
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
                loaders = ['train', 'val', 'test'],
                verbose = verbose 
                ) 

        for drill_key, driller in drillers.items():
            if (driller._empp_file).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
            else:
                t0 = time()
                print(f'Fitting classifier for {drill_key}')
                driller.fit(
                        corevectors = cv,
                        loader = 'train',
                        verbose=verbose
                        )
                print(f'Fitting time for {drill_key}  = ', time()-t0)

                driller.compute_empirical_posteriors(
                        corevectors = cv,
                        loader = 'train',
                        batch_size = bs,
                        verbose=verbose
                        )
        
                # save classifiers
                print(f'Saving classifier for {drill_key}')
                driller.save()

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'val', 'test'],
                verbose = verbose 
                ) 

        ph.get_peepholes(
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = n_threads,
                verbose = verbose
                )

        # get scores
        scores, protoclasses = proto_score(
                peepholes = ph,
                corevectors = cv,
                verbose = verbose
                )
                                        
        scores = mconf_score(
                corevectors = cv,
                append_scores = scores,
                verbose = verbose
                )

        # make plots
        plot_confidence(
                corevectors = cv,
                scores = scores,
                max_score = 1.,
                path = plots_path,
                verbose = verbose
                )
                                                                                  
        plot_calibration(
                corevectors = cv,
                scores = scores,
                calib_bin = 0.1,
                path = plots_path,
                verbose = verbose
                )
        
        # plot conceptograms
        # get `n_conceptograms` random samples for each score interval
        idx_hh = (scores['test']['Proto-Class'] > 0.90).nonzero().squeeze()[:10]
        idx_mh = (torch.logical_and(scores['test']['Proto-Class']>0.7, scores['test']['Proto-Class']<0.8)).nonzero().squeeze()[:10]
        idx_mm = (torch.logical_and(scores['test']['Proto-Class']>0.45, scores['test']['Proto-Class']<0.55)).nonzero().squeeze()[:10]
        idx_ml = (torch.logical_and(scores['test']['Proto-Class']>0.2, scores['test']['Proto-Class']<0.3)).nonzero().squeeze()[:10]
        idx_ll = (scores['test']['Proto-Class']<0.1).nonzero().squeeze()[:10]
        idx = torch.hstack([
            idx_hh[torch.randperm(idx_hh.shape[0])[:n_conceptograms]],
            idx_mh[torch.randperm(idx_mh.shape[0])[:n_conceptograms]],
            idx_mm[torch.randperm(idx_mm.shape[0])[:n_conceptograms]],
            idx_ml[torch.randperm(idx_ml.shape[0])[:n_conceptograms]],
            idx_ll[torch.randperm(idx_ll.shape[0])[:n_conceptograms]]
            ]).tolist()  

        plot_conceptogram(
                path = plots_path,
                name = 'conceptogram',
                corevectors = cv,
                peepholes = ph,
                loaders = ['test'],
                samples = idx,
                target_modules = target_layers,
                classes = ds._classes,
                protoclasses = protoclasses,
                scores = scores,
                verbose = verbose,
                )
