import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from pathlib import Path as Path
from time import time
from functools import partial
from matplotlib import pyplot as plt
import detectors
import timm

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 


# torch stuff
import torch
from cuda_selector import auto_cuda
import torchvision
import torch.nn as nn

if __name__ == "__main__":
    torch.cuda.empty_cache()
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
    bs = 128
    n_threads = 32

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'vgg16_BN_dataset=CIFAR100.pth_optim=SGD_scheduler=CosineAnnealingLR.pth'

    svds_path = Path.cwd()/'data'
    svds_name = 'svds' 

    cvs_path = Path.cwd()/'data/corevectors'
    cvs_name = 'corevectors'
    
    drill_path = Path.cwd()/'data/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'data/peepholes'
    phs_name = 'peepholes'

    verbose = False

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

    nn = timm.create_model("vgg16_bn_cifar100", pretrained=True)
    print(nn.state_dict().keys())
    in_features = nn.get_classifier().in_features
    print("in features", in_features)
    nn.head.fc = torch.nn.Linear(512, len(ds.get_classes()))
    device = torch.device("cpu")
    model = ModelWrap(
        model=nn,
        path=model_dir,
        name=model_name,
        device=device
        )
    model.load_checkpoint(verbose=verbose)

    
    target_layers = ['features.14', 'features.17', 'features.20',
                    'features.24', 'features.30',
                    'features.34','features.37','features.40', 
                     ]
    
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

    direction = {'save_input':True, 'save_output':True}
    model.add_hooks(**direction, verbose=False) 
        
    dry_img, _ = ds._dss['train'][0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------
    print('target modules: ', model.get_target_modules()) 
    t0 = time()

    model.get_svds(
            target_modules = target_layers,
            path=svds_path,
            name=svds_name,
            rank=100,
            #channel_wise=False,
            verbose=verbose
            )
    print('time: ', time()-t0)
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")

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
    #quit()
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #dss = ds._dss
    dss = random_subsampling(ds._dss, 0.05)
    print(f"Subset size after subsampling: {len(dss['train'])}")
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
    )
    device = torch.device("cpu")
    reduction_fns = {
        'features.14': partial(svd_Conv2D,
            reduct_m=model._svds['features.14']['Vh'],
            layer=model._target_modules['features.14'],
            device=device
            ),
        'features.17': partial(svd_Conv2D,
            reduct_m=model._svds['features.17']['Vh'],
            layer=model._target_modules['features.17'],
            device=device
            ),
        'features.20': partial(svd_Conv2D,
            reduct_m=model._svds['features.20']['Vh'],
            layer=model._target_modules['features.20'],
            device=device
            ),
        'features.24': partial(svd_Conv2D,
            reduct_m=model._svds['features.24']['Vh'],
            layer=model._target_modules['features.24'],
            device=device
            ),
        'features.30': partial(svd_Conv2D,
            reduct_m=model._svds['features.30']['Vh'],
            layer=model._target_modules['features.30'],
            device=device
            ),
        'features.34': partial(svd_Conv2D, 
            reduct_m=model._svds['features.34']['Vh'], 
            layer=model._target_modules['features.34'], 
            device=device
            ),
        'features.37': partial(svd_Conv2D, 
            reduct_m=model._svds['features.37']['Vh'], 
            layer=model._target_modules['features.37'], 
            device=device
            ),
        'features.40': partial(svd_Conv2D, 
            reduct_m=model._svds['features.40']['Vh'], 
            layer=model._target_modules['features.40'], 
            device=device
            ),
    }

    with corevecs as cv: 
        # copy dataset to coreVect dataset
        cv.get_activations(
            batch_size = bs,
            datasets = dss,
            n_threads = n_threads,
            verbose = verbose
        )
         
        cv.get_coreVectors(
            batch_size = bs,
            reduction_fns = reduction_fns,
            n_threads = n_threads,
            verbose = verbose
        )

        cv_dl = cv.get_dataloaders(verbose=verbose)
        device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")

        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['train']:
            print('\nfeatures.34')
            print(data['features.34'][34:56,:])
            i += 1
            if i == 1: break
        
        cv.normalize_corevectors(
                wrt='train',
                #from_file=cvs_path/(cvs_name+'.normalization.pt'),
                to_file=cvs_path/(cvs_name+'.normalization.pt'),
                batch_size=bs,
                n_threads=n_threads,
                verbose=verbose
                )
        
        i = 0
        print('after norm')
        for data in cv_dl['test']:
            print(data['features.34'][34:56,:])
            i += 1
            if i == 1: break

    #--------------------------------
    # Peepholes
    #--------------------------------

    n_classes = 100
    n_cluster = 100
    cv_dim = 10
    peep_layers = ['features.14',
                   'features.17',
                   'features.20',
                   'features.24',
                   'features.30',
                    'features.34',
                   'features.37',
                   'features.40',
                ]
    

    cls_kwargs = {}#{'batch_size':256} 

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        )
    
    cv_parsers = {
        'features.14': partial(trim_corevectors,
                          module = 'features.14',
                          cv_dim = cv_dim),
        'features.17': partial(trim_corevectors,
                          module = 'features.17',
                          cv_dim = cv_dim),
        'features.20': partial(trim_corevectors,
                          module = 'features.20',
                          cv_dim = cv_dim),
        'features.24': partial(trim_channelwise_corevectors,
                          module = 'features.24',
                          cv_dim = cv_dim),
        'features.30': partial(trim_channelwise_corevectors,
                          module = 'features.30',
                          cv_dim = cv_dim),
        'features.34': partial(trim_channelwise_corevectors,
                          module = 'features.34',
                          cv_dim = cv_dim),
        'features.37': partial(trim_channelwise_corevectors,
                            module = 'features.37',
                            cv_dim = cv_dim),
        'features.40': partial(trim_channelwise_corevectors,
                          module = 'features.40',
                          cv_dim = cv_dim),
                                      
    }
    
    feature_sizes = {
            'features.14': cv_dim*model._svds['features.14']['Vh'].shape[0],
            'features.17': cv_dim*model._svds['features.17']['Vh'].shape[0],
            'features.20': cv_dim*model._svds['features.20']['Vh'].shape[0],
            'features.24': cv_dim*model._svds['features.24']['Vh'].shape[0],
            'features.30': cv_dim*model._svds['features.30']['Vh'].shape[0],
            'features.34': cv_dim*model._svds['features.34']['Vh'].shape[0],
            'features.37': cv_dim*model._svds['features.37']['Vh'].shape[0],
            'features.40': cv_dim*model._svds['features.40']['Vh'].shape[0],
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
                device = device,
                )

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            driller = drillers,
            target_modules = peep_layers,
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
                        actds=cv._actds['train'],
                        corevds=cv._corevds['train'],
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
                batch_size = bs,
                n_threads = n_threads,
                verbose = verbose
                )

        ph.get_scores(
            batch_size = bs,
            verbose=verbose
            )

        i = 0
        print('\nPrinting some peeps')
        ph_dl = ph.get_dataloaders(verbose=verbose)
        for data in ph_dl['train']:
            print('phs\n', data['features.34']['peepholes'])
            print('max\n', data['features.34']['score_max'])
            print('ent\n', data['features.34']['score_entropy'])
            i += 1
            if i == 3: break

        ph.evaluate_dists(
                score_type = 'max',
                activations = cv._actds,
                bins = 20
                )
