import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
import matplotlib.pyplot as plt
# Our stuff
from peepholelib.datasets.imagenet import ImageNet
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.transforms import vgg16_cifar100

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader


if __name__ == "__main__":
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    OOD_dataset = 'imagenet-1k'

    ds_path_ori = '/srv/newpenny/dataset/CIFAR100' 
    ds_path_OOD = f'/srv/newpenny/dataset/{OOD_dataset}/data'

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
    # Dataset 
    #--------------------------------

    ds_ori = Cifar(
            data_path = ds_path_ori,
            dataset=dataset
            )
    ds_ori.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            )
    
    ds_OOD = ImageNet(
            data_path = ds_path_OOD,
            dataset='ImageNet',
            )
    ds_OOD.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 4, 'pin_memory': True},
            seed = seed,
            #transform = vgg16_cifar100,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    n_classes = len(ds_ori.get_classes()) 
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
            #'features.7',
            #'features.14',
            #'features.28',
            ]
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds_ori._dss['train'][0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------
    print('target layers: ', model.get_target_modules()) 
    model.get_svds(
            target_modules = target_layers,
            path=svds_path,
            name=svds_name,
            verbose=verbose
            )

    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #dss = ds._dss
    dss = random_subsampling(ds_OOD._dss, 0.005)
    cvs_path_ = cvs_path/f'{OOD_dataset}'
    phs_path_ = phs_path/f'{OOD_dataset}'
    
    corevecs = CoreVectors(
            path = cvs_path_,
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
        #     'features.28': partial(
        #        svd_Conv2D, 
        #        reduct_m=model._svds['features.28']['Vh'], 
        #        layer=model._target_modules['features.28'], 
        #        device=device
        #        ),
            }
    
    shapes = {
            'classifier.0': 4096,
            'classifier.3': 4096,
            #'features.28': 300,
            }

    with corevecs as cv: 
        # copy dataset to activatons file
        cv.get_activations(
                batch_size = bs,
                datasets = dss,
                verbose = verbose
                )        

        # computing the corevectors
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                shapes = shapes,
                verbose = verbose
                )

        cv_dl = cv.get_dataloaders(verbose=verbose)
    
        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['test']:
            print('\nclassifier.0')
            print(data['classifier.0'][34:56,:])
            i += 1
            if i == 1: break
        
        # cv.normalize_corevectors(
        #         wrt='train',
        #         #from_file=cvs_path/(cvs_name+'.normalization.pt'),
        #         to_file=cvs_path/(cvs_name+'.normalization.pt'),
        #         verbose=verbose
        #         )
        
        i = 0
        print('after norm')
        for data in cv_dl['test']:
            print(data['classifier.0'][34:56,:])
            i += 1
            if i == 1: break
        print(cv._actds['test']['image'][0].shape)
        plt.imshow(cv._actds['test']['image'][4].cpu().numpy().reshape(224, 224, 3))
        plt.savefig('test.png')
    quit()
    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100
    n_cluster = 10
    cv_dim = 10
    parser_cv = trim_corevectors
    peep_layers = ['classifier.0', 'classifier.3']
    
    cls_kwargs = {}#{'batch_size': bs} 

    drillers = {}
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    drillers = {}
    for peep_layer in peep_layers:
        parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = cv_dim,
                parser = parser_cv,
                parser_kwargs = parser_kwargs,
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
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 
        for drill_key, driller in drillers.items():
            if (drill_path/(driller._suffix+'.empp.pt')).exists():
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
                verbose = verbose
                )

        ph.get_scores(
            batch_size = bs,
            verbose=verbose
            )

        i = 0
        print('\nPrinting some peeps')
        ph_dl = ph.get_dataloaders(verbose=verbose)
        for data in ph_dl['test']:
            print('phs\n', data['classifier.0']['peepholes'])
            print('max\n', data['classifier.0']['score_max'])
            print('ent\n', data['classifier.0']['score_entropy'])
            i += 1
            if i == 3: break

        ph.evaluate_dists(
                score_type = 'max',
                activations = cv._actds,
                bins = 20
                )