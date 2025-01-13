# python stuff
from pathlib import Path as Path
from numpy.random import randint
from time import time

# Our stuff
from datasets.cifar import Cifar
from models.model_wrap import ModelWrap 
from coreVectors.coreVectors import CoreVectors 
from coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from classifier.classifier_base import trim_corevectors
from classifier.tkmeans import KMeans as tKMeans 
from classifier.tgmm import GMM as tGMM 
from peepholes.peepholes import Peepholes
from utils.testing import trim_dataloaders

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
            'classifier.0',
            'classifier.3',
            #'features.7',
            #'features.14',
            #'features.28',
            ]
    model.set_target_layers(target_layers=target_layers, verbose=verbose)

    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------
    print('target layers: ', model.get_target_layers()) 
    model.get_svds(path=svds_path, name=svds_name, verbose=verbose)
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #ds_loaders = ds.get_dataset_loaders()
    ds_loaders = trim_dataloaders(ds.get_dataset_loaders(), 0.05)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )

    with corevecs as cv: 
        # copy dataset to coreVect dataset
        cv.get_coreVec_dataset(
                loaders = ds_loaders, 
                verbose = verbose
                ) 

        cv.get_activations(
                batch_size = bs,
                loaders = ds_loaders,
                verbose = verbose
                )

        cv.get_coreVectors(
                batch_size = bs,
                reduct_matrices = model._svds,
                parser = parser_fn,
                verbose = verbose
                )

        cv_dl = cv.get_dataloaders(verbose=verbose)
    
        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['train']:
            print(data['coreVectors']['classifier.0'].shape)
            print(data['coreVectors']['classifier.0'][34:56,:])
            i += 1
            if i == 3: break

        cv.normalize_corevectors(
                wrt='train',
                #from_file=cvs_path/(cvs_name+'.normalization.pt'),
                to_file=cvs_path/(cvs_name+'.normalization2.pt'),
                verbose=verbose
                )
        i = 0
        print('after norm')
        for data in cv_dl['train']:
            print(data['coreVectors']['classifier.0'][34:56,:])
            i += 1
            if i == 3: break
    quit()
    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100
    parser_cv = trim_corevectors
    peep_layer = 'features.7'
    parser_kwargs = {'layer': peep_layer, 'peep_size':100}
    cls_kwargs = {}#{'batch_size':256} 
    cls = tGMM(
            nl_classifier = 100,
            nl_model = n_classes,
            parser = parser_cv,
            parser_kwargs = parser_kwargs,
            cls_kwargs = cls_kwargs,
            device = device
            )

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    peepholes = Peepholes(
            path = phs_path,
            name = phs_name+'.'+peep_layer,
            classifier = cls,
            layer = peep_layer,
            device = device
            )

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        cv_dl = cv.get_dataloaders(
                batch_size = bs,
                verbose = True,
                )
    
        t0 = time()
        cls.fit(dataloader = cv_dl['train'], verbose=verbose)
        print('Fitting time = ', time()-t0)
        
        cls.compute_empirical_posteriors(verbose=verbose)

        ph.get_peepholes(
                loaders = cv_dl,
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
            print('phs\n', data[peep_layer]['peepholes'])
            print('max\n', data[peep_layer]['score_max'])
            print('ent\n', data[peep_layer]['score_entropy'])
            i += 1
            if i == 3: break

        ph.evaluate_dists(
                score_type = 'max',
                coreVectors = cv_dl,
                bins = 20
                )
