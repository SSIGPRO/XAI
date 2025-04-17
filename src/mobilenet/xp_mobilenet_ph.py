import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from pathlib import Path as Path
from time import time
from functools import partial


# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors
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
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------

    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 64
    
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'

    svds_path = Path.cwd()/'data/data_200_150clusters'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'data/data_200_200clusters/corevectors'
    cvs_name = 'corevectors'

    act_path = Path.cwd()/'data/data_200_200clusters/corevectors'
    act_name = 'activations'
    
    drill_path = Path.cwd()/'data/data_200_200clusters/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'data/data_200_200clusters/peepholes'
    phs_name = 'peepholes'

    cls_path = Path.cwd()/'data/data_200_200clusters/classifier'
    cls_name = 'clustering'

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

    nn = torchvision.models.mobilenet_v2(pretrained=True)
    print(nn.state_dict().keys())

    in_features = nn.classifier[-1].in_features
    print("in features", in_features)
    nn.classifier[-1] = torch.nn.Linear(in_features, len(ds.get_classes()))
    model = ModelWrap(
        model=nn,
        path=model_dir,
        name=model_name,
        device=device
        )
    
    model.load_checkpoint(verbose=verbose)
    
    target_layers = [ 'features.4.conv.1.0', 'features.5.conv.1.0', 'features.6.conv.1.0',
                     # 'features.7.conv.1.0', 'features.8.conv.1.0', 'features.9.conv.1.0', 'features.10.conv.1.0', 
            #    'features.11.conv.1.0', 'features.12.conv.1.0',
            #    'features.13.conv.1.0', 'features.14.conv.1.0', 
            #    'features.15.conv.1.0', 'features.16.conv.1.0', 'features.16.conv.2', 'features.17.conv.0.0',
            #    'classifier.1'
               ]
    
    model.set_target_modules(target_modules=target_layers, verbose=False)


    direction = {'save_input':True, 'save_output':True}
    model.add_hooks(**direction, verbose=False) 
        
    dry_img, _ = ds._dss['train'][0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------

    model.get_svds(
            target_modules = target_layers,
            path=svds_path,
            name=svds_name,
            verbose=verbose
            )
    
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)
        

    quit()

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    dss = ds._dss
    #dss = random_subsampling(ds._dss, 0.05)

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
    )

    reduction_fns = {
            # 'features.4.conv.1.0': partial(svd_Conv2D,
            #                         reduct_m=model._svds['features.4.conv.1.0']['Vh'], 
            #                         layer =  model._target_modules['features.4.conv.1.0'],
            #                         device=device),
            # 'features.5.conv.1.0': partial(svd_Conv2D,
            #                         reduct_m=model._svds['features.5.conv.1.0']['Vh'], 
            #                         layer = model._target_modules['features.5.conv.1.0'],
            #                         device=device),
            # 'features.6.conv.1.0': partial(svd_Conv2D,        
            #                        reduct_m=model._svds['features.6.conv.1.0']['Vh'], 
            #                        layer = model._target_modules['features.6.conv.1.0'],
            #                        device=device),
            # 'features.7.conv.1.0': partial(svd_Conv2D,        
            #                        reduct_m=model._svds['features.7.conv.1.0']['Vh'], 
            #                        layer = model._target_modules['features.7.conv.1.0'],
            #                        device=device),
            # 'features.8.conv.1.0': partial(svd_Conv2D,
            #                        reduct_m=model._svds['features.8.conv.1.0']['Vh'], 
            #                        layer = model._target_modules['features.8.conv.1.0'],
            #                        device=device), 
            # 'features.9.conv.1.0': partial(svd_Conv2D,
            #                        reduct_m=model._svds['features.9.conv.1.0']['Vh'], 
            #                        layer = model._target_modules['features.9.conv.1.0'],
            #                        device=device),
            # 'features.10.conv.1.0': partial(svd_Conv2D,
            #                        reduct_m=model._svds['features.10.conv.1.0']['Vh'], 
            #                        layer = model._target_modules['features.10.conv.1.0'],
            #                        device=device),
            # 'features.11.conv.1.0': partial(svd_Conv2D,
            #                        reduct_m=model._svds['features.11.conv.1.0']['Vh'], 
            #                        layer = model._target_modules['features.11.conv.1.0'],
            #                        device=device),
            # 'features.12.conv.1.0': partial(svd_Conv2D,
            #                        reduct_m=model._svds['features.12.conv.1.0']['Vh'], 
            #                        layer = model._target_modules['features.12.conv.1.0'],
            #                        device=device),
            'features.13.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.13.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.13.conv.1.0'],
                                   device=device),
            'features.14.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.14.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.14.conv.1.0'],
                                   device=device),
            'features.15.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.15.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.15.conv.1.0'],
                                   device=device),
            'features.16.conv.1.0': partial(svd_Conv2D,       
                                   reduct_m=model._svds['features.16.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.16.conv.1.0'],
                                   device=device), 
            'features.16.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.16.conv.2']['Vh'], 
                                   layer = model._target_modules['features.16.conv.2'],
                                   device=device),
            'features.17.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.17.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.17.conv.0.0'],
                                   device=device),
            'classifier.1': partial(svd_Linear,
                        reduct_m=model._svds['classifier.1']['Vh'], 
                        device=device),  
            
    }

    shapes = {
            # 'features.4.conv.1.0': 300, 
            # 'features.5.conv.1.0':300,
            #'features.6.conv.1.0':300,
            #'features.7.conv.1.0':300,
            #'features.8.conv.1.0':300,
            #'features.9.conv.1.0':300,
            #'features.10.conv.1.0':300,
            #'features.11.conv.1.0':300,
            #'features.12.conv.1.0':300,
            'features.13.conv.1.0':300,
            'features.14.conv.1.0':300,
            'features.15.conv.1.0':300,
            'features.16.conv.1.0':300,
            'features.16.conv.2':300,
            'features.17.conv.0.0':300,
            'classifier.1': 300,
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
        # print('\nPrinting some corevecs')
        # for data in cv_dl['test']:
        #     print('\nfeatures.16.conv.1.0')
        #     print(data['features.16.conv.1.0'][34:56,:])
        #     i += 1
        #     if i == 1: break
        
        cv.normalize_corevectors(
                wrt='train',
                #from_file=cvs_path/(cvs_name+'.normalization.pt'),
                to_file=cvs_path/(cvs_name+'.normalization.pt'),
                verbose=verbose
                )
        
        # i = 0
        # print('after norm')
        # for data in cv_dl['test']:
        #     print(data['features.16.conv.1.0'][34:56,:])
        #     i += 1
        #     if i == 1: break
    quit()

    #--------------------------------
    # Peepholes
    #--------------------------------

    n_classes = 100
    n_cluster = 200
    cv_dim = 200
    parser_cv = trim_corevectors


    peep_layers =[ #'features.4.conv.1.0', 'features.5.conv.1.0', 'features.6.conv.1.0', 'features.7.conv.1.0', 'features.8.conv.1.0', 'features.9.conv.1.0', 'features.10.conv.1.0', 
               #'features.11.conv.1.0', 'features.12.conv.1.0', 
               'features.13.conv.1.0', 'features.14.conv.1.0', 
               'features.15.conv.1.0', 'features.16.conv.1.0', 'features.16.conv.2', 'features.17.conv.0.0',
                'classifier.1'
        
               ]
    

    cls_kwargs = {}#{'batch_size':256} 

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        )
    
    cv_parsers = {
        'features.13.conv.1.0': partial(trim_corevectors,
                          module = 'features.13.conv.1.0',
                          cv_dim = cv_dim),
        'features.14.conv.1.0': partial(trim_corevectors,
                          module = 'features.14.conv.1.0',
                          cv_dim = cv_dim),        
        'features.15.conv.1.0': partial(trim_corevectors,
                          module = 'features.15.conv.1.0',
                          cv_dim = cv_dim),
        'features.16.conv.1.0': partial(trim_corevectors,
                          module = 'features.16.conv.1.0',
                          cv_dim = cv_dim),
        'features.16.conv.2': partial(trim_corevectors,
                          module = 'features.16.conv.2',
                          cv_dim = cv_dim),
        'features.17.conv.0.0': partial(trim_corevectors,   
                          module = 'features.17.conv.0.0',
                          cv_dim = cv_dim),
        'classifier.1': partial(trim_corevectors,
                            module = 'classifier.1',
                            cv_dim = cv_dim),
    }

    feature_sizes = {
        'features.13.conv.1.0': cv_dim,
        'features.14.conv.1.0': cv_dim,
        'features.15.conv.1.0': cv_dim,
        'features.16.conv.1.0': cv_dim,
        'features.16.conv.2': cv_dim,
        'features.17.conv.0.0': cv_dim,
        'classifier.1': cv_dim,
    }
    
    drillers = {}
    for peep_layer in peep_layers:
        #parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

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
            print('phs\n', data[peep_layer]['peepholes'])
            print('max\n', data[peep_layer]['score_max'])
            print('ent\n', data[peep_layer]['score_entropy'])
            i += 1
            if i == 3: break

        ph.evaluate_dists(
                score_type = 'max',
                activations = cv._actds,
                bins = 20
                )
