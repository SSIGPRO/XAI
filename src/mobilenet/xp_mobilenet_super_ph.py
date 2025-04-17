import sys
from pathlib import Path as Path

sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from pathlib import Path as Path
from time import time
from functools import partial
from matplotlib import pyplot as plt
import pickle


# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.viz_conceptogram import get_conceptogram_class


from peepholelib.utils.samplers import random_subsampling 
from torch.utils.data import DataLoader
from tensordict import MemoryMappedTensor as MMT
from math import ceil
from tqdm import tqdm


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
    bs = 256
    n_threads = 512

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'

    svds_path = Path.cwd()/'data/data_300_150clusters'
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'superdata/data_300_30clusters/corevectors'
    cvs_name = 'corevectors'
    
    drill_path = Path.cwd()/'superdata/data_300_30clusters/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'superdata/data_300_30clusters/peepholes'
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
    #device = torch.device("cpu")

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
    
    target_layers = [ #'features.4.conv.1.0', 'features.5.conv.1.0', 'features.6.conv.1.0', 'features.7.conv.1.0', 'features.8.conv.1.0', 'features.9.conv.1.0', 'features.10.conv.1.0', 
               #'features.11.conv.1.0', 'features.12.conv.1.0',
               'features.13.conv.1.0', 'features.14.conv.1.0', 
               'features.15.conv.1.0', 'features.16.conv.1.0', 'features.16.conv.2', 'features.17.conv.0.0',
               'classifier.1'
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
    t0 = time()
    model.get_svds(
            target_modules = target_layers,
            path = svds_path,
            rank = 300,
            channel_wise = False,
            name = svds_name,
            verbose = verbose
            )
    print('time: ', time()-t0)
    #device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")

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


    #--------------------------------
    # CoreVectors 
    #--------------------------------
    dss = ds._dss

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
            'classifier.1': 100,
            }

    #--------------------------------
    # Superclasses 
    #--------------------------------
    
    with open('/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta', 'rb') as f:
        meta = pickle.load(f, encoding='latin1')
    fine_label_names = meta['fine_label_names']      # List of 100 fine label names
    coarse_label_names = meta['coarse_label_names']  # List of 20 coarse label names


    coarse_to_fine = {
                        'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                        'fish': ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                        'flowers': ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                        'food_containers': ['bottle', 'bowl', 'can', 'cup', 'plate'],
                        'fruit_and_vegetables': ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                        'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                        'household_furniture': ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                        'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                        'large_carnivores': ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                        'large_man-made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                        'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                        'large_omnivores_and_herbivores': ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                        'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                        'non-insect_invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
                        'people': ['baby', 'boy', 'girl', 'man', 'woman'],
                        'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                        'small_mammals': ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                        'trees': ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                        'vehicles_1': ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                        'vehicles_2': ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']
                    }
    fine_to_coarse_index = {}
    for coarse_idx, coarse_name in enumerate(coarse_label_names):
        # For each coarse class, go over its fine class names using your mapping:
        for fine_name in coarse_to_fine[coarse_name]:
                fine_idx = fine_label_names.index(fine_name)  # get index of the fine label name
                fine_to_coarse_index[fine_idx] = coarse_idx

    with corevecs as cv: 
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
        
        for portion, _acts in cv._actds.items():
            
            n_samples = cv._n_samples[portion]
            _acts['superclass'] = MMT.empty(shape=torch.Size((n_samples,)))
           
            print("After insertion, keys are:", list(_acts.keys()))
            dl_label = DataLoader(_acts['label'], batch_size=bs, collate_fn=lambda x:x, shuffle=False)
            dl_superclass = DataLoader(_acts['superclass'], batch_size=bs, collate_fn=lambda x:x, shuffle=False)
            start_idx = 0
            
            with _acts.unlock_():
                # Clone the current "superclass" tensor into a temporary variable
                        temp_super = _acts['superclass'].clone()
                        # Iterate over batches using the data loaders (dl_superclass is not used in the loop body but shown for symmetry)
                        for data_in, _ in tqdm(zip(dl_label, dl_superclass),
                                                disable=not verbose,
                                                total=ceil(n_samples / bs)):
                                # Compute the batch of new values.
                                # This builds a tensor by mapping each label (converted to int) to its coarse index.
                                batch_values = torch.tensor(
                                        [fine_to_coarse_index[int(label)] for label in data_in],
                                        dtype=torch.long,
                                        device=temp_super.device
                                )
                                
                                # Update the appropriate slice in the temporary tensor using copy_()
                                temp_super[start_idx:start_idx + len(batch_values)].copy_(batch_values)
                                start_idx += len(batch_values)

                        # After processing all batches, write the updated tensor back into the tensordict.
                        _acts.set('superclass', temp_super)


        print(cv._actds['train']['superclass'], cv._actds['val']['superclass'], cv._actds['test']['superclass'])  



    #--------------------------------
    # Peepholes
    #--------------------------------
    #device = torch.device("cpu")
    n_classes = 20
    n_cluster = 30
    cv_dim = 300
    parser_cv = trim_corevectors

    peep_layers = [ #'features.4.conv.1.0', 'features.5.conv.1.0', 'features.6.conv.1.0', 'features.7.conv.1.0', 'features.8.conv.1.0', 'features.9.conv.1.0', 'features.10.conv.1.0', 
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
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.14.conv.1.0': partial(trim_corevectors,
                          module = 'features.14.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.15.conv.1.0': partial(trim_corevectors,
                          module = 'features.15.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.16.conv.1.0': partial(trim_corevectors,
                          module = 'features.16.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.16.conv.2': partial(trim_corevectors,
                          module = 'features.16.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.17.conv.0.0': partial(trim_corevectors,   
                          module = 'features.17.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'classifier.1': partial(trim_corevectors,
                            module = 'classifier.1',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
                                      
    }

    feature_sizes = {
            'features.13.conv.1.0': cv_dim,
            'features.14.conv.1.0': cv_dim,
            'features.15.conv.1.0': cv_dim,
            'features.16.conv.1.0': cv_dim,
            'features.16.conv.2': cv_dim,
            'features.17.conv.0.0': cv_dim,
            'classifier.1': 300,

            # 'features.13.conv.1.0': cv_dim*model._svds['features.13.conv.1.0']['Vh'].shape[0],
            # 'features.14.conv.1.0': cv_dim*model._svds['features.14.conv.1.0']['Vh'].shape[0],
            # 'features.15.conv.1.0': cv_dim*model._svds['features.15.conv.1.0']['Vh'].shape[0],
            # 'features.16.conv.1.0': cv_dim*model._svds['features.16.conv.1.0']['Vh'].shape[0],
            # 'features.16.conv.2': cv_dim*model._svds['features.16.conv.2']['Vh'].shape[0],
            # 'features.17.conv.0.0': cv_dim*model._svds['features.17.conv.0.0']['Vh'].shape[0],
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
        print("cv shape ", cv._corevds['train'].shape)

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



