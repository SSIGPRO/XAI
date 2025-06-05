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
from peepholelib.datasets.transforms import mobilenet_v2 as ds_transform 

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes


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

def from_dataset_with_superclass(batch, fine_to_coarse_index=None):
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Compute superclasses
    if fine_to_coarse_index is None:
        raise ValueError("You must provide `fine_to_coarse_index` to extract superclasses.")
    superclasses = torch.tensor([fine_to_coarse_index[int(lbl)] for lbl in labels])

    return {
        'image': images,
        'label': labels,
        'superclass': superclasses,
    }

if __name__ == "__main__":
    torch.cuda.empty_cache()
    use_cuda = torch.cuda.is_available()
    #device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    device = torch.device("cuda:4")
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

    svds_path = Path("/srv/newpenny/XAI/CN/data")
    svds_name = 'svds' 
    
    cvs_path = Path("/srv/newpenny/XAI/CN/superdata/corevectors")
    cvs_name = 'corevectors'
    
    drill_path = Path("/srv/newpenny/XAI/CN//superdata/data20cvdim300/drillers")
    drill_name = 'classifier'

    phs_path = Path("/srv/newpenny/XAI/CN/superdata/data20cvdim300/peepholes")
    phs_name = 'peepholes'

    verbose = True


    target_layers = [ 
        'features.1.conv.0.0', 'features.1.conv.1',
        'features.2.conv.0.0','features.2.conv.1.0', 'features.2.conv.2',
        'features.3.conv.0.0', 'features.3.conv.1.0', 'features.3.conv.2',
        'features.4.conv.0.0','features.4.conv.1.0', 'features.4.conv.2',
        'features.5.conv.0.0', 'features.5.conv.1.0','features.5.conv.2', 
        'features.6.conv.0.0', 'features.6.conv.1.0', 'features.6.conv.2', #B3
        'features.7.conv.0.0', 'features.7.conv.1.0', 'features.7.conv.2', #B3
        'features.8.conv.0.0', 'features.8.conv.1.0', 'features.8.conv.2', #B4
        'features.9.conv.0.0', 'features.9.conv.1.0', 'features.9.conv.2', #B4
        'features.10.conv.0.0', 'features.10.conv.1.0', 'features.10.conv.2', #B5
        'features.11.conv.0.0', 'features.11.conv.1.0','features.11.conv.2', #B5
        'features.12.conv.0.0', 'features.12.conv.1.0', 'features.12.conv.2', #B5
        'features.13.conv.0.0', 'features.13.conv.1.0', 'features.13.conv.2', #B5
        'features.14.conv.0.0', 'features.14.conv.1.0','features.14.conv.2', #B6
        'features.15.conv.0.0', 'features.15.conv.1.0', 'features.15.conv.2', #B6
        'features.16.conv.0.0', 'features.16.conv.1.0','features.16.conv.2', #B6
        'features.17.conv.0.0', 'features.17.conv.1.0', 'features.17.conv.2', #B7
        'features.18.0', 
        'classifier.1'
 ]
    
    n_cluster = 20
    cv_dim = 300
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
        data_path = ds_path,
        dataset=dataset
        )
    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    nn = torchvision.models.mobilenet_v2()
    n_classes = len(ds.get_classes()) 
    model = ModelWrap(
            model = nn,
            device = device
            )
    
    model.update_output(
            output_layer = 'classifier.1', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    
    model.set_target_modules(target_modules=target_layers, verbose=False)

    #--------------------------------
    # SVDs 
    #--------------------------------
    model.get_svds(
            path = svds_path,
            name = svds_name,
            target_modules = target_layers,
            sample_in = ds._dss['train'][0][0],
            rank = 300,
            channel_wise = False,
            verbose = True
            )
    #quit()
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

                'features.1.conv.0.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.1.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.1.conv.0.0'],
                                device=device),
                'features.1.conv.1': partial(svd_Conv2D,
                                reduct_m=model._svds['features.1.conv.1']['Vh'], 
                                layer = model._target_modules['features.1.conv.1'],
                                device=device),
                'features.2.conv.0.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.2.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.2.conv.0.0'],
                                device=device),
                'features.2.conv.1.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.2.conv.1.0']['Vh'], 
                                layer = model._target_modules['features.2.conv.1.0'],
                                device=device),
                'features.2.conv.2': partial(svd_Conv2D,
                                reduct_m=model._svds['features.2.conv.2']['Vh'], 
                                layer = model._target_modules['features.2.conv.2'],
                                device=device),
                
                'features.3.conv.0.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.3.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.3.conv.0.0'],
                                device=device),
                'features.3.conv.1.0': partial(svd_Conv2D,        
                                reduct_m=model._svds['features.3.conv.1.0']['Vh'], 
                                layer = model._target_modules['features.3.conv.1.0'],
                                device=device),
                'features.3.conv.2': partial(svd_Conv2D,        
                                reduct_m=model._svds['features.3.conv.2']['Vh'], 
                                layer = model._target_modules['features.3.conv.2'],
                                device=device),
                'features.4.conv.0.0': partial(svd_Conv2D,        
                                reduct_m=model._svds['features.4.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.4.conv.0.0'],
                                device=device),
                'features.4.conv.1.0': partial(svd_Conv2D,        
                                reduct_m=model._svds['features.4.conv.1.0']['Vh'], 
                                layer = model._target_modules['features.4.conv.1.0'],
                                device=device),
                'features.4.conv.2': partial(svd_Conv2D,
                                reduct_m=model._svds['features.4.conv.2']['Vh'], 
                                layer = model._target_modules['features.4.conv.2'],
                                device=device),
                'features.5.conv.0.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.5.conv.0.0']['Vh'], 
                                layer = model._target_modules['features.5.conv.0.0'],
                                device=device),
                'features.5.conv.1.0': partial(svd_Conv2D,
                                reduct_m=model._svds['features.5.conv.1.0']['Vh'],
                                layer = model._target_modules['features.5.conv.1.0'],
                                device=device),
                'features.5.conv.2': partial(svd_Conv2D,
                                reduct_m=model._svds['features.5.conv.2']['Vh'], 
                                layer = model._target_modules['features.5.conv.2'],
                                device=device),
                'features.6.conv.0.0': partial(svd_Conv2D,        
                                   reduct_m=model._svds['features.6.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.6.conv.0.0'],
                                   device=device),
                'features.6.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.6.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.6.conv.1.0'],
                                   device=device),
                'features.6.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.6.conv.2']['Vh'], 
                                   layer = model._target_modules['features.6.conv.2'],
                                   device=device),
                'features.7.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.7.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.7.conv.0.0'],
                                   device=device),
                'features.7.conv.1.0': partial(svd_Conv2D,        
                                   reduct_m=model._svds['features.7.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.7.conv.1.0'],
                                   device=device), 
                'features.7.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.7.conv.2']['Vh'], 
                                   layer = model._target_modules['features.7.conv.2'],
                                   device=device),
                'features.8.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.8.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.8.conv.0.0'],
                                   device=device),

                'features.8.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.8.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.8.conv.1.0'],
                                   device=device), 
                 'features.8.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.8.conv.2']['Vh'], 
                                   layer = model._target_modules['features.8.conv.2'],
                                   device=device),     
                'features.9.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.9.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.9.conv.0.0'],
                                   device=device),              
                'features.9.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.9.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.9.conv.1.0'],
                                   device=device),
                'features.9.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.9.conv.2']['Vh'], 
                                   layer = model._target_modules['features.9.conv.2'],
                                   device=device),
                'features.10.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.10.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.10.conv.0.0'],
                                   device=device),
                'features.10.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.10.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.10.conv.1.0'],
                                   device=device),
                'features.10.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.10.conv.2']['Vh'], 
                                   layer = model._target_modules['features.10.conv.2'],
                                   device=device),
                'features.11.conv.0.0': partial(svd_Conv2D,     
                                          reduct_m=model._svds['features.11.conv.0.0']['Vh'], 
                                          layer = model._target_modules['features.11.conv.0.0'],
                                          device=device),
                'features.11.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.11.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.11.conv.1.0'],
                                   device=device),
                'features.11.conv.2': partial(svd_Conv2D,       
                                          reduct_m=model._svds['features.11.conv.2']['Vh'], 
                                          layer = model._target_modules['features.11.conv.2'],
                                          device=device),
                'features.12.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.12.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.12.conv.0.0'],
                                   device=device),
                'features.12.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.12.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.12.conv.1.0'],
                                   device=device),
                'features.12.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.12.conv.2']['Vh'], 
                                   layer = model._target_modules['features.12.conv.2'],
                                   device=device),   
                'features.13.conv.0.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.13.conv.0.0']['Vh'], 
                                   layer = model._target_modules['features.13.conv.0.0'],
                                   device=device),                       
                'features.13.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.13.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.13.conv.1.0'],
                                   device=device),
                'features.13.conv.2': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.13.conv.2']['Vh'], 
                                   layer = model._target_modules['features.13.conv.2'],
                                   device=device),
                'features.14.conv.0.0': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.14.conv.0.0']['Vh'], 
                                          layer = model._target_modules['features.14.conv.0.0'],
                                          device=device),
                'features.14.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.14.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.14.conv.1.0'],
                                   device=device),
                'features.14.conv.2': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.14.conv.2']['Vh'], 
                                          layer = model._target_modules['features.14.conv.2'],
                                          device=device),
                'features.15.conv.0.0': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.15.conv.0.0']['Vh'], 
                                          layer = model._target_modules['features.15.conv.0.0'],
                                          device=device),
                'features.15.conv.1.0': partial(svd_Conv2D,
                                   reduct_m=model._svds['features.15.conv.1.0']['Vh'], 
                                   layer = model._target_modules['features.15.conv.1.0'],
                                   device=device),
                'features.15.conv.2': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.15.conv.2']['Vh'], 
                                          layer = model._target_modules['features.15.conv.2'],
                                          device=device),
                'features.16.conv.0.0': partial(svd_Conv2D,
                                          reduct_m=model._svds['features.16.conv.0.0']['Vh'], 
                                          layer = model._target_modules['features.16.conv.0.0'],
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
                'features.17.conv.1.0': partial(svd_Conv2D,
                                        reduct_m=model._svds['features.17.conv.1.0']['Vh'], 
                                        layer = model._target_modules['features.17.conv.1.0'],
                                        device=device),
                'features.17.conv.2': partial(svd_Conv2D,
                                        reduct_m=model._svds['features.17.conv.2']['Vh'], 
                                        layer = model._target_modules['features.17.conv.2'],
                                        device=device),
                'features.18.0': partial(svd_Conv2D,
                                        reduct_m=model._svds['features.18.0']['Vh'], 
                                        layer = model._target_modules['features.18.0'],
                                        device=device),
                'classifier.1': partial(svd_Linear,
                                reduct_m=model._svds['classifier.1']['Vh'], 
                                device=device),  
            
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
        cv.parse_ds(
            batch_size = bs,
            datasets = ds,
            n_threads = n_threads,
            verbose = verbose,
            ds_parser=partial(from_dataset_with_superclass, fine_to_coarse_index=fine_to_coarse_index),
            key_list=['image', 'label', 'superclass']
        )
        
        
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
        
        for portion, _dss in cv._dss.items():
            
            n_samples =len(cv._dss[portion])
            _dss['superclass'] = MMT.empty(shape=torch.Size((n_samples,)))
           
            print("After insertion, keys are:", list(_dss.keys()))
            dl_label = DataLoader(_dss['label'], batch_size=bs, collate_fn=lambda x:x, shuffle=False)
            dl_superclass = DataLoader(_dss['superclass'], batch_size=bs, collate_fn=lambda x:x, shuffle=False)
            start_idx = 0
            
            with _dss.unlock_():
                # Clone the current "superclass" tensor into a temporary variable
                        temp_super = _dss['superclass'].clone()
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
                        _dss.set('superclass', temp_super)


        print(cv._dss['train']['superclass'], cv._dss['val']['superclass'], cv._dss['test']['superclass'])  



    #--------------------------------
    # Peepholes
    #--------------------------------

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        )

    cv_parsers = {
        'features.1.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.1.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.1.conv.1': partial(trim_channelwise_corevectors,
                          module = 'features.1.conv.1',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.2.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.2.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.2.conv.1.0': partial(trim_channelwise_corevectors,    
                          module = 'features.2.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.2.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.2.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.3.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.3.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.3.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.3.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.3.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.3.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.4.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.4.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.4.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.4.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.4.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.4.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.5.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.5.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.5.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.5.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.5.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.5.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'), 
        'features.6.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.6.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.6.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.6.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.6.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.6.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.7.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.7.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.7.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.7.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.7.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.7.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.8.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.8.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.8.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.8.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.8.conv.2': partial(trim_channelwise_corevectors, 
                            module = 'features.8.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.9.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.9.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.9.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.9.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.9.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.9.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.10.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.10.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.10.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.10.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.10.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.10.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.11.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.11.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.11.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.11.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.11.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.11.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.12.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.12.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),    
        'features.12.conv.1.0': partial(trim_channelwise_corevectors,
                          module = 'features.12.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.12.conv.2': partial(trim_channelwise_corevectors,
                          module = 'features.12.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.13.conv.0.0': partial(trim_channelwise_corevectors,
                          module = 'features.13.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),   
        'features.13.conv.1.0': partial(trim_corevectors,
                          module = 'features.13.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.13.conv.2': partial(trim_corevectors,
                          module = 'features.13.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.14.conv.0.0': partial(trim_corevectors,
                          module = 'features.14.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.14.conv.1.0': partial(trim_corevectors,
                          module = 'features.14.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.14.conv.2': partial(trim_corevectors,
                          module = 'features.14.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.15.conv.0.0': partial(trim_corevectors,
                          module = 'features.15.conv.0.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.15.conv.1.0': partial(trim_corevectors,
                          module = 'features.15.conv.1.0',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.15.conv.2': partial(trim_corevectors,
                          module = 'features.15.conv.2',
                          cv_dim = cv_dim,
                          label_key = 'superclass'),
        'features.16.conv.0.0': partial(trim_corevectors,
                          module = 'features.16.conv.0.0',
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
        'features.17.conv.1.0': partial(trim_corevectors,	
                            module = 'features.17.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.17.conv.2': partial(trim_corevectors,
                            module = 'features.17.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.18.0': partial(trim_corevectors,
                            module = 'features.18.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'classifier.1': partial(trim_corevectors,
                            module = 'classifier.1',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
                                      
    }

    feature_sizes = {
            'features.1.conv.0.0': cv_dim,
            'features.1.conv.1': cv_dim,
            'features.2.conv.0.0': cv_dim,
            'features.2.conv.1.0': cv_dim,
            'features.2.conv.2': cv_dim,    
            'features.3.conv.0.0': cv_dim,
            'features.3.conv.1.0': cv_dim,
            'features.3.conv.2': cv_dim,
            'features.4.conv.0.0': cv_dim,
            'features.4.conv.1.0': cv_dim,
            'features.4.conv.2': cv_dim,
            'features.5.conv.0.0': cv_dim,
            'features.5.conv.1.0': cv_dim,
            'features.5.conv.2': cv_dim,
            'features.6.conv.0.0': cv_dim,
            'features.6.conv.1.0': cv_dim,
            'features.6.conv.2': cv_dim,
            'features.7.conv.0.0': cv_dim,
            'features.7.conv.1.0': cv_dim,
            'features.7.conv.2': cv_dim,
            'features.8.conv.0.0': cv_dim,
            'features.8.conv.1.0': cv_dim,
            'features.8.conv.2': cv_dim,
            'features.9.conv.0.0': cv_dim,
            'features.9.conv.1.0': cv_dim,
            'features.9.conv.2': cv_dim,
            'features.10.conv.0.0': cv_dim,
            'features.10.conv.1.0': cv_dim,
            'features.10.conv.2': cv_dim,
            'features.11.conv.0.0': cv_dim,
            'features.11.conv.1.0': cv_dim,
            'features.11.conv.2': cv_dim,
            'features.12.conv.0.0': cv_dim,
            'features.12.conv.1.0': cv_dim,
            'features.12.conv.2': cv_dim,
            'features.13.conv.0.0': cv_dim,          
            'features.13.conv.1.0': cv_dim,
            'features.13.conv.2': cv_dim,
            'features.14.conv.0.0': cv_dim,
            'features.14.conv.1.0': cv_dim,
            'features.14.conv.2': cv_dim,
            'features.15.conv.0.0': cv_dim,
            'features.15.conv.1.0': cv_dim,
            'features.15.conv.2': cv_dim,
            'features.16.conv.0.0': cv_dim,
            'features.16.conv.1.0': cv_dim,
            'features.16.conv.2': cv_dim,
            'features.17.conv.0.0': cv_dim,
            'features.17.conv.1.0': cv_dim,
            'features.17.conv.2': cv_dim,
            'features.18.0': cv_dim,
            'classifier.1': 100,

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



