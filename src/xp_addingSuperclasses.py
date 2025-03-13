import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from math import ceil
from tqdm import tqdm
import pickle

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
from tensordict import TensorDict, PersistentTensorDict
from tensordict import MemoryMappedTensor as MMT
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda
from torch.utils.data import DataLoader


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
    
    dry_img, _ = ds._dss['train'][0]
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
    dss = random_subsampling(ds._dss, 0.05)
    
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
            #'features.28': partial(
            #    svd_Conv2D, 
            #    reduct_m=model._svds['features.28']['Vh'], 
            #    layer=model._target_layers['features.28'], 
            #    device=device
            #    ),
            }
    
    shapes = {
            'classifier.0': 4096,
            'classifier.3': 4096,
            #'features.28': 300,
            }
  
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
            
        