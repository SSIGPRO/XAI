import sys
from pathlib import Path as Path

import numpy as np
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())


# torch stuff
import torch
import pickle

# our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.conceptograms import *
from peepholelib.utils.viz_conceptogram import *
from peepholelib.utils.viz_empp import empirical_posterior_heatmaps

# python stuff
import pandas as pd
from pathlib import Path as Path

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    cuda_index = 0
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    bs = 512 
    seed = 29

    #--------------------------------
    # Dataset
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    dataset = 'CIFAR100' 

    ds = Cifar(
        data_path = ds_path,
        dataset=dataset
        )
    
    ds.load_data(
            batch_size = bs,
            data_kwargs = {'num_workers': 8, 'pin_memory': True},
            seed = seed,
            )
    
    #--------------------------------
    # Core Vectors
    #--------------------------------
    cvs_path = Path('/home/claranunesbarrancos/repos/XAI/src/mobilenet/superdata/data_200_30clusters/corevectors')
    cvs_name = 'corevectors'
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        device = device 
        )
    #--------------------------------
    # Peepholes
    #--------------------------------

    phs_path = Path('/home/claranunesbarrancos/repos/XAI/src/mobilenet/superdata/data_200_30clusters/peepholes')
    phs_name = 'peepholes'

    drill_path = Path('/home/claranunesbarrancos/repos/XAI/src/mobilenet/superdata/data_200_30clusters/drillers')
    drill_name = 'classifier'

    target_layers = [ 
                    # 'features.13.conv.1.0', 'features.14.conv.1.0', 'features.15.conv.1.0', 'features.16.conv.1.0', 'features.16.conv.2', 
                    #  'features.17.conv.0.0', 
                     'classifier.1',
                     ]
    
    n_clusters = 30
    n_classes = 20
    cv_dim = 100

    parser_cv = trim_corevectors
    drillers = {}
    ph_dict = {}
    
    for peep_layer in target_layers:
        parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

        drillers[peep_layer] = tGMM(
                        path = drill_path,
                        name = drill_name+'.'+peep_layer,
                        nl_classifier = n_clusters,
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
        target_modules = target_layers,
        device = device
    )
        
    #--------------------------------
    # Superclass for Cifar100
    #--------------------------------
    # Load fine to coarse index from CIFAR-100 meta
    meta_path = '/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')
    fine_label_names = meta['fine_label_names']
    coarse_label_names = meta['coarse_label_names']
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
        for fine_name in coarse_to_fine[coarse_name]:
            fine_idx = fine_label_names.index(fine_name)
            fine_to_coarse_index[fine_idx] = coarse_idx

    empirical_posterior_heatmaps(peepholes._driller, '/home/claranunesbarrancos/repos/XAI/src/mobilenet/superempp/200cvdim_30clusters')


    #--------------------------------
    # Conceptograms
    #--------------------------------

    #save_path = '/home/claranunesbarrancos/repos/XAI/src/mobilenet/superconcepto/200cvdim_30clusters'
    
    #with corevecs as cv:

        # generate_conceptograms(
        #     peepholes=peepholes,
        #     save_path=save_path,
        #     target_layers=target_layers,
        #     n_classes=n_classes,
        # )
        # generate_conceptogram(
        #     peepholes=peepholes,
        #     sample=89, # [0,9999]
        #     save_path=save_path,
        #     target_layers=target_layers,
        #     n_classes=n_classes,
        # )
        
        # generate_sample_conceptogram(
        #     peepholes=peepholes,
        #     sample=9998, # [0,9999]
        #     ds=ds,
        #     corevecs=cv,
        #     result=1,
        #     confidence=0,
        #     save_path=save_path,
        #     target_layers=target_layers,
        #     n_classes=n_classes,
        #     fine_to_coarse_index=fine_to_coarse_index,
        #     coarse_label_names=coarse_label_names,
        # )
        
        # generate_sample_conceptograms(
        #     peepholes=peepholes,
        #     class_id=89, # [0,99]
        #     ds=ds,
        #     corevecs=cv,
        #     result=1,
        #     confidence=0,
        #     save_path=save_path,
        #     target_layers=target_layers,
        #     n_classes=n_classes,
        # )


    # with corevecs as cv, peepholes as ph:
    #     cv.load_only(loaders = ['train', 'test', 'val'],
    #                 verbose=True)
    #     ph.get_peepholes(
    #             corevectors = cv,
    #             batch_size = bs,
    #             n_threads = 32,
    #             verbose = True
    #             )

    #     get_conceptogram_class(
    #         cv=cv,
    #         ph=peepholes,
    #         idx=9999,
    #         target_layers=target_layers,
    #         portion='test',
    #         ticks=target_layers,
    #         k_rows=10,
    #         list_classes=ds._classes,
    #         path='/home/claranunesbarrancos/repos/XAI/src/mobilenet/concepto/200cvdim_200clusters/class_conceptogram_9999_100.png',
        # )
    #     get_conceptogram_superclass(
    #         cv=cv,
    #         ph=peepholes,
    #         idx=8000,
    #         target_layers=target_layers,
    #         portion='test',
    #         ticks=target_layers,
    #         k_rows=10,
    #         list_classes=ds._classes,
    #         list_superclasses=coarse_label_names,
    #         path='/home/claranunesbarrancos/repos/XAI/src/mobilenet/superconcepto/200cvdim_20clusters/class_conceptogram_8000_20.png',
    #     )




