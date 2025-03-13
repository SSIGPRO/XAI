import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial

# Our stuff
from peepholelib.datasets.cifar import Cifar

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.viz_conceptogram import get_conceptogram_class, get_conceptogram_superclass

# torch stuff
import torch
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
    drill_path = Path.cwd()/'../data/drillers_superclasses'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes_superclasses'
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

    target_layers = [
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            ]

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    drillers = {}

    n_classes = 20
    n_cluster = 10
    cv_dim = 10
    parser_cv = trim_corevectors

    for peep_layer in target_layers:
        parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim, 'label_key': 'superclass'}

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
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            driller = drillers,
            target_modules = target_layers,
            device = device
            )

    with corevecs as cv, peepholes as ph:

        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 
        
        ph.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                )
        idx=106
        ticks = [
            'f.24', 
            'f.26', 
            'f.28', 
            'c.0',
            'c.3']
        superclass_names = [
                "Aquatic Mammals", "Fish", "Flowers", "Food Containers", "Fruit and Vegetables",
                "Household Electrical Devices", "Household Furniture", "Insects", "Large Carnivores", "Large Man-made Outdoor Things",
                "Large Natural Outdoor Scenes", "Large Omnivores and Herbivores", "Medium-sized Mammals", "Non-insect Invertebrates",
                "People", "Reptiles", "Small Mammals", "Trees", "Vehicles 1", "Vehicles 2"
                ]
        class_names = ds._classes

        #get_conceptogram_superclass(cv, ph, idx, target_layers, 'train', ticks, k_rows=1, list_classes=class_names, list_superclasses=superclass_names)
        get_conceptogram_class(cv, ph, idx, target_layers, 'train', ticks, k_rows=1, list_classes=class_names)
        
        