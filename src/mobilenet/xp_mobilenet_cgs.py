import sys
from pathlib import Path as Path

import numpy as np
import torchvision
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())


# torch stuff
import torch
import pickle
import torch.nn.functional as F

# our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.transforms import mobilenet_v2 as ds_transform 
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
#from peepholelib.utils.conceptograms import get_all_class_names, get_indices_by_classname, get_conceptogram
from peepholelib.utils.viz_empp import empirical_posterior_heatmaps

# python stuff
import pandas as pd
from pathlib import Path as Path
from functools import partial


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

def superclass_pred_fn():
    """
    Creates a prediction function that maps fine class softmax scores to superclass probabilities.

    Returns:
        Callable[[Tensor], Tensor]: Function that takes network output logits and returns superclass scores.
    """

    meta_path = '/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')

    fine_label_names = meta['fine_label_names']
    coarse_label_names = meta['coarse_label_names']

    # fine-coarse mapping
    fine_to_coarse_idx = {}
    for coarse_idx, coarse_name in enumerate(coarse_label_names):
        for fine_name in coarse_to_fine[coarse_name]:
            fine_idx = fine_label_names.index(fine_name)
            fine_to_coarse_idx[fine_idx] = coarse_idx

    # Tensor version of the map
    map_tensor = torch.tensor([fine_to_coarse_idx[i] for i in range(100)])

    def pred_fn(logits: torch.Tensor) -> torch.Tensor:
        """
        Converts fine logits to superclass probabilities.

        Args:
            logits (Tensor): Logits from the model (size: [100])

        Returns:
            Tensor: Superclass probabilities (size: [20])
        """
        probs = F.softmax(logits, dim=0)
        superclass_probs = torch.zeros(20, dtype=torch.float32)

        for fine_idx, prob in enumerate(probs):
            coarse_idx = map_tensor[fine_idx].item()
            superclass_probs[coarse_idx] += prob

        return superclass_probs

    return pred_fn

def get_all_class_names(ds, superclass):
    if ds._classes is None:
        raise RuntimeError("Dataset not loaded. Please run load_data() first.")

    if not superclass:
        return [ds._classes[i] for i in sorted(ds._classes)]

    # Handle superclass logic (only applies to CIFAR-100)
    if ds.dataset != 'CIFAR100':
        raise ValueError("Superclasses are only defined for CIFAR100.")

    meta_path = '/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')

    return meta['coarse_label_names']

def get_indices_by_classname(ds, split: str, class_name: str):
    """
    Returns all indices from a dataset split that belong to the given class or superclass name.

    Args:
        ds (DatasetBase): Instance of a DatasetBase dataset (e.g., Cifar).
        split (str): One of 'train', 'val', or 'test'.
        class_name (str): Class name or superclass name.

    Returns:
        List[int]: List of matching indices.
    """
    if ds._dss is None:
        raise RuntimeError("Dataset not loaded. Please run load_data() first.")
    if split not in ds._dss:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(ds._dss.keys())}")

    split_data = ds._dss[split]

    if ds.dataset == 'CIFAR100':
        # Load CIFAR-100 meta
        meta_path = '/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta'
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f, encoding='latin1')

        fine_to_idx = {v: k for k, v in ds._classes.items()}

        # if superclass
        if class_name in coarse_to_fine: 
            fine_classes = coarse_to_fine[class_name]
            target_class_indices = {fine_to_idx[c] for c in fine_classes}
            return [i for i, (_, y) in enumerate(split_data) if y in target_class_indices]
    
    # if not superclass
    class_to_idx = {v: k for k, v in ds._classes.items()}
    if class_name not in class_to_idx:
        raise ValueError(f"Class name '{class_name}' not found in dataset classes.")
    
    target_class_idx = class_to_idx[class_name]
    return [i for i, (_, y) in enumerate(split_data) if y == target_class_idx]

def print_empirical_scores(drillers, threshold=0.9):
    """
    For each module, check if each class is represented in at least one cluster,
    i.e., if any cluster assigns P(g|c) >= threshold for that class.

    Returns a dictionary mapping module name → (n_classes_represented, total_classes, coverage_percent)
    """
    scores = {}

    for module_name, driller in drillers.items():
        driller.load()

        if not hasattr(driller, "_empp") or driller._empp is None:
            print(f" No empirical posterior found for {module_name}, skipping.")
            continue

        empp = driller._empp  # [n_clusters, n_classes]
        empp = empp.cpu() if empp.is_cuda else empp

        # Transpose to [n_classes, n_clusters]
        empp_t = empp.T

        # For each class, check if any cluster has prob >= threshold
        class_represented = (empp_t >= threshold).any(dim=1)  # shape: [n_classes]
        n_represented = class_represented.sum().item()
        n_classes = empp.shape[1]
        coverage = 100 * n_represented / n_classes

        scores[module_name] = (n_represented, n_classes, coverage)

        print(f"[{module_name}] {n_represented}/{n_classes} classes represented (≥ {threshold}) — {coverage:.2f}%")

    return scores

    
 

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    cuda_index = 0
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    bs = 512 
    seed = 29

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'

    cvs_path = Path("/srv/newpenny/XAI/CN/data/corevectors")
    cvs_name = 'corevectors'

    drill_path = Path("/srv/newpenny/XAI/CN/data200cvdim300/drillers")
    drill_name = 'classifier'

    phs_path = Path("/srv/newpenny/XAI/CN/data200cvdim300/peepholes")
    phs_name = 'peepholes'

    target_layers = ['features.3.conv.1.0',
        #'features.4.conv.1.0', 'features.5.conv.1.0', 
        'features.6.conv.1.0',
                # 'features.7.conv.1.0', 'features.8.conv.1.0', 'features.9.conv.1.0', 'features.10.conv.1.0', 
        #'features.11.conv.0.0', 'features.11.conv.1.0','features.11.conv.2', #B5
        #'features.12.conv.0.0', 'features.12.conv.1.0', 'features.12.conv.2', #B5
        #'features.13.conv.0.0', 
        'features.13.conv.1.0', #'features.13.conv.2', #B5
        #'features.14.conv.0.0', 
        'features.14.conv.1.0',#'features.14.conv.2', #B6
        #'features.15.conv.0.0', 
        'features.15.conv.1.0', #'features.15.conv.2', #B6
        #'features.16.conv.0.0', 
        'features.16.conv.1.0','features.16.conv.2', #B6
        'features.17.conv.0.0', #'features.17.conv.1.0', 'features.17.conv.2', #B7
        'classifier.1'
        ]
    
    cv_dim = 300
    n_cluster = 200
    
    superclass = False

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
            transform = ds_transform,
            seed = seed,
            )
    

    #print("class names ", get_all_class_names(ds, True))
    #indexes = get_indices_by_classname(ds, 'train', 'aquatic_mammals')
    #print("indexes ", indexes)

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
            verbose = False
            )
    
    model.set_target_modules(target_modules=target_layers, verbose=False)
    #--------------------------------
    # Core Vectors
    #--------------------------------
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        )
    #--------------------------------
    # Peepholes
    #--------------------------------
    
    feature_sizes = {
        'features.3.conv.1.0': cv_dim,
        'features.6.conv.1.0': cv_dim,
        'features.13.conv.1.0': cv_dim,
        'features.14.conv.1.0': cv_dim,
        'features.15.conv.1.0': cv_dim,
        'features.16.conv.1.0': cv_dim,
        'features.16.conv.2': cv_dim,
        'features.17.conv.0.0': cv_dim,
        'classifier.1': 100,
    }

    if superclass:
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
    else:
        cv_parsers = {
            'features.3.conv.1.0': partial(trim_corevectors,
                            module = 'features.3.conv.1.0',
                            cv_dim = cv_dim),
            'features.6.conv.1.0': partial(trim_corevectors,
                            module = 'features.6.conv.1.0',
                            cv_dim = cv_dim),
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
                                cv_dim = 100),
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
        
    #--------------------------------
    # Superclass for Cifar100
    #--------------------------------
    # # Load fine to coarse index from CIFAR-100 meta
    # meta_path = '/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta'
    # with open(meta_path, 'rb') as f:
    #     meta = pickle.load(f, encoding='latin1')
    # fine_label_names = meta['fine_label_names']
    # coarse_label_names = meta['coarse_label_names']

    # fine_to_coarse_index = {}
    # for coarse_idx, coarse_name in enumerate(coarse_label_names):
    #     for fine_name in coarse_to_fine[coarse_name]:
    #         fine_idx = fine_label_names.index(fine_name)
    #         fine_to_coarse_index[fine_idx] = coarse_idx

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
                n_threads = 32,
                verbose = False
                )

    #--------------------------------
    # Conceptograms
    #--------------------------------

    #save_path = '/home/claranunesbarrancos/repos/XAI/src/mobilenet/superconcepto/200cvdim_30clusters'
    
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
                n_threads = 32,
                verbose = False
                )
        print_empirical_scores(ph._drillers, 0.75)

        #empirical_posterior_heatmaps(ph._drillers, '/home/claranunesbarrancos/repos/XAI/src/mobilenet/empp/300cvdim_150clusters')

        # get_conceptogram(
        #     path = '/home/claranunesbarrancos/repos/XAI/src/mobilenet/superconcepto',
        #     name = 'cp_300cvdim_100clusters'
        #     corevecs = cv,
        #     peepholes =ph,
        #     portion = 'test',
        #     sample = 9999,
        #     target_layers = target_layers,
        #     classes = ds._classes,
        #     ticks = target_layers,
        #     label_key = 'label'
        #     pred_fn = pred_fun
        #     ds = ds,
        # )






