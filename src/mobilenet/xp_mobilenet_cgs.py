import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# torch stuff
import torch
import torchvision
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
from peepholelib.utils.conceptograms import  *
from peepholelib.utils.mappings import coarse_to_fine_cifar100 as coarse_to_fine
from peepholelib.utils.viz_empp import *

# python stuff
import pandas as pd
from pathlib import Path as Path
from functools import partial
import numpy as np


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
        print("superclass_probs ", superclass_probs)
        return superclass_probs
    print("pred_fn ", pred_fn)
    return pred_fn

def compute_average_confidence(cvs, indices, split='test', pred_fn=None):
    """
    Computes the average confidence for a list of sample indices.

    Args:
        cvs: Corevectors object containing predictions and outputs.
        indices (List[int]): List of sample indices.
        split (str): Dataset split ('train', 'val', or 'test').
        pred_fn (callable): Function to convert model output to probabilities (default: softmax).

    Returns:
        float: Average confidence (in %).
    """

    if pred_fn is None:
        pred_fn = partial(F.softmax, dim=0)

    total_conf = 0.0
    for idx in indices:
        _d = cvs._dss[split][idx]
        output = pred_fn(_d['output'])
        if not torch.is_tensor(output):
            output = torch.tensor(output)

        conf = output.max().item() * 100
        total_conf += conf

    if len(indices) == 0:
        return 0.0  # Avoid division by zero

    return total_conf / len(indices)


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

    drill_path = Path("/srv/newpenny/XAI/CN/data100cvdim300/drillers")
    drill_name = 'classifier'

    phs_path = Path("/srv/newpenny/XAI/CN/data100cvdim300/peepholes")
    phs_name = 'peepholes'

    # target_layers = [ 
    #     'features.1.conv.0.0', 'features.1.conv.1',
    #     'features.2.conv.0.0','features.2.conv.1.0', 'features.2.conv.2',
    #     'features.3.conv.0.0', 'features.3.conv.1.0', 'features.3.conv.2',
    #     'features.4.conv.0.0','features.4.conv.1.0', 'features.4.conv.2',
    #     'features.5.conv.0.0', 'features.5.conv.1.0','features.5.conv.2', 
    #     'features.6.conv.0.0', 'features.6.conv.1.0', 'features.6.conv.2', #B3
    #     'features.7.conv.0.0', 'features.7.conv.1.0', 'features.7.conv.2', #B3
    #     'features.8.conv.0.0', 'features.8.conv.1.0', 'features.8.conv.2', #B4
    #     'features.9.conv.0.0', 'features.9.conv.1.0', 'features.9.conv.2', #B4
    #     'features.10.conv.0.0', 'features.10.conv.1.0', 'features.10.conv.2', #B5
    #     'features.11.conv.0.0', 'features.11.conv.1.0','features.11.conv.2', #B5
    #     'features.12.conv.0.0', 'features.12.conv.1.0', 'features.12.conv.2', #B5
    #     'features.13.conv.0.0', 'features.13.conv.1.0', 'features.13.conv.2', #B5
    #     'features.14.conv.0.0', 'features.14.conv.1.0','features.14.conv.2', #B6
    #     'features.15.conv.0.0', 'features.15.conv.1.0', 'features.15.conv.2', #B6
    #     'features.16.conv.0.0', 'features.16.conv.1.0','features.16.conv.2', #B6
    #     'features.17.conv.0.0', 'features.17.conv.1.0', 'features.17.conv.2', #B7
    #     'features.18.0', 
    #     'classifier.1'
    # ]

    target_layers = [ 
        'features.1.conv.1',
        'features.2.conv.0.0','features.2.conv.1.0', 'features.2.conv.2',
        'features.3.conv.0.0', 'features.3.conv.1.0', 'features.3.conv.2',
        'features.4.conv.1.0', 
        'features.5.conv.1.0',
        'features.6.conv.1.0',  #B3
        'features.7.conv.0.0', #B3
        'features.8.conv.1.0', #B4
        'features.9.conv.1.0', #B4
        'features.10.conv.1.0', #B5
        'features.11.conv.0.0', 'features.11.conv.2', #B5
        'features.13.conv.0.0', 'features.13.conv.1.0', #B5
        'features.14.conv.1.0','features.14.conv.2', #B6
        'features.15.conv.0.0', 'features.15.conv.1.0', 'features.15.conv.2', #B6
        'features.16.conv.0.0', 'features.16.conv.1.0','features.16.conv.2', #B6
        'features.17.conv.0.0', 'features.17.conv.1.0', 'features.17.conv.2', #B7
        'features.18.0', 
        'classifier.1'
    ]
    
    cv_dim = 300
    n_cluster = 100  
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

    if superclass:
    #     cv_parsers = {
    #         'features.1.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.1.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.1.conv.1': partial(trim_corevectors,
    #                         module = 'features.1.conv.1',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.2.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.2.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.2.conv.1.0': partial(trim_corevectors,    
    #                         module = 'features.2.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.2.conv.2': partial(trim_corevectors,
    #                         module = 'features.2.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.3.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.3.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.3.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.3.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.3.conv.2': partial(trim_corevectors,
    #                         module = 'features.3.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.4.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.4.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.4.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.4.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.4.conv.2': partial(trim_corevectors,
    #                         module = 'features.4.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.5.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.5.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.5.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.5.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.5.conv.2': partial(trim_corevectors,
    #                         module = 'features.5.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'), 
    #         'features.6.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.6.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.6.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.6.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.6.conv.2': partial(trim_corevectors,
    #                         module = 'features.6.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.7.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.7.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.7.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.7.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.7.conv.2': partial(trim_corevectors,
    #                         module = 'features.7.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.8.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.8.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.8.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.8.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.8.conv.2': partial(trim_corevectors, 
    #                             module = 'features.8.conv.2',
    #                             cv_dim = cv_dim,
    #                             label_key = 'superclass'),
    #         'features.9.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.9.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.9.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.9.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.9.conv.2': partial(trim_corevectors,
    #                         module = 'features.9.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.10.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.10.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.10.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.10.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.10.conv.2': partial(trim_corevectors,
    #                         module = 'features.10.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.11.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.11.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.11.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.11.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.11.conv.2': partial(trim_corevectors,
    #                         module = 'features.11.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.12.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.12.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),    
    #         'features.12.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.12.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.12.conv.2': partial(trim_corevectors,
    #                         module = 'features.12.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.13.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.13.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),   
    #         'features.13.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.13.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.13.conv.2': partial(trim_corevectors,
    #                         module = 'features.13.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.14.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.14.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.14.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.14.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.14.conv.2': partial(trim_corevectors,
    #                         module = 'features.14.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.15.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.15.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.15.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.15.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.15.conv.2': partial(trim_corevectors,
    #                         module = 'features.15.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.16.conv.0.0': partial(trim_corevectors,
    #                         module = 'features.16.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.16.conv.1.0': partial(trim_corevectors,
    #                         module = 'features.16.conv.1.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.16.conv.2': partial(trim_corevectors,
    #                         module = 'features.16.conv.2',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.17.conv.0.0': partial(trim_corevectors,   
    #                         module = 'features.17.conv.0.0',
    #                         cv_dim = cv_dim,
    #                         label_key = 'superclass'),
    #         'features.17.conv.1.0': partial(trim_corevectors,	
    #                             module = 'features.17.conv.1.0',
    #                             cv_dim = cv_dim,
    #                             label_key = 'superclass'),
    #         'features.17.conv.2': partial(trim_corevectors,
    #                             module = 'features.17.conv.2',
    #                             cv_dim = cv_dim,
    #                             label_key = 'superclass'),
    #         'features.18.0': partial(trim_corevectors,
    #                             module = 'features.18.0',
    #                             cv_dim = cv_dim,
    #                             label_key = 'superclass'),
    #         'classifier.1': partial(trim_corevectors,
    #                             module = 'classifier.1',
    #                             cv_dim = cv_dim,
    #                             label_key = 'superclass'),                                  
    # }

        cv_parsers = {
        'features.1.conv.1': partial(trim_corevectors,  
                            module = 'features.1.conv.1',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.2.conv.0.0': partial(trim_corevectors,
                            module = 'features.2.conv.0.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.2.conv.1.0': partial(trim_corevectors,
                            module = 'features.2.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.2.conv.2': partial(trim_corevectors,
                            module = 'features.2.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.3.conv.0.0': partial(trim_corevectors,
                            module = 'features.3.conv.0.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.3.conv.1.0': partial(trim_corevectors,
                            module = 'features.3.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.3.conv.2': partial(trim_corevectors,
                            module = 'features.3.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.4.conv.1.0': partial(trim_corevectors,
                            module = 'features.4.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.5.conv.1.0': partial(trim_corevectors,
                            module = 'features.5.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'features.6.conv.1.0': partial(trim_corevectors,
                            module = 'features.6.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B3
        'features.7.conv.0.0': partial(trim_corevectors,
                            module = 'features.7.conv.0.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B3
        'features.8.conv.1.0': partial(trim_corevectors,
                            module = 'features.8.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B4
        'features.9.conv.1.0': partial(trim_corevectors,
                            module = 'features.9.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B4
        'features.10.conv.1.0': partial(trim_corevectors,
                            module = 'features.10.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B5
        'features.11.conv.0.0': partial(trim_corevectors,
                            module = 'features.11.conv.0.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B5
        'features.11.conv.2': partial(trim_corevectors,
                            module = 'features.11.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B5
        'features.13.conv.0.0': partial(trim_corevectors,
                            module = 'features.13.conv.0.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B5
        'features.13.conv.1.0': partial(trim_corevectors,
                            module = 'features.13.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B5
        'features.14.conv.1.0': partial(trim_corevectors,
                            module = 'features.14.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B6
        'features.14.conv.2': partial(trim_corevectors,
                            module = 'features.14.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B6
        'features.15.conv.0.0': partial(trim_corevectors,
                            module = 'features.15.conv.0.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B6
        'features.15.conv.1.0': partial(trim_corevectors,
                            module = 'features.15.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B6
        'features.15.conv.2': partial(trim_corevectors,
                            module = 'features.15.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B6
        'features.16.conv.0.0': partial(trim_corevectors,
                            module = 'features.16.conv.0.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B6
        'features.16.conv.1.0': partial(trim_corevectors,
                            module = 'features.16.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B6
        'features.16.conv.2': partial(trim_corevectors,
                            module = 'features.16.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B6
        'features.17.conv.0.0': partial(trim_corevectors,
                            module = 'features.17.conv.0.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B7
        'features.17.conv.1.0': partial(trim_corevectors,
                            module = 'features.17.conv.1.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B7
        'features.17.conv.2': partial(trim_corevectors,
                            module = 'features.17.conv.2',
                            cv_dim = cv_dim,
                            label_key = 'superclass'), #B7
        'features.18.0': partial(trim_corevectors,
                            module = 'features.18.0',
                            cv_dim = cv_dim,
                            label_key = 'superclass'),
        'classifier.1': partial(trim_corevectors,
                            module = 'classifier.1',
                            cv_dim = 100,
                            label_key = 'superclass'),
        }
                                        
    else:
        # cv_parsers = {
        # 'features.1.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.1.conv.0.0', cv_dim = cv_dim),
        # 'features.1.conv.1': partial(trim_corevectors,
        #                     module = 'features.1.conv.1', cv_dim = cv_dim),
        # 'features.2.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.2.conv.0.0', cv_dim = cv_dim),   
        # 'features.2.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.2.conv.1.0', cv_dim = cv_dim),
        # 'features.2.conv.2': partial(trim_corevectors,
        #                     module = 'features.2.conv.2', cv_dim = cv_dim),
        # 'features.3.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.3.conv.0.0', cv_dim = cv_dim),
        # 'features.3.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.3.conv.1.0', cv_dim = cv_dim),
        # 'features.3.conv.2': partial(trim_corevectors,
        #                     module = 'features.3.conv.2', cv_dim = cv_dim),
        # 'features.4.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.4.conv.0.0', cv_dim = cv_dim),
        # 'features.4.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.4.conv.1.0', cv_dim = cv_dim),
        # 'features.4.conv.2': partial(trim_corevectors,
        #                     module = 'features.4.conv.2', cv_dim = cv_dim),
        # 'features.5.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.5.conv.0.0', cv_dim = cv_dim),
        # 'features.5.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.5.conv.1.0', cv_dim = cv_dim),
        # 'features.5.conv.2': partial(trim_corevectors,
        #                     module = 'features.5.conv.2', cv_dim = cv_dim),
        # 'features.6.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.6.conv.0.0', cv_dim = cv_dim),
        # 'features.6.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.6.conv.1.0', cv_dim = cv_dim),
        # 'features.6.conv.2': partial(trim_corevectors,
        #                     module = 'features.6.conv.2', cv_dim = cv_dim),
        # 'features.7.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.7.conv.0.0', cv_dim = cv_dim),
        # 'features.7.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.7.conv.1.0', cv_dim = cv_dim),
        # 'features.7.conv.2': partial(trim_corevectors,
        #                     module = 'features.7.conv.2', cv_dim = cv_dim),
        # 'features.8.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.8.conv.0.0', cv_dim = cv_dim),
        # 'features.8.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.8.conv.1.0', cv_dim = cv_dim),
        # 'features.8.conv.2': partial(trim_corevectors,
        #                     module = 'features.8.conv.2', cv_dim = cv_dim),
        # 'features.9.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.9.conv.0.0', cv_dim = cv_dim),
        # 'features.9.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.9.conv.1.0', cv_dim = cv_dim),
        # 'features.9.conv.2': partial(trim_corevectors,
        #                     module = 'features.9.conv.2', cv_dim = cv_dim),
        # 'features.10.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.10.conv.0.0', cv_dim = cv_dim),   
        # 'features.10.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.10.conv.1.0', cv_dim = cv_dim),
        # 'features.10.conv.2': partial(trim_corevectors,
        #                     module = 'features.10.conv.2', cv_dim = cv_dim),
        # 'features.11.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.11.conv.0.0', cv_dim = cv_dim),
        # 'features.11.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.11.conv.1.0', cv_dim = cv_dim),
        # 'features.11.conv.2': partial(trim_corevectors,
        #                     module = 'features.11.conv.2', cv_dim = cv_dim),
        # 'features.12.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.12.conv.0.0', cv_dim = cv_dim),
        # 'features.12.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.12.conv.1.0', cv_dim = cv_dim),
        # 'features.12.conv.2': partial(trim_corevectors,
        #                     module = 'features.12.conv.2', cv_dim = cv_dim),
        # 'features.13.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.13.conv.0.0',cv_dim = cv_dim),
        # 'features.13.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.13.conv.1.0', cv_dim = cv_dim),
        # 'features.13.conv.2': partial(trim_corevectors,
        #                     module = 'features.13.conv.2', cv_dim = cv_dim),
        # 'features.14.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.14.conv.0.0', cv_dim = cv_dim),
        # 'features.14.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.14.conv.1.0', cv_dim = cv_dim),
        # 'features.14.conv.2': partial(trim_corevectors,
        #                     module = 'features.14.conv.2', cv_dim = cv_dim),
        # 'features.15.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.15.conv.0.0', cv_dim = cv_dim),
        # 'features.15.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.15.conv.1.0', cv_dim = cv_dim),
        # 'features.15.conv.2': partial(trim_corevectors,
        #                     module = 'features.15.conv.2', cv_dim = cv_dim),
        # 'features.16.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.16.conv.0.0', cv_dim = cv_dim),
        # 'features.16.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.16.conv.1.0', cv_dim = cv_dim),
        # 'features.16.conv.2': partial(trim_corevectors,
        #                     module = 'features.16.conv.2', cv_dim = cv_dim),
        # 'features.17.conv.0.0': partial(trim_corevectors,
        #                     module = 'features.17.conv.0.0', cv_dim = cv_dim),
        # 'features.17.conv.1.0': partial(trim_corevectors,
        #                     module = 'features.17.conv.1.0', cv_dim = cv_dim),
        # 'features.17.conv.2': partial(trim_corevectors,
        #                     module = 'features.17.conv.2', cv_dim = cv_dim),
        # 'features.18.0': partial(trim_corevectors,
        #                     module = 'features.18.0', cv_dim = cv_dim),
        # 'classifier.1': partial(trim_corevectors,
        #                     module = 'classifier.1', cv_dim = 100),
        # }
        cv_parsers = {
            'features.1.conv.1': partial(trim_corevectors,
                            module = 'features.1.conv.1',
                            cv_dim = cv_dim),
            'features.2.conv.0.0': partial(trim_corevectors,
                            module = 'features.2.conv.0.0',
                            cv_dim = cv_dim),
            'features.2.conv.1.0': partial(trim_corevectors,
                            module = 'features.2.conv.1.0',
                            cv_dim = cv_dim),
            'features.2.conv.2': partial(trim_corevectors,
                            module = 'features.2.conv.2',
                            cv_dim = cv_dim),
            'features.3.conv.0.0': partial(trim_corevectors,
                            module = 'features.3.conv.0.0',
                            cv_dim = cv_dim),
            'features.3.conv.1.0': partial(trim_corevectors,
                            module = 'features.3.conv.1.0',
                            cv_dim = cv_dim),
            'features.3.conv.2': partial(trim_corevectors,
                            module = 'features.3.conv.2',
                            cv_dim = cv_dim),
            'features.4.conv.1.0': partial(trim_corevectors,
                            module = 'features.4.conv.1.0',
                            cv_dim = cv_dim),
            'features.5.conv.1.0': partial(trim_corevectors,
                            module = 'features.5.conv.1.0',
                            cv_dim = cv_dim),
            'features.6.conv.1.0': partial(trim_corevectors,
                            module = 'features.6.conv.1.0',
                            cv_dim = cv_dim),
            'features.7.conv.0.0': partial(trim_corevectors,
                            module = 'features.7.conv.0.0',
                            cv_dim = cv_dim),
            'features.8.conv.1.0': partial(trim_corevectors,
                            module = 'features.8.conv.1.0',
                            cv_dim = cv_dim),
            'features.9.conv.1.0': partial(trim_corevectors,
                            module = 'features.9.conv.1.0',
                            cv_dim = cv_dim),
            'features.10.conv.1.0': partial(trim_corevectors,
                            module = 'features.10.conv.1.0',
                            cv_dim = cv_dim),
            'features.11.conv.0.0': partial(trim_corevectors,
                            module = 'features.11.conv.0.0',
                            cv_dim = cv_dim),
            'features.11.conv.2': partial(trim_corevectors,
                            module = 'features.11.conv.2',
                            cv_dim = cv_dim),      
            'features.13.conv.0.0': partial(trim_corevectors,
                            module = 'features.13.conv.0.0',
                            cv_dim = cv_dim), 
            'features.13.conv.1.0': partial(trim_corevectors,
                            module = 'features.13.conv.1.0',
                            cv_dim = cv_dim),
            'features.14.conv.1.0': partial(trim_corevectors,
                            module = 'features.14.conv.1.0',
                            cv_dim = cv_dim),
            'features.14.conv.2': partial(trim_corevectors,
                            module = 'features.14.conv.2',
                            cv_dim = cv_dim),  
            'features.15.conv.0.0': partial(trim_corevectors,
                            module = 'features.15.conv.0.0',
                            cv_dim = cv_dim),    
            'features.15.conv.1.0': partial(trim_corevectors,
                            module = 'features.15.conv.1.0',
                            cv_dim = cv_dim),
            'features.15.conv.2': partial(trim_corevectors,
                            module = 'features.15.conv.2',
                            cv_dim = cv_dim),
            'features.16.conv.0.0': partial(trim_corevectors,
                            module = 'features.16.conv.0.0',
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
            'features.17.conv.1.0': partial(trim_corevectors,
                            module = 'features.17.conv.1.0',
                            cv_dim = cv_dim),
            'features.17.conv.2': partial(trim_corevectors,
                            module = 'features.17.conv.2',
                            cv_dim = cv_dim),
            'features.18.0': partial(trim_corevectors,
                            module = 'features.18.0',
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
    # Load fine to coarse index from CIFAR-100 meta
    meta_path = '/srv/newpenny/dataset/CIFAR100/cifar-100-python/meta'
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f, encoding='latin1')
    fine_label_names = meta['fine_label_names']
    coarse_label_names = meta['coarse_label_names']

    fine_to_coarse_index = {}
    for coarse_idx, coarse_name in enumerate(coarse_label_names):
        for fine_name in coarse_to_fine[coarse_name]:
            fine_idx = fine_label_names.index(fine_name)
            fine_to_coarse_index[fine_idx] = coarse_idx

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
        
        samples = get_filtered_samples(
            ds = ds,
            corevectors = cv,
            #class_name = 'trees',
            correct = True,
            #conf_range = [80,100],
        )
        print ('samples: ', samples)
        #print(compute_average_confidence(cv, samples))
        quit()

        # print(samples)
        # quit()

        #get_emp_coverage_scores(ph._drillers, 0.8)
        #empirical_posterior_heatmaps(ph._drillers, '/home/claranunesbarrancos/repos/XAI/src/mobilenet/empp/300cvdim_150clusters')
        get_conceptogram(
            path = Path('/home/claranunesbarrancos/repos/XAI/src/mobilenet/conceptos'),
            name = 'flower_bestlayers',
            corevectors = cv,
            peepholes = ph,
            portion = 'test',
            samples = [367, 547, 798, 1236, 1277, 1598, 1631],
            target_layers = target_layers,
            classes = coarse_label_names if superclass else ds.get_classes(),
            ticks = target_layers,
            krows = 5,
            # label_key = 'superclass',
            # pred_fn = superclass_pred_fn(),
            ds = ds,
        )






