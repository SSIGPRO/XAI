import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())


# torch stuff
import torch
from tensordict import MemoryMappedTensor as MMT

# our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.conceptograms import *
from nb_utils import *

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

    use_cuda = torch.cuda.is_available()
    cuda_index = 0
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

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
    cvs_path = Path('/home/claranunesbarrancos/repos/data/corevectors')
    cvs_name = 'corevectors'
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        device = device 
        )
    #--------------------------------
    # Peepholes
    #--------------------------------

    phs_path = Path('/home/claranunesbarrancos/repos/data/peepholes')
    phs_name = 'peepholes'

    drill_path = Path('/home/claranunesbarrancos/repos/data/drillers')
    drill_name = 'classifier'

    target_layers = [ 'features.13.conv.1.0', 'features.14.conv.1.0', 'features.15.conv.1.0', 'features.16.conv.1.0', 'features.16.conv.2', 'features.17.conv.0.0']
    
    n_clusters = 100
    n_classes = 100
    cv_dim = 10
    ph_config_names = {'peepholes.ps_10.nc_10'}

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

    
    for ph_config_name in ph_config_names:
        peepholes = Peepholes(
        path = phs_path,
        name = phs_name,
        driller = drillers,
        target_modules = target_layers,
        device = device
        )

    #--------------------------------
    # Conceptograms
    #--------------------------------

    save_path = '/home/claranunesbarrancos/repos/XAI/conceptograms'

    generate_conceptograms(
        peepholes=peepholes,
        save_path=save_path,
        ph_config_names=ph_config_names,
        target_layers=target_layers,
        n_classes=n_classes,
    )
    generate_conceptogram(
        peepholes=peepholes,
        sample=89, # [0,99]
        save_path=save_path,
        ph_config_names=ph_config_names,
        target_layers=target_layers,
        n_classes=n_classes,
    )
    
    generate_sample_conceptogram(
        peepholes=peepholes,
        sample=89, # [0,99]
        ds=ds,
        corevecs=corevecs,
        result=1,
        confidence=0,
        save_path=save_path,
        ph_config_names=ph_config_names,
        target_layers=target_layers,
        n_classes=n_classes,
    )
    
    generate_sample_conceptograms(
        peepholes=peepholes,
        sample=89, # [0,99]
        ds=ds,
        corevecs=corevecs,
        result=1,
        confidence=0,
        save_path=save_path,
        ph_config_names=ph_config_names,
        target_layers=target_layers,
        n_classes=n_classes,
    )




