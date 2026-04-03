import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# torch
import torch
import torchvision

# python
import numpy as np
from pathlib import Path as Path
from time import time

# peepholelib
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from peepholelib.utils.testing import trim_dataloaders
from peepholelib.classifier.classifier_base import trim_corevectors
from peepholelib.classifier.tgmm import GMM as tGMM
from peepholelib.peepholes.peepholes import Peepholes

############################################################################################
#### In this .py file we compute COREVECTORS for every layer and for the entire dataset ####
############################################################################################

def get_st_list(state_dict):
    '''
    Return a clean list of the layers of the model

    Args:
    - state_dict: state dict of the model

    Return:
    - st_sorted: list of the name of the layers 
    '''
    print('getting all the layers we want')
    state_dict_list = list(state_dict)

    # remove .weight and .bias from the strings in the state_dict list
    st_clean = [s.replace(".bias", "").replace(".weight", "") for s in state_dict_list]
    st_sorted = sorted(list(set(st_clean)))
    filtered_layers = [layer for layer in st_sorted if 'mlp.0' in layer]# or 
                                                       # 'mlp.3' in layer or 
                                                       # 'heads' in layer]
    print('FILTERED LAYERS = ', filtered_layers)
    return filtered_layers


def get_st_list_enc0_5_mlp0(state_dict):
    
    state_dict_list = list(state_dict)

    # remove .weight and .bias from the strings in the state_dict list
    st_clean = [s.replace(".bias", "").replace(".weight", "") for s in state_dict_list]
    st_sorted = sorted(list(set(st_clean)))
    
    # mlp.0 from encoder_layer 0 to 5
    filtered_layers = [key for key in st_sorted if 'mlp.0' in key and any(f'encoder_layer_{i}.' in key for i in range(6))]
    
    print('FILTERED LAYERS = ', filtered_layers)
    return filtered_layers

def get_st_list_enc6_11_mlp0(state_dict):
    
    state_dict_list = list(state_dict)

    # remove .weight and .bias from the strings in the state_dict list
    st_clean = [s.replace(".bias", "").replace(".weight", "") for s in state_dict_list]
    st_sorted = sorted(list(set(st_clean)))
    
    # mlp.0 from encoder_layer 6 to 11
    filtered_layers = [key for key in st_sorted if 'mlp.0' in key and any(f'encoder_layer_{i}' in key for i in np.arange(6, 12))]
    
    print('FILTERED LAYERS = ', filtered_layers)
    return filtered_layers



if __name__ == "__main__":
    # gpu selection
    use_cuda = torch.cuda.is_available()
    cuda_index = 5
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    seed = 42
    verbose = True
    bs = 64                             # should be 254 at least -> 512 otherwise too slow (3 days to compute corevectors)

    # path
    dataset = 'CIFAR100'
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    model_dir = '/srv/newpenny/XAI/models/'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    fold = 'vit_1'
    # fold = 'tmux_terminal_prova'

    svd_path = f'/home/saravorabbi/Documents/{fold}'
    svd_name = 'svd'

    cvs_path = Path(f'/home/saravorabbi/Documents/{fold}/corevectors')
    cvs_name = 'corevectors'

    # dataset
    ds = Cifar(
        dataset=dataset,
        data_path=ds_path
    )

    ds.load_data(
        dataset=dataset,
        batch_size=64,                  # should be 254 at least -> 512
        data_kwargs = {'num_workers': 8, 'pin_memory': True},
        seed=seed
    )

    # model wrap
    nn = torchvision.models.vit_b_16()
    in_features = nn.heads.head.in_features
    nn.heads.head = torch.nn.Linear(in_features, 100)

    wrap = ModelWrap(device=device)
    wrap.set_model(
        model = nn,
        path = model_dir,
        name = model_name
    )

    target_layers = get_st_list_enc6_11_mlp0(nn.state_dict().keys())
    # target_layers = ['encoder.layers.encoder_layer_0.mlp.0']
    

    wrap.set_target_layers(target_layers=target_layers)
    
    direction = {'save_input':True, 'save_output':True}
    wrap.add_hooks(verbose=verbose)

    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    wrap.dry_run(x=dry_img)


    # svd
    wrap.get_svds(path=svd_path, name=svd_name)


    # corevectors
    
    ds_loaders = ds.get_dataset_loaders()                   # compute corevectors for the entire dataset
    # ds_loaders = trim_dataloaders(ds.get_dataset_loaders(), 0.03)

    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = wrap,
        device = device
        )

    with corevecs as cv:
        cv.get_coreVec_dataset(
            loaders = ds_loaders,
            verbose = verbose
        )
        cv.get_activations(
            batch_size = 64,                # should be 254 at least -> 512
            loaders = ds_loaders,
            verbose = verbose
        )
        cv.get_coreVectors(
            batch_size = 64,                # should be 254 at least -> 512
            reduct_matrices = wrap._svds,
            parser = parser_fn,
            verbose = verbose
        )
        cv_dl = cv.get_dataloaders(verbose=verbose)

        cv.normalize_corevectors(
            wrt='train',
            #from_file=cvs_path/(cvs_name+'.normalization.pt'),
            to_file=cvs_path/(cvs_name+'.normalization2.pt'),
            verbose=verbose
        )
        