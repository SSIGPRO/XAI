import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# torch
import torch
import torchvision

# python
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

##########################################################################################
#### In this .py file we compute PEEPHOLES for every layer and for the entire dataset ####
##########################################################################################

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
    

if __name__ == "__main__":
    # gpu selection
    use_cuda = torch.cuda.is_available()
    cuda_index = 3  # torch.cuda.device_count() -1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    seed = 42
    verbose = True
    bs = 512

    # path
    dataset = 'CIFAR100'
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    model_dir = '/srv/newpenny/XAI/models/'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    fold = 'vit_1'

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
        batch_size=512,
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

    target_layers = get_st_list(nn.state_dict().keys())

    wrap.set_target_layers(target_layers=target_layers)
    
    direction = {'save_input':True, 'save_output':True}
    wrap.add_hooks(verbose=verbose)

    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    wrap.dry_run(x=dry_img)

    # svd
    wrap.get_svds(path=svd_path, name=svd_name)
    
    #######################################################################################

    # ----------
    # peepholes computed only for mlp 0
    # ----------

    n_classes = 100
    parser_cv = trim_corevectors
    peep_size = 200
    n_cluster = 50


    corevecs = CoreVectors( 
        path = cvs_path,
        name = cvs_name,
        )

    phs_path = Path(f'/home/saravorabbi/Documents/{fold}/peepholes_ps_200_nc_{n_cluster}_full')
    phs_name = 'peepholes'

    print(f'SEARCHING NUMBER OF CLUSTER = {n_cluster}')
        
    # context manager
    with corevecs as cv:               # load only dei cv
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
        ) 
        cv_dl = cv.get_dataloaders(
            batch_size = bs,
            verbose = True,
        )
        for peep_layer in target_layers:
    
            parser_kwargs = {'layer': peep_layer, 'peep_size':peep_size}
        
            cls_kwargs = {}#{'batch_size':256} 
        
            cls = tGMM(
                    nl_classifier = n_cluster,
                    nl_model = n_classes,
                    parser = parser_cv,
                    parser_kwargs = parser_kwargs,
                    cls_kwargs = cls_kwargs,
                    device = device
                    )
        
            peepholes = Peepholes(
                    path = phs_path,
                    name = phs_name+'.'+peep_layer,
                    classifier = cls,
                    layer = peep_layer,
                    device = device
                    )
            
            t0 = time()
            cls.fit(dataloader = cv_dl['train'], verbose=verbose)
            print('Fitting time = ', time()-t0)
            
            cls.compute_empirical_posteriors(verbose=verbose)
            
            with peepholes as ph:
                ph.get_peepholes(
                    loaders = cv_dl,
                    verbose = verbose
                )
        
                ph.get_scores(
                    batch_size = bs,
                    verbose=verbose
                )
    
                ph.evaluate_dists(
                    score_type = 'max',
                    coreVectors = cv_dl,
                    bins = 20
                )

