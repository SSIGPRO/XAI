import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# python stuff
from pathlib import Path as Path
from numpy.random import randint
from time import time
import pickle
from itertools import product
import matplotlib.pyplot as plt

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.coreVectors.svd_coreVectors import reduct_matrices_from_svds as parser_fn
from peepholelib.classifier.classifier_base import trim_corevectors, map_labels
from peepholelib.classifier.tkmeans import KMeans as tKMeans 
from peepholelib.classifier.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.testing import trim_dataloaders

# torch stuff
import torch
import torchvision
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda

#####################################################
#### .py to extract peepholes with Livias code   ####
#### To visualize them use xp_phs_cgs_SARA.ipynb ####
#####################################################

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


if __name__ == "__main__":
    # gpu selection
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    # utils
    seed = 42
    bs = 512    #256
    verbose = True 

    #--------------------------------
    # path
    #--------------------------------
    dataset = 'CIFAR100' 
    ds_path = f'/srv/newpenny/dataset/{dataset}'

    # model parameters
    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'
    
    # data_dir = '/srv/newpenny/XAI/generated_data'
    #data_dir = '/home/saravorabbi/'

    fold = 'vit_1'

    svd_path = f'/home/saravorabbi/Documents/{fold}'
    svd_name = 'svd'

    cvs_path = Path(f'/home/saravorabbi/Documents/{fold}/corevectors')
    cvs_name = 'corevectors'

    
    cls_type = 'tGMM' # 'tKMeans'
    
    superclass = False
    
    phs_name = 'peepholes'
    #phs_path = '/home/saravorabbi/Desktop/new_ph'          # PATH TO MODIFY IF YOU WANT TO CREATE NEW PEEPHOLES
    #results_dir = '/home/saravorabbi/Desktop/new_ph'
    phs_path = '/home/saravorabbi/Desktop/again_ph'
    results_dir = '/home/saravorabbi/Desktop/again_ph'





    # if superclass:
    #     phs_path = Path(data_dir)/f'peepholes/{dataset}/{model_id}/{cls_type}/superclass'
    #     with open(Path(model_dir)/'superclass_mapping_CIFAR100.pkl', 'rb') as f:
    #         mapping_dict = pickle.load(f)
    # else:
    #     phs_path = Path(data_dir)/f'peepholes/{dataset}/{model_id}/{cls_type}'
    #     sarebbe -> /srv/newpenny/XAI/generated_data/peepholes/CIFAR100/vgg16/tKMeans
    

    
    #--------------------------------
    # Dataset 
    #--------------------------------
    
    # pretrained = True


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
    # Model wrap
    #--------------------------------

    nn = torchvision.models.vit_b_16()
    in_features = nn.heads.head.in_features
    nn.heads.head = torch.nn.Linear(in_features, 100)

    wrap = ModelWrap(device=device)
    wrap.set_model(
        model = nn,
        path = model_dir,
        name = model_name
    )

    #target_layers = ['encoder.layers.encoder_layer_10.mlp.0', 'encoder.layers.encoder_layer_11.mlp.0']
    target_layers = get_st_list(nn.state_dict().keys())
    wrap.set_target_layers(target_layers=target_layers)



    # commented code: dry run - svd - corevectors
    '''
    direction = {'save_input':True, 'save_output':False}
    model.add_hooks(**direction, verbose=False) 
    
    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    model.dry_run(x=dry_img)

    #--------------------------------
    # SVDs 
    #--------------------------------
    print('target layers: ', model.get_target_layers()) 
    model.get_svds(path=svds_path, name=svds_name, verbose=verbose)
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    ds_loaders = ds.get_dataset_loaders()
    #ds_loaders = trim_dataloaders(ds.get_dataset_loaders(), 0.05)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )

    with corevecs as cv: 
        # copy dataset to coreVect dataset
        cv.get_coreVec_dataset(
                loaders = ds_loaders, 
                verbose = verbose
                ) 

        cv.get_activations(
                batch_size = bs,
                loaders = ds_loaders,
                verbose = verbose
                )

        cv.get_coreVectors(
                batch_size = bs,
                reduct_matrices = model._svds,
                parser = parser_fn,
                verbose = verbose
                )

        cv_dl = cv.get_dataloaders(verbose=verbose)
    
        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['train']:
            print(data['coreVectors'][target_layers[0]][34:56,:])
            i += 1
            if i == 3: break

        cv.normalize_corevectors(
                wrt='train',
                verbose=verbose
                )
        i = 0
        print('after norm')
        for data in cv_dl['train']:
            print(data['coreVectors'][target_layers[0]][34:56,:])
            i += 1
            if i == 3: break
    quit()
    '''
    
    
    #--------------------------------
    # Peepholes
    #--------------------------------


    # if superclass:
    #     results_dir = Path(data_dir) / f'results_{dataset}_superclass'
    # else:
    #     results_dir = Path(data_dir) / f'results_{dataset}'

    # ps = [2**i for i in range(4, 7)] + [100] # used for classifier.6
    # ncls = [20] + [2**i for i in range(5, 7)] # used for classifier.6
    
    ps = 200
    ncls = 150
    #all_combinations = list(product(ps, ncls))
    all_combinations = [(ps, ncls)]


    if superclass:
        n_classes = len(mapping_dict.keys())
    else:
        n_classes = len(ds.get_classes())


    for peep_size, n_cls in all_combinations:

        ph_config_name = phs_name+f'.{peep_size}.{n_cls}'

        print('ph config name: ', ph_config_name)

        for layer in target_layers:
            parser_cv = trim_corevectors
            parser_kwargs = {'layer': layer, 'peep_size':peep_size}
            cls_kwargs = {} # {'batch_size':256}
            if superclass:
                superclass_parser = map_labels
                superclass_kwargs = {'label_mapping': mapping_dict}
            else:
                superclass_parser = None
                superclass_kwargs = {}
            
            if cls_type=='tGMM':
                cls = tGMM(
                    nl_classifier = n_cls,
                    nl_model = n_classes,
                    parser = parser_cv,
                    parser_kwargs = parser_kwargs,
                    cls_kwargs = cls_kwargs,
                    superclass_parser = map_labels, # new
                    superclass_kwargs = superclass_kwargs, # new
                    device = device
                    )
            elif cls_type=='tKMeans':
                cls = tKMeans(
                    nl_classifier = n_cls,
                    nl_model = n_classes,
                    parser = parser_cv,
                    parser_kwargs = parser_kwargs,
                    cls_kwargs = cls_kwargs,
                    superclass_parser = map_labels, # new
                    superclass_kwargs = superclass_kwargs, # new
                    device = device
                    )

            corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                device = device 
                )
            
            peepholes = Peepholes(
                path = phs_path,
                # name = phs_name,
                name = ph_config_name,
                classifier = cls,
                # classifier = None,
                layer = layer,
                device = device
                )

            with corevecs as cv, peepholes as ph:
                cv.load_only(
                    loaders = ['train', 'test', 'val'],
                    verbose = True
                    ) 
                    
                cv_dl = cv.get_dataloaders(
                    batch_size = bs,
                    verbose = True,
                    )
                
                # i = 0
                # print('\nPrinting some corevecs')
                # for data in cv_dl['val'].dataset['coreVectors']:
                #     print('cvs\n', data[layer])
                #     i += 1
                #     if i == 3: break
                
                t0 = time()
                cls.fit(dataloader = cv_dl['train'], verbose=True)
                print('Fitting time = ', time()-t0)
                
                cls.compute_empirical_posteriors(verbose=verbose)
                
                plt.figure()
                plt.imshow(cls._empp)
                stringa = f'/empp_{cls_type}_{layer}_{ph_config_name}.png'
                print('il path per la immagine è: ', results_dir + stringa)
                plt.savefig(results_dir + stringa)
        
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
                
                print('tutto ok')


        
                # i = 0
                # print('\nPrinting some peeps')
                # ph_dl = ph.get_dataloaders(verbose=verbose)
                # for data in ph_dl['val']:
                #     print('phs\n', data[layer]['peepholes'])
                #     print('max\n', data[layer]['score_max'])
                #     print('ent\n', data[layer]['score_entropy'])
                #     i += 1
                #     if i == 3: break

 
