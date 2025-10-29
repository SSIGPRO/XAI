import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.viz import viz_singular_values

from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear_ViT, svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 
from peepholelib.utils.analyze import evaluate, evaluate_dists 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
from cuda_selector import auto_cuda

if __name__ == "__main__":
    # use_cuda = torch.cuda.is_available()
    # device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    # print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 1
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    pretrained = True
    dataset = 'CIFAR100' 
    name_model = 'ViT'
    #name_model = 'vgg16'
    seed = 29
    bs = 64 
    n_threads = 32
    n_cluster = 150
    cv_dim = 100

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    #model_name = 'SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'
    
    svds_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv') #Path.cwd()/'../data' 
    svds_name = f'svds' 
    
    #cvs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/corevectors')
    cvs_path = Path.cwd()/f'../data/{name_model}/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/corevectors') #
    cvs_name = 'corevectors' # 'corevectors'

    #drill_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/drillers')
    drill_path = Path.cwd()/f'../data/{name_model}/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/drillers_on_class') #
    drill_name = 'classifier'

    #phs_path = Path('/srv/newpenny/XAI/generated_data/temp_results/toeplitz_conv/peepholes')
    phs_path = Path.cwd()/f'../data/{name_model}/peepholes_{cv_dim}' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/peepholes_on_class') #
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
    
#     nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    
    nn = vit_b_16()
    n_classes = len(ds.get_classes())

    model = ModelWrap(
            model = nn,
            device = device
            )

    model.update_output(
            output_layer = 'classifier.6', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    
    target_layers = [ f'encoder.layers.encoder_layer_{i}.mlp.{j}'for i in range(12) for j in [0,3]]

    target_layers.append('heads.head')
    
    
#     target_layers = [
#             #'features.0',
#             #'features.2',
#             #'features.5',
#             #'features.7',
#             'features.10',
#             'features.12',
#             'features.14',
#             'features.17',
#             'features.19',
#             'features.21',
#             'features.24',
#             'features.26',
#             'features.28',
#             'classifier.0',
#             'classifier.3',
#             'classifier.6',
#             ]

    model.set_target_modules(target_modules=target_layers, verbose=verbose)

    #--------------------------------
    # SVDs 
    #--------------------------------
    
    model.get_svds(
            path = svds_path,
            name = svds_name,
            target_modules = target_layers,
            sample_in = ds._dss['train'][0][0],
            rank = 10,
            channel_wise = True,
            verbose = verbose
            )

#     for k in model._svds.keys():
#         for kk in model._svds[k].keys():
#             print('svd shapes: ', k, kk, model._svds[k][kk].shape)
#         s = model._svds[k]['s']
#         if len(s.shape) == 1:
#             print('-----------------------')
#             print('HIM HERE')
#             print('-----------------------')
#             plt.figure()
#             plt.plot(s, '-')
#             plt.xlabel('Rank')
#             plt.ylabel('EigenVec')
#         else:
#             fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#             _x = torch.linspace(0, s.shape[1]-1, s.shape[1])
#             for r in range(s.shape[0]):
#                 plt.plot(xs=_x, ys=s[r,:], zs=r, zdir='y')
#             ax.set_xlabel('Rank')
#             ax.set_ylabel('Channel')
#             ax.set_zlabel('EigenVec')
#         plt.savefig(f'prova_{k}.png')
#         plt.savefig((svds_path/(svds_name+'/'+k+'.png')).as_posix(), dpi=300, bbox_inches='tight')
#         plt.close()
    
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #random_subsampling(ds, 0.025)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer

    reduction_fns = {}
    for _layer in target_layers:
        if 'features' in _layer:
            reduction_fns[_layer] =  partial(
               svd_Conv2D, 
               reduct_m=model._svds[_layer]['Vh'], 
               layer=model._target_modules[_layer], 
               device=device
               ),
        elif 'classifier' in _layer:
            reduction_fns[_layer] = partial(
                svd_Linear,
                reduct_m=model._svds[_layer]['Vh'], 
                device=device
                ),
    print('reduction_fns = ', reduction_fns.keys())
    
#     reduction_fns = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': partial(
#                 svd_Linear_ViT,
#                 reduct_m=model._svds[f'encoder.layers.encoder_layer_{i}.mlp.{j}']['Vh'], 
#                 device=device
#                 ) for i in range(12) for j in [0,3]}
#     reduction_fns['heads.head'] = partial(svd_Linear,
#                                         reduct_m=model._svds['heads.head']['Vh'], 
#                                         device=device
#                                         )
    
    with corevecs as cv: 
        cv.parse_ds(
                batch_size = bs,
                datasets = ds,
                n_threads = n_threads,
                verbose = verbose
                )
        
        '''
        # This occupies a lot of space. Only do if you need it
        # copy dataset to activatons file
        cv.get_activations(
                batch_size = bs,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                )        
        '''

        # computing the corevectors
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                n_threads = n_threads,
                save_input = True,
                save_output = False,
                verbose = verbose
                )
        
        # cv_dl = cv.get_dataloaders(verbose=verbose)
    
        # i = 0
        # print('\nPrinting some corevecs')
        # for data in cv_dl['test']:
        #     print('\nencoder.layers.encoder_layer_9.mlp.0')
        #     print(data['encoder.layers.encoder_layer_9.mlp.0'].shape)
            
        #     print(data['encoder.layers.encoder_layer_9.mlp.0'][0][:64])
        #     i += 1
        #     if i == 1: break

        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
            cv.normalize_corevectors(
                    wrt = 'train',
                    #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                    to_file = cvs_path/(cvs_name+'.normalization.pt'),
                    batch_size = bs,
                    n_threads = n_threads,
                    verbose=verbose
                    )
        
        # i = 0
        # print('after norm')
        # for data in cv_dl['test']:
        #     print(data['encoder.layers.encoder_layer_9.mlp.0'].shape)
            
        #     print(data['encoder.layers.encoder_layer_9.mlp.0'][0][:64])
        #     i += 1
        #     if i == 1: break
    
     
    #--------------------------------
    # Peepholes
    #--------------------------------

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    #cv_parsers = {
        # 'features.24': partial(
        #         trim_corevectors,
        #         module = 'features.24',
        #         cv_dim = 100,
        #         # cols = [0] 
        #         ),
        #     'features.26': partial(
        #         trim_corevectors,
        #         module = 'features.26',
        #         cv_dim = 100,
        #         # cols = [0] 
        #         ),
        #     'features.28': partial(
        #         trim_corevectors,
        #         module = 'features.28',
        #         cv_dim = 100,
        #         # cols = [0]
        #         ),
            # 'classifier.0': partial(
            #     trim_corevectors,
            #     module = 'classifier.0',
            #     cv_dim = 100
            #     ),
            # 'classifier.3': partial(
            #     trim_corevectors,
            #     module = 'classifier.3',
            #     cv_dim = 100
            #     ),
            # 'classifier.6': partial(
            #     trim_corevectors,
            #     module = 'classifier.6',
            #     cv_dim = 100
                
            #     ),
            # }

    cv_parsers = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': partial(
                trim_corevectors,
                module = f'encoder.layers.encoder_layer_{i}.mlp.{j}',
                cv_dim = cv_dim
                ) for i in range(12) for j in [0,3]} 
    
    cv_parsers['heads.head'] = partial(
                trim_corevectors,
                module = f'heads.head',
                cv_dim = cv_dim
                )
    

    drillers = {}

    feature_sizes = { f'encoder.layers.encoder_layer_{i}.mlp.{j}': cv_dim for i in range(12) for j in [0,3]}

    feature_sizes['heads.head'] = cv_dim 

    feature_sizes = {
            'features.24': 100, #1*model._svds['features.24']['Vh'].shape[0],
            'features.26': 100, #1*model._svds['features.26']['Vh'].shape[0],
            'features.28': 100, #1*model._svds['features.28']['Vh'].shape[0],
            'classifier.0': 100,
            'classifier.3': 100,
            'classifier.6': 100,
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
            name = f'{phs_name}.nc_{n_cluster}',
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