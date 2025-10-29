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
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import pandas as pd


from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
from cuda_selector import auto_cuda


if __name__ == "__main__":
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 3
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
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
    
    svds_path = Path.cwd()/'../data' #Path('/srv/newpenny/XAI/Peephole-Analysis')  
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'../data/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/corevectors') #
    cvs_name = 'corevectors' # 'corevectors'

    drill_path = Path.cwd()/'../data/drillers/tuning_clusters' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/drillers_on_class') #
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/peepholes_on_class') #
    phs_name = 'peepholes'
    
    verbose = True 
    num_workers = 4
    
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
           'features.24',
           'features.26',
           'features.28',
        #    'classifier.0',
        #    'classifier.3',
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
            path = svds_path,
            channel_wise = True,
            name = svds_name,
            verbose = verbose
            )

    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)
        s = model._svds[k]['s']
        if len(s.shape) == 1:
            plt.figure()
            plt.plot(s, '-')
            plt.xlabel('Rank')
            plt.ylabel('EigenVec')
        else:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            _x = torch.linspace(0, s.shape[1]-1, s.shape[1])
            for r in range(s.shape[0]):
                plt.plot(xs=_x, ys=s[r,:], zs=r, zdir='y')
            ax.set_xlabel('Rank')
            ax.set_ylabel('Channel')
            ax.set_zlabel('EigenVec')
        plt.savefig(f'prova_{k}.png')
        # plt.savefig((svds_path/(svds_name+'/'+k+'.png')).as_posix(), dpi=300, bbox_inches='tight')
        plt.close()
        
    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100
    list_clusters = [10, 30, 50, 70, 90, 110, 130, 150, 170, 190, 200]
    results = []
    for n_cluster in list_clusters:
#     cv_dim = 512
        parser_cv = trim_corevectors
        peep_layers = [
            'features.24',
            'features.26',
            'features.28',
            #  'classifier.0', 
            #  'classifier.3', 
            ]
        
        cls_kwargs = {}#{'batch_size': bs} 

        drillers = {}
        corevecs = CoreVectors(
                path = cvs_path,
                name = cvs_name,
                )
        
        cv_parsers = {
            'features.24': partial(
                    trim_corevectors,
                    module = 'features.24',
                    cv_dim = 100,
                    # cols = [0] 
                    ),
                'features.26': partial(
                    trim_corevectors,
                    module = 'features.26',
                    cv_dim = 100,
                    # cols = [0] 
                    ),
                'features.28': partial(
                    trim_corevectors,
                    module = 'features.28',
                    cv_dim = 100,
                    # cols = [0]
                    ),
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
                
                }
        feature_sizes = {
                # for channel_wise corevectors, the size is n_channels * cv_dim
                'features.24': 100, #1*model._svds['features.24']['Vh'].shape[0],
                'features.26': 100, #1*model._svds['features.26']['Vh'].shape[0],
                'features.28': 100, #1*model._svds['features.28']['Vh'].shape[0],
                # 'classifier.0': 100,
                # 'classifier.3': 100,
                }
        drillers = {}
        for peep_layer in peep_layers:
            drillers[peep_layer] = tGMM(
                    path = drill_path,
                    name = drill_name+'.'+peep_layer,
                    nl_classifier = n_cluster,
                    nl_model = n_classes,
                    n_features = feature_sizes[peep_layer],
                    parser = cv_parsers[peep_layer],
                    device = device
                    )
        
        # fitting classifiers
        with corevecs as cv:
            cv.load_only(
                    loaders = ['train', 'val'],
                    verbose = True
                    ) 
            for drill_key, driller in drillers.items():
                if (drill_path/(driller._clas_path)).exists():
                    print(f'Loading Classifier for {drill_key}') 
                    driller.load()
                else:
                    t0 = time()
                    print(f'Fitting classifier for {drill_key} time = ', time()-t0)
                    driller.fit(corevectors = cv._corevds['train'], verbose=verbose)
            
                    # save classifiers
                    print(f'Saving classifier for {drill_key}')
                    driller.save()

                X = trim_corevectors(cvs=cv._corevds['val'], module = drill_key, cv_dim = 100)
                
                labels = driller._classifier.predict(X)
                sil = silhouette_score(X, labels)
                db  = davies_bouldin_score(X, labels)
                ch  = calinski_harabasz_score(X, labels)
                
                #  Record
                results.append({
                    'layer':            drill_key,
                    'n_cluster':        n_cluster,
                    'silhouette':       sil,
                    'davies_bouldin':   db,
                    'calinski_harabasz':ch
                })

# --- after all loops, save to CSV ---
    df = pd.DataFrame(results)
    df.to_csv('cluster_quality_metrics.csv', index=False)
    print("Saved metrics to cluster_quality_metrics.csv")
                    
            

            