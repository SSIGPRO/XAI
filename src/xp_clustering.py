import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Needed for 3D plotting
import math

# sklearn stuff 
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D
from peepholelib.peepholes.classifiers.parsers import trim_corevectors
from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights
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
    
    drill_path = Path.cwd()/'../data/drillers'
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes'
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
    
    nn = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_features = 4096
    n_classes = len(ds.get_classes()) 
    nn.classifier[-1] = torch.nn.Linear(in_features, n_classes)
    model = ModelWrap(
            model=nn,
            path=model_dir,
            name=model_name,
            device=device)

    target_layers = [
            'classifier.0',
            'classifier.3',
            #'features.7',
            #'features.14',
            #'features.28',
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
            path=svds_path,
            name=svds_name,
            verbose=verbose
            )
    
    for k in model._svds.keys():
        for kk in model._svds[k].keys():
            print('svd shapes: ', k, kk, model._svds[k][kk].shape)

    #--------------------------------
    # CoreVectors 
    #--------------------------------
    #dss = ds._dss
    dss = random_subsampling(ds._dss, 0.05)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    reduction_fns = {
            'classifier.0': partial(svd_Linear,
                                    reduct_m=model._svds['classifier.0']['Vh'], 
                                    device=device),
            'classifier.3': partial(svd_Linear,
                                    reduct_m=model._svds['classifier.3']['Vh'], 
                                    device=device),
        #     'features.28': partial(svd_Conv2D, 
        #                             reduct_m=model._svds['features.28']['Vh'], 
        #                             layer=model._target_layers['features.28'], 
        #                             device=device),
            }
    
    shapes = {
            'classifier.0': 4096,
            'classifier.3': 4096,
            #'features.28': 300,
            }

    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100
    n_cluster = 10
    cv_dim = 10
    parser_cv = trim_corevectors
    peep_layers = ['classifier.0', 'classifier.3']
    
    cls_kwargs = {}#{'batch_size':256} 

    drillers = {}
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    for peep_layer in peep_layers:
        parser_kwargs = {'module': peep_layer, 'cv_dim':cv_dim}

        drillers[peep_layer] = tGMM(
                                path = drill_path,
                                name = drill_name+'.'+peep_layer,
                                nl_classifier = n_cluster,
                                nl_model = n_classes,
                                n_features = cv_dim,
                                parser = parser_cv,
                                parser_kwargs = parser_kwargs,
                                device = device,
                                batch_size = 512
                                )
    
    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 
        
        for drill_key, driller in drillers.items():
            print(drill_path/(driller._suffix+'.empp.pt'))
            if (drill_path/(driller._suffix+'.empp.pt')).exists():
                print(f'Loading Classifier for {drill_key}') 
                driller.load()
                
                #### VISUALIZATION OF THE CORRESPONDENCE BETWEEN RESPONSABILITY AND VALIDATION SET FREQUENCY ####
                phi_prob = driller._classifier.model_._buffers['component_probs'] # shape: (n_components,)
                clusters = range(len(phi_prob))
                fig, axs = plt.subplots(figsize=(12, 4))
                width = 0.35
                x = np.array(clusters)

                axs.bar(x - width/2, phi_prob, width=width, label='Mixing Coefficients (φ)')
                axs.set_xlabel("Cluster Index")
                axs.set_xticks(clusters)
                axs.grid(True)
                
                cv_dl = cv.get_dataloaders(verbose=verbose)
                scores = []
                
                for data in cv_dl['val']:
                        data = trim_corevectors(cvs=data, layer=drill_key, cv_dim=parser_kwargs['cv_dim'])
                        scores.append(driller._classifier.predict(data))
                        
                scores = torch.concatenate(scores)

                ### loglikelihood ####
                total_log_likelihood = -driller._classifier.score(cv._corevds['val'][drill_key][:,:cv_dim])

                #### AIC & BIC ####

                # since the covariance matrix is diagonal, the number of parameters is given by the number of
                # components times the number of features (for the means) plus the number of components times the
                # number of features (for the diagonal of the covariance matrix) plus the number of components minus 1
                # (for the mixing coefficients)
                
                k = n_cluster * cv_dim + n_cluster * cv_dim + (n_cluster - 1)

                AIC = 2 * k - 2 * total_log_likelihood
                BIC = k * math.log(cv._corevds['train'][drill_key].shape[0]) - 2 * total_log_likelihood

                print(f'AIC: {AIC}, BIC: {BIC}')

                #### Silhouette Score and Davis-bouldin index ####

                silhouette = silhouette_score(cv._corevds['val'][drill_key][:,:cv_dim], scores)
                db_index = davies_bouldin_score(cv._corevds['val'][drill_key][:,:cv_dim], scores)

                print("Silhouette Score:", silhouette)
                print("Davies-Bouldin Index:", db_index)

                frequency = scores.bincount(minlength=n_cluster)/scores.numel()

                # Create bin edges so that each cluster label gets its own bin
                
                axs.bar(x + width/2, frequency, width=width, label='Validation set Frequency')
                
                axs.set_title(f'Cluster Anlaysis layer: {drill_key} cv_dim: {parser_kwargs['cv_dim']}')
                axs.set_ylabel('Frequency')
                axs.set_xticks(x)  # Ensure tick marks correspond to each cluster label
                axs.legend()
                fig.savefig(f'n_classes={n_classes}_{drill_key}_n_cluster={n_cluster}_cv_dim={cv_dim}.png')

            else:
                raise RuntimeError('the file does not exist')
 

        