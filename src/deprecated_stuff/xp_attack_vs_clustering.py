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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc

# scipy stuff 
from scipy.stats import entropy

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.viz import viz_singular_values

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

    cvs_path = Path.cwd()/'../data/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/corevectors') # Path.cwd()/'../data/corevectors' 
    cvs_name = 'corevectors'
   
    drill_path = Path.cwd()/'../data/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/drillers_on_superclass') 
    drill_name = 'classifier'

    verbose = True 
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    attack_list = [
                'PGD',
                'CW',
                'DeepFool',
                'BIM'
                ]

    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100
    n_cluster = 150
    peep_layers = [
        #  'features.24',
        #  'features.26',
        #  'features.28',
         'classifier.0', 
         'classifier.3',
         'classifier.6'
         ]

    cv_parsers = {
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
        #         #cols = [0]
        #         ),
            'classifier.0': partial(
                trim_corevectors,
                module = 'classifier.0',
                cv_dim = 100
                ),
            'classifier.3': partial(
                trim_corevectors,
                module = 'classifier.3',
                cv_dim = 100
                ),
            'classifier.6': partial(
                trim_corevectors,
                module = 'classifier.6',
                cv_dim = 100
                ),
            
            }
    feature_sizes = {
            # for channel_wise corevectors, the size is n_channels * cv_dim
        #     'features.24': 100, #1*model._svds['features.24']['Vh'].shape[0],
        #     'features.26': 100, #1*model._svds['features.26']['Vh'].shape[0],
        #     'features.28': 100, #1*model._svds['features.28']['Vh'].shape[0],
            'classifier.0': 100,
            'classifier.3': 100,
            'classifier.6': 100,
            }
#     drillers = {}
#     for peep_layer in peep_layers:
#         drillers[peep_layer] = tGMM(
#                 path = drill_path,
#                 name = drill_name+'.'+peep_layer,
#                 nl_classifier = n_cluster,
#                 nl_model = n_classes,
#                 n_features = feature_sizes[peep_layer],
#                 parser = cv_parsers[peep_layer],
#                 device = device
#                 )
    #--------------------------------
    # Attacks and original dataset on clusters
    #--------------------------------        

    for atk in attack_list:
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        cvs_path_ = cvs_path/f'{atk}'

        corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
        corevecs_atk = CoreVectors(
            path = cvs_path_,
            name = cvs_name,
            )
        with corevecs as cv, corevecs_atk as cv_atk:
                cv.load_only(
                        loaders = ['val'],
                        verbose = True
                        ) 
                cv_atk.load_only(
                        loaders = ['val'],
                        verbose = True
                        ) 
                
                cv_dl = cv.get_dataloaders(verbose=verbose)
                cv_atk_dl = cv_atk.get_dataloaders(verbose=verbose)
                for layer in peep_layers:
                # for drill_key, driller in drillers.items():
                        # diff = cv._corevds['val'][drill_key]-cv_atk._corevds['val'][drill_key]
                        # # If diff is a PyTorch tensor:
                        # mean_diff = diff.mean(dim=0).cpu().numpy()[:300]   # shape [D]
                        # std_diff  = diff.std(dim=0).cpu().numpy()[:300]    # shape [D]

                        # # 2) Prepare x‑axis
                        # dims = np.arange(mean_diff.shape[0])

                        # # 3) Plot
                        # plt.figure(figsize=(10, 4))
                        # # shaded ±1σ region
                        # plt.fill_between(
                        # dims,
                        # mean_diff - std_diff,
                        # mean_diff + std_diff,
                        # color='gray',
                        # alpha=0.3,
                        # label='±1 std dev'
                        # )
                        # # mean line
                        # plt.plot(dims, mean_diff, lw=1.5, label='Mean difference')
                        # plt.xlabel('Dimension index')
                        # plt.ylabel('Original – Attacked')
                        # plt.title('Per-Dimension Mean and Std of Difference Vectors')
                        # plt.legend()
                        # plt.tight_layout()
                        # plt.show()
                        # plt.savefig(f'../data/img/dim_diff_{drill_key}_atk={atk}.png')


                        # # --- inside your loop, after trimming/loading the driller ---
                        # # 1) Accumulate features
                        # features_ori = []
                        # features_atk = []
                        # for d_ori, d_atk in zip(cv_dl['val'], cv_atk_dl['val']):
                        #         # trim_corevectors returns a tensor of shape (batch_size, cv_dim)
                        #         fo = trim_corevectors(cvs=d_ori, module=drill_key, cv_dim=driller.n_features)
                        #         fa = trim_corevectors(cvs=d_atk, module=drill_key, cv_dim=driller.n_features)
                        #         # move to CPU & numpy
                        #         features_ori.append(fo.detach().cpu().numpy())
                        #         features_atk.append(fa.detach().cpu().numpy())
                        features_ori = cv._corevds['val'][layer].detach().cpu().numpy()  # shape (N_ori, cv_dim)
                        features_atk = cv_atk._corevds['val'][layer].detach().cpu().numpy()  # shape (N_atk, cv_dim)

                        # features_ori = np.concatenate(features_ori, axis=0)  # shape (N_ori, cv_dim)
                        # features_atk = np.concatenate(features_atk, axis=0)  # shape (N_atk, cv_dim)
                        print(f'features_ori: {features_ori.shape}, features_atk: {features_atk.shape}')
                        
                        plt.plot(features_ori[1,:100])
                        plt.plot(features_atk[1,:100])
                        plt.title(f'Coreavg for {layer} atk={atk}')
                        plt.grid(True)
                        plt.savefig(f'../data/img/corevector_{layer}_atk={atk}.png')
                        plt.close()
                        # # 2) Combine and label
                        # X = np.vstack([features_ori, features_atk])         # (N_ori+N_atk, cv_dim)
                        # y = np.array([0]*len(features_ori) + [1]*len(features_atk))  
                        # # 0 = original, 1 = attacked

                        # # 3) (Optional) Pre‑PCA 
                        # pca = PCA(n_components=min(50, X.shape[1]), random_state=0)
                        # X_pca = pca.fit_transform(X)

                        # # 4) t‑SNE embedding
                        # tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=0)
                        # X_emb = tsne.fit_transform(X_pca)  # shape (N_samples, 2)

                        # # 5) Plot
                        # plt.figure(figsize=(8,8))
                        # plt.scatter(
                        # X_emb[y==0, 0], X_emb[y==0, 1],
                        # label='Original', alpha=0.6, s=5
                        # )
                        # plt.scatter(
                        # X_emb[y==1, 0], X_emb[y==1, 1],
                        # label='Attacked', alpha=0.6, s=5
                        # )
                        # plt.legend()
                        # plt.title(f't-SNE of core‑vectors at layer {layer} (atk={atk})')
                        # plt.xlabel("t-SNE dim 1")
                        # plt.ylabel("t-SNE dim 2")
                        # plt.grid(True)
                        # plt.savefig(f'../data/img/tsne_{layer}_atk={atk}.png')
                        # plt.close()
                        
                        # print(drill_path/(driller._clas_path))
                        # if (drill_path/(driller._clas_path)).exists():
                        #         print(f'Loading Classifier for {drill_key}') 
                        #         driller.load()
                        #         plt.imshow(driller._empp.detach().cpu().numpy())
                        #         plt.title(f'Empirical Posterior for {drill_key}')
                        #         plt.savefig(f'../data/img/empp_{drill_key}_cv_dim_{feature_sizes[drill_key]}_n_cluster={n_cluster}.png')
                        #         plt.close()
                        #         s_ori = []
                        #         s_atk = []
                                        
                        #         for d_ori, d_atk in zip(cv_dl['val'], cv_atk_dl['val']):
                        #                 d_ori = trim_corevectors(cvs=d_ori, module=drill_key, cv_dim=driller.n_features)
                        #                 d_atk = trim_corevectors(cvs=d_atk, module=drill_key, cv_dim=driller.n_features)
                        #                 s_ori.append(driller._classifier.predict(d_ori))
                        #                 s_atk.append(driller._classifier.predict(d_atk))
                        #         s_ori = torch.cat(s_ori)
                        #         s_atk = torch.cat(s_atk)
                        #         f_ori = s_ori.bincount(minlength=n_cluster)/s_ori.numel()
                        #         f_atk = s_atk.bincount(minlength=n_cluster)/s_atk.numel()

                        #         # Create bin edges so that each cluster label gets its own bin
                        #         fig, axs = plt.subplots(figsize=(12, 4))
                        #         width = 0.35
                        #         x = np.array(range(n_cluster))

                        #         axs.bar(x - width/2, f_ori, width=width, label='Original set Frequency')
                        #         axs.set_title(f'Cluster Anlaysis layer: {drill_key} cv_dim: {feature_sizes[drill_key]}')
                        #         axs.set_ylabel('Frequency')
                        #         axs.set_xlabel("Cluster Index")
                        #         axs.set_xticks(range(n_cluster))
                        #         axs.grid(True)
                                
                        #         axs.bar(x + width/2, f_atk, width=width, label='Attack set Frequency')
                                
                        #         axs.set_title(f'Cluster Anlaysis layer: {drill_key} cv_dim: {feature_sizes[drill_key]}')
                        #         axs.set_ylabel('Frequency')
                        #         axs.set_xticks(x)  # Ensure tick marks correspond to each cluster label
                        #         axs.legend()
                        #         fig.savefig(f'../data/img/atk={atk}_n_classes={n_classes}_{drill_key}_n_cluster={n_cluster}_cv_dim={feature_sizes[drill_key]}.png')
                        #         fig.clf()
    quit()
    #--------------------------------
    # Corevectors attacks entropy or MAX
    #--------------------------------
    
    for atk in attack_list:
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        cvs_path_ = cvs_path/f'{atk}'

        corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
        corevecs_atk = CoreVectors(
            path = cvs_path_,
            name = cvs_name,
            )
        with corevecs as cv, corevecs_atk as cv_atk:
                cv.load_only(
                        loaders = ['val'],
                        verbose = True
                        ) 
                cv_atk.load_only(
                        loaders = ['val'],
                        verbose = True
                        ) 
                cv_dl = cv.get_dataloaders(verbose=verbose)
                cv_atk_dl = cv_atk.get_dataloaders(verbose=verbose)
                for drill_key, driller in drillers.items():
                        
                        print(drill_path/(driller._clas_path))
                        if (drill_path/(driller._clas_path)).exists():
                                print(f'Loading Classifier for {drill_key}') 
                                driller.load()
                                s_ori = []
                                s_atk = []
                                        
                                for d_ori, d_atk in zip(cv_dl['val'], cv_atk_dl['val']):
                                        d_ori = trim_corevectors(cvs=d_ori, module=drill_key, cv_dim=driller.n_features)
                                        d_atk = trim_corevectors(cvs=d_atk, module=drill_key, cv_dim=driller.n_features)
                                        s_ori.append(driller._classifier.score_samples(d_ori).squeeze())
                                        s_atk.append(driller._classifier.score_samples(d_atk).squeeze())
                                s_ori = torch.cat(s_ori)
                                s_atk = torch.cat(s_atk)

                                # Assume s_ori and s_atk are your score tensors
                                labels_pos = torch.ones(len(s_ori))
                                labels_neg = torch.zeros(len(s_atk))

                                # Concatenate scores and labels:
                                all_scores = torch.cat([s_ori, s_atk], dim=0)
                                all_labels = torch.cat([labels_pos, labels_neg], dim=0)

                                # Convert tensors to numpy arrays (using .detach().cpu().numpy() if needed)
                                #all_scores_np = entropy(all_scores.detach().cpu().numpy(), axis=1)
                                all_scores_np = all_scores.detach().cpu().numpy()
                                
                                all_labels_np = all_labels.detach().cpu().numpy()

                                # Compute ROC curve and AUC
                                fpr, tpr, thresholds = roc_curve(all_labels_np, all_scores_np)
                                roc_auc = auc(fpr, tpr)

                                print("AUC:", roc_auc)

                                # Plot ROC curve
                                plt.figure()
                                plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
                                plt.plot([0, 1], [0, 1], 'k--', label='Chance')
                                plt.xlabel('False Positive Rate')
                                plt.ylabel('True Positive Rate')
                                plt.title('Receiver Operating Characteristic')
                                plt.legend(loc="lower right")
                                plt.savefig(f'../data/img/AUC/NLL_attack={atk}_n_classes={n_classes}_{drill_key}_n_cluster={n_cluster}_cv_dim={driller.n_features}.png')

    quit()
    #------------------------------------- 
    # Clustering analysis
    #-------------------------------------   
    for atk in attack_list:
        print('--------------------------------')
        print('atk type: ', atk)
        print('--------------------------------')
        cvs_path_ = cvs_path/f'{atk}'

        corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
        corevecs_atk = CoreVectors(
            path = cvs_path_,
            name = cvs_name,
            )
        with corevecs as cv, corevecs_atk as cv_atk:
                cv.load_only(
                        loaders = ['val'],
                        verbose = True
                        ) 
                cv_atk.load_only(
                        loaders = ['val'],
                        verbose = True
                        ) 
                cv_dl = cv.get_dataloaders(verbose=False)
                cv_atk_dl = cv_atk.get_dataloaders(verbose=False)
                for drill_key, driller in drillers.items():
                        
                        print(drill_path/(driller._clas_path))
                        if (drill_path/(driller._clas_path)).exists():
                                print(f'Loading Classifier for {drill_key}') 
                                driller.load()
                                s_ori = []
                                s_atk = []
                                        
                                for d_ori, d_atk in zip(cv_dl['val'], cv_atk_dl['val']):
                                        d_ori = trim_corevectors(cvs=d_ori, module=drill_key, cv_dim=driller.n_features)
                                        d_atk = trim_corevectors(cvs=d_atk, module=drill_key, cv_dim=driller.n_features)
                                        s_ori.append(driller._classifier.predict(d_ori))
                                        s_atk.append(driller._classifier.predict(d_atk))
                                s_ori = torch.cat(s_ori)
                                s_atk = torch.cat(s_atk)
                                d_ori = s_ori.bincount(minlength=n_cluster)
                                d_atk = s_atk.bincount(minlength=n_cluster)

                                # For attack scores:
                                idx = 3
                                topk_values_atk, topk_indices_atk = d_atk.topk(idx)
                                total_atk = d_atk.sum()
                                fraction_atk = topk_values_atk.sum().float() / total_atk.float()

                                print("\nAttack Dataset:")
                                print("Top 5 bin counts:", topk_values_atk)
                                print("Top 5 bin indices:", topk_indices_atk)
                                print(f"Fraction of samples in top 5 bins: {fraction_atk.item():.2%}")

                                # For original scores:
                                _values_ori = d_ori[topk_indices_atk]
                                total_ori = d_ori.sum()
                                fraction_ori = _values_ori.sum().float() / total_ori.float()

                                print("Original Dataset:")
                                print("Top 5 bin counts:", _values_ori) 
                                print(f"Fraction of samples in top attack 5 bins: {fraction_ori.item():.2%}")

                                
                                with open(f"results_{drill_key}.txt", "a") as f:
                                        
                                        print(f"\n{atk} Dataset:", file=f)
                                        print(f"Top {idx} bin counts:", topk_values_atk.tolist(), file=f)
                                        print(f"Top {idx} bin indices:", topk_indices_atk.tolist(), file=f)
                                        print(f"Fraction of samples in top {idx} bins: {fraction_atk.item():.2%}", file=f)
                                        print("Original Dataset:", file=f)
                                        print(f"Top {idx} bin counts:", _values_ori.tolist(), file=f) 
                                        print(f"Fraction of samples in top attack {idx} bins: {fraction_ori.item():.2%}", file=f)
                                        print("="*40, file=f)  # Separator for different configurations


                                
#     # fitting classifiers
#     with corevecs as cv:
#         cv.load_only(
#                 loaders = ['train', 'test', 'val'],
#                 verbose = True
#                 ) 
        
#         for drill_key, driller in drillers.items():
#             print(drill_path/(driller._suffix+'.empp.pt'))
#             if (drill_path/(driller._suffix+'.empp.pt')).exists():
#                 print(f'Loading Classifier for {drill_key}') 
#                 driller.load()
                
#                 #### VISUALIZATION OF THE CORRESPONDENCE BETWEEN RESPONSABILITY AND VALIDATION SET FREQUENCY ####
#                 phi_prob = driller._classifier.model_._buffers['component_probs'] # shape: (n_components,)
#                 clusters = range(len(phi_prob))
#                 fig, axs = plt.subplots(figsize=(12, 4))
#                 width = 0.35
#                 x = np.array(clusters)

#                 axs.bar(x - width/2, phi_prob, width=width, label='Mixing Coefficients (φ)')
#                 axs.set_xlabel("Cluster Index")
#                 axs.set_xticks(clusters)
#                 axs.grid(True)
                
#                 cv_dl = cv.get_dataloaders(verbose=verbose)
#                 scores = []
                
#                 for data in cv_dl['val']:
#                         data = trim_corevectors(cvs=data, layer=drill_key, cv_dim=parser_kwargs['cv_dim'])
#                         scores.append(driller._classifier.predict(data))
                        
#                 scores = torch.concatenate(scores)

#                 ### loglikelihood ####
#                 total_log_likelihood = -driller._classifier.score(cv._corevds['val'][drill_key][:,:cv_dim])

#                 #### AIC & BIC ####

#                 # since the covariance matrix is diagonal, the number of parameters is given by the number of
#                 # components times the number of features (for the means) plus the number of components times the
#                 # number of features (for the diagonal of the covariance matrix) plus the number of components minus 1
#                 # (for the mixing coefficients)
                
#                 k = n_cluster * cv_dim + n_cluster * cv_dim + (n_cluster - 1)

#                 AIC = 2 * k - 2 * total_log_likelihood
#                 BIC = k * math.log(cv._corevds['train'][drill_key].shape[0]) - 2 * total_log_likelihood

#                 print(f'AIC: {AIC}, BIC: {BIC}')

#                 #### Silhouette Score and Davis-bouldin index ####

#                 silhouette = silhouette_score(cv._corevds['val'][drill_key][:,:cv_dim], scores)
#                 db_index = davies_bouldin_score(cv._corevds['val'][drill_key][:,:cv_dim], scores)

#                 print("Silhouette Score:", silhouette)
#                 print("Davies-Bouldin Index:", db_index)

#                 frequency = scores.bincount(minlength=n_cluster)/scores.numel()

#                 # Create bin edges so that each cluster label gets its own bin
                
#                 axs.bar(x + width/2, frequency, width=width, label='Validation set Frequency')
                
#                 axs.set_title(f'Cluster Anlaysis layer: {drill_key} cv_dim: {parser_kwargs['cv_dim']}')
#                 axs.set_ylabel('Frequency')
#                 axs.set_xticks(x)  # Ensure tick marks correspond to each cluster label
#                 axs.legend()
#                 fig.savefig(f'n_classes={n_classes}_{drill_key}_n_cluster={n_cluster}_cv_dim={cv_dim}.png')

#             else:
#                 raise RuntimeError('the file does not exist')
 

        