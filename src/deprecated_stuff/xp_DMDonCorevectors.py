import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
from sklearn import covariance
from sklearn.metrics import roc_curve, auc

# Attcks
import torchattacks
from peepholelib.adv_atk.attacks_base import ftd
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW

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
from torch.utils.data import DataLoader


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
    name_model = 'vgg16'
    pretrained = True
    dataset = 'CIFAR100' 
    seed = 29
    bs = 512 

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_path = Path.cwd()/'../data' #Path('/srv/newpenny/XAI/Peephole-Analysis')  
    svds_name = 'svds' 
    
    cvs_path = Path.cwd()/'../data/corevectors' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/corevectors') 
    cvs_name = 'corevectors_channel_wise'

    drill_path = Path.cwd()/'../data/drillers' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/drillers_on_class') 
    drill_name = 'classifier'

    phs_path = Path.cwd()/'../data/peepholes' #Path('/srv/newpenny/XAI/Peephole-Analysis/channel_wise/peepholes_on_class') 
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
            channel_wise = False,
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
    # Attacks
    #--------------------------------
    dss = ds._dss
    #dss = random_subsampling(ds._dss, 0.05)
    dss_ = {
           #'train': dss['train'],
           'val': dss['val'],        
           'test': dss['test']
              }
    
    loaders = {}
    for key in dss_.keys():
        loaders[key] = DataLoader(dss_[key], batch_size=bs, shuffle=False) 

    atcks = {
             'myPGD':
                     {'model': model._model,
                      'eps' : 8/255, 
                      'alpha' : 2/255, 
                      'steps' : 10,
                      'device' : device,
                      'path' : '/srv/newpenny/XAI/generated_data/attacks/PGD',
                      'name' : 'PGD',
                      'dl' : loaders,
                      'name_model' : name_model,
                      'verbose' : True,
                      'mode' : 'random',},
             'myBIM': 
                     {'model': model._model,
                      'eps' : 8/255, 
                      'alpha' : 2/255, 
                      'steps' : 10,
                      'device' : device,
                      'path' : '/srv/newpenny/XAI/generated_data/attacks/BIM',
                      'name' : 'BIM',
                      'dl' : loaders,
                      'name_model' : name_model,
                      'verbose' : True,
                      'mode' : 'random',},
             'myCW':{
                      'model': model._model,
                      'device' : device,
                      'path' : '/srv/newpenny/XAI/generated_data/attacks/CW',
                      'name' : 'CW',
                      'dl' : loaders,
                      'name_model' : name_model,
                      'verbose' : True,
                      'nb_classes' : n_classes,
                      'confidence': 0,
                      'c_range': (1e-3, 1e10),
                      'max_steps': 1000,
                      'optimizer_lr': 1e-2,
                      'verbose': True,},
             'myDeepFool':{
                           'model': model._model,
                            'steps' : 50,
                            'overshoot' : 0.02,
                            'device' : device,
                            'path' : '/srv/newpenny/XAI/generated_data/attacks/DeepFool',
                            'name' : 'DeepFool',
                            'dl' : loaders,
                            'name_model' : name_model,
                            'verbose' : True,
                            }
                  }

    atk_dss_dict = {}
    for atk_type, kwargs in atcks.items():
        atk = eval(atk_type)(**kwargs)

        if not atk.atk_path.exists():
            atk.get_ds_attack()
        
        atk_dss_dict[atk_type] = atk._atkds
        
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    dss = ds._dss
    #dss = random_subsampling(ds._dss, 0.05)
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            model = model,
            )
    
    # define a dimensionality reduction function for each layer
    reduction_fns = {
            'features.24': partial(
               svd_Conv2D, 
               reduct_m=model._svds['features.24']['Vh'], 
               layer=model._target_modules['features.24'], 
               device=device
               ),
            'features.26': partial(
               svd_Conv2D, 
               reduct_m=model._svds['features.26']['Vh'], 
               layer=model._target_modules['features.26'], 
               device=device
               ),
            'features.28': partial(
               svd_Conv2D, 
               reduct_m=model._svds['features.28']['Vh'], 
               layer=model._target_modules['features.28'], 
               device=device
               ),
            # 'classifier.0': partial(
            #     svd_Linear,
            #     reduct_m=model._svds['classifier.0']['Vh'], 
            #     device=device
            #     ),
            # 'classifier.3': partial(
            #     svd_Linear,
            #     reduct_m=model._svds['classifier.3']['Vh'], 
            #     device=device
            #     ),    
            }

    with corevecs as cv: 
        # copy dataset to activatons file
        cv.get_activations(
                batch_size = bs,
                datasets = dss,
                verbose = verbose,
                n_threads=num_workers
                )        

        # computing the corevectors
        cv.get_coreVectors(
                batch_size = bs,
                reduction_fns = reduction_fns,
                verbose = verbose,
                n_threads=num_workers
                )

        cv_dl = cv.get_dataloaders(verbose=verbose)
    
        i = 0
        print('\nPrinting some corevecs')
        for data in cv_dl['test']:
            print('\nfeatures.28')
            print(data['features.28'][0])
            i += 1
            if i == 3: break
        
        # cv.normalize_corevectors(
        #         wrt='train',
        #         #from_file=cvs_path/(cvs_name+'.normalization.pt'),
        #         to_file=cvs_path/(cvs_name+'.normalization.pt'),
        #         verbose=verbose,
        #         n_threads=num_workers
        #         )
        
        i = 0
        print('after norm')
        for data in cv_dl['test']:
            print(data['features.24'].shape)
            print(data['features.26'].shape)
            print(data['features.28'].shape)
            
            print(data['features.28'][0])
            i += 1
            if i == 1: break
        
    #--------------------------------
    # Peepholes
    #--------------------------------
    n_classes = 100
    # n_cluster = 40
#     cv_dim = 512
    parser_cv = trim_corevectors
    peep_layers = [
         'features.24',
         'features.26',
         'features.28',
        #  'classifier.0', 
        #  'classifier.3' 
         ]
    
    cls_kwargs = {}#{'batch_size': bs} 

    drillers = {}
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    cv_parsers = {
        'features.24': partial(
                trim_channelwise_corevectors,
                module = 'features.24',
                cv_dim = 196,
                cols = [0] 
                ),
            'features.26': partial(
                trim_channelwise_corevectors,
                module = 'features.26',
                cv_dim = 196,
                cols = [0] 
                ),
            'features.28': partial(
                trim_channelwise_corevectors,
                module = 'features.28',
                cv_dim = 196,
                cols = [0]
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
            'features.24': 196, #1*model._svds['features.24']['Vh'].shape[0],
            'features.26': 196, #1*model._svds['features.26']['Vh'].shape[0],
            'features.28': 196, #1*model._svds['features.28']['Vh'].shape[0],
            # 'classifier.0': 100,
            # 'classifier.3': 100,
            }
    
    # fitting classifiers
    print('fitting classifiers')
    with corevecs as cv:
        cv.load_only(
                loaders = ['train'],
                verbose = True
                ) 
        labels = cv._actds['train']['label'].int()
        _means = {}
        _precision = {}
        for layer, size in feature_sizes.items():
            print(f'layer: {layer}, size: {size}')
            group_lasso = covariance.EmpiricalCovariance(assume_centered=False)
            
            _means[layer] = torch.zeros(n_classes, size, device=device)
             
            list_features = trim_channelwise_corevectors(cvs=cv._corevds['train'], module=layer, cv_dim=size, cols=[0]).to(device) # create a copy of cvs to device
            print(list_features.shape)
            for i in range(n_classes):
                _means[layer][i] = list_features[labels == i].mean(dim=0).to(device)
                list_features[labels == i] -= _means[layer][i]
            
            # find inverse            
            group_lasso.fit(list_features.cpu().numpy())
            _precision[layer] = torch.from_numpy(group_lasso.precision_).float().to(device)
    print('computing scores')
    with corevecs as cv:
        cv.load_only(
                loaders = ['val'],
                verbose = True
                )
        cv_dl = cv.get_dataloaders(verbose=verbose)
        
        gaussian_score = {}
        for layer in peep_layers:
            list_features = []
            for d_ori in cv_dl['val']:
                list_features.append(trim_channelwise_corevectors(cvs=d_ori, module=layer, cv_dim=feature_sizes[layer], cols=[0]).to(device))
            list_features = torch.cat(list_features, dim=0)
            n_samples = list_features.shape[0]
            gaussian_score[layer] = torch.zeros(n_samples, n_classes, device=device)
            for i in range(n_classes):
                zero_f = list_features - _means[layer][i]
                term_gau = -0.5*torch.mm(torch.mm(zero_f, _precision[layer]), zero_f.t()).diag()
                gaussian_score[layer][:,i] = term_gau
    #--------------------------------
    # Attacks analysis
    #--------------------------------   
    print('Attacks analysis')            
    gaussian_score_atk = {}

    for atk_type, atk_dss in atk_dss_dict.items():

        gaussian_score_atk[atk_type] = {}
        
        print('--------------------------------')
        print('atk type: ', atk_type)
        print('--------------------------------')
        cvs_path_ = cvs_path/f'{atk_type.replace('my', "")}'
        phs_path_ = phs_path/f'{atk_type.replace('my', "")}'

        corevecs = CoreVectors(
            path = cvs_path_,
            name = cvs_name,
            model = model,
            )

        with corevecs as cv: 

            # copy dataset to coreVect dataset
            cv.load_only(
                    loaders = ['val'],
                    verbose = True
                    )
            cv_dl = cv.get_dataloaders(verbose=verbose)

            gaussian_score_atk[atk_type] = {}
            
            gaussian_score_atk[atk_type] = {}
            for layer in peep_layers:
                
                list_features = []
                for d_ori in cv_dl['val']:
                    list_features.append(trim_channelwise_corevectors(cvs=d_ori, module=layer, cv_dim=feature_sizes[layer], cols=[0]).to(device))
                list_features = torch.cat(list_features, dim=0)
                
                n_samples = list_features.shape[0]
                gaussian_score_atk[atk_type][layer] = torch.zeros(n_samples, n_classes, device=device)
                for i in range(n_classes):
                    zero_f = list_features - _means[layer][i]
                    term_gau = -0.5*torch.mm(torch.mm(zero_f, _precision[layer]), zero_f.t()).diag()
                    gaussian_score_atk[atk_type][layer][:,i] = term_gau

                s_ori,_ = gaussian_score[layer].max(dim=1)
                s_atk,_ = gaussian_score_atk[atk_type][layer].max(dim=1)
                print('s_ori: ', s_ori)
                print('s_atk: ', s_atk)
                
                labels_pos = torch.ones(len(s_ori))
                labels_neg = torch.zeros(len(s_atk))
                all_scores = torch.cat([s_ori, s_atk], dim=0)
                all_labels = torch.cat([labels_pos, labels_neg], dim=0)

                # Convert tensors to numpy arrays (using .detach().cpu().numpy() if needed)
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
                plt.savefig(f'Corevectors_vs_DMD_atk={atk_type}_{layer}.png')
            
        