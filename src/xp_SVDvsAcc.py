import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
from tqdm import tqdm

# Our stuff
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.CW import myCW

from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 
from peepholelib.models.model_wrap import ModelWrap 

from peepholelib.coreVectors.dimReduction.svds import svd_Linear, svd_Conv2D
from peepholelib.HeadSqueezing import HeadSqueezingDetector 

# torch stuff
import torch
from torch.utils.data import DataLoader
from torchvision.models import vgg16
from cuda_selector import auto_cuda
from sklearn.metrics import roc_curve, auc


if __name__ == "__main__":
#     use_cuda = torch.cuda.is_available()
#     device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
#     print(f"Using {device} device")
    use_cuda = torch.cuda.is_available()
    cuda_index = torch.cuda.device_count() - 2
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    # model parameters
    dataset = 'CIFAR100'
    name_model= 'vgg16' 
    seed = 29
    bs = 128 
    n_threads = 32

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'
    
    svds_path = Path.cwd()/'../data'
    svds_name = 'svds' 
    
    verbose = True 
    
    target_layers = ['classifier.6']

    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset = dataset
            )

    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    nn = vgg16()
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

    model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )
    
    #--------------------------------
    # SVDs 
    #--------------------------------
    t0 = time()
    model.get_svds(
            path = svds_path,
            name = svds_name,
            target_modules = target_layers,
            sample_in = ds._dss['train'][0][0],
            rank = 10,
            channel_wise = True,
            verbose = verbose
            )
    print('time: ', time()-t0)
    
    # print('\n----------- svds:')
    # for k in model._svds.keys():
    #     for kk in model._svds[k].keys():
    #         print('svd shapes: ', k, kk, model._svds[k][kk].shape)
    #     s = model._svds[k]['s']
    #     if len(s.shape) == 1:
    #         plt.figure()
    #         plt.plot(s, '-')
    #         plt.xlabel('Rank')
    #         plt.ylabel('EigenVec')
    #     else:
    #         fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #         _x = torch.linspace(0, s.shape[1]-1, s.shape[1])
    #         for r in range(s.shape[0]):
    #             plt.plot(xs=_x, ys=s[r,:], zs=r, zdir='y')
    #         ax.set_xlabel('Rank')
    #         ax.set_ylabel('Channel')
    #         ax.set_zlabel('EigenVec')
    #     plt.savefig((svds_path/(svds_name+'/'+k+'.png')).as_posix(), dpi=300, bbox_inches='tight')
    #     plt.close()
    
    
    #--------------------------------
    # Attacks 
    #-------------------------------- 
    dss = ds._dss

    #dss = random_subsampling(ds._dss, 0.05)
    dss_ = {#'train': dss['train'],
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
        atk.load_data()

        if not atk.atk_path.exists():
            atk.get_ds_attack()
        
        atk_dss_dict[atk_type] = atk

    acc = torch.zeros(100)
    print(acc.shape)
    model.set_activations(save_input=True, save_output=False)
    svd = model._svds[target_layers[0]]
    
    Vh = svd['Vh'].T
    U = svd['U']
    s = svd['s']
    print(model._model.state_dict().keys())
    weight = model._model.state_dict()['classifier.6.weight']
    bias = model._model.state_dict()['classifier.6.bias']
    W_ = torch.hstack((weight, bias.reshape(-1,1)))
    
    print(weight.shape, bias.shape, W_.shape)

    w = Vh*s@U
    print(w.shape)

    forb_norm = torch.norm(W_-w.T.to(device))
    print(forb_norm)

    S = torch.diag(s)  # Convert singular values to a diagonal matrix

    # Reconstruct the original matrix
    W_reconstructed = U @ S @ Vh.T
    forb_norm = torch.norm(W_-W_reconstructed.to(device))
    print(forb_norm)
    quit()

#     for data in loaders['test']:
        
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         output = model(inputs)
#         pred = torch.argmax(output, dim=1)
#         result = pred==labels
#         acc[len(acc)-1] += result.sum()

#         act_data = model._model._acts['in_activations'][model.target_module[0]].to(device)
#         n_act = act_data.shape[0]
        
#         acts_flat = act_data.flatten(start_dim=1)
#         ones = torch.ones(n_act, 1, device=device)
#         _acts = torch.hstack((acts_flat, ones))

#         for k in range(99):
            
            
#         # self.output_dict['squeezed'] = (self._A.to(device).T@_acts.T).T
    
#     acc /= len(loaders['test'])
    quit()    

    s_ori = torch.cat(ori_list, dim=0)

    for atk, atk_dss in atk_dss_dict.items():
        
        loader = DataLoader(atk_dss._dss['test'], batch_size=bs, collate_fn=lambda x:x, shuffle=False, num_workers=4) 

        atk_list = []

        for data in loader:            
            
            inputs = data['image'].to(device)
            output = detector(image=inputs, device=device)

            atk_list.append(output.detach().cpu())

        s_atk = torch.cat(atk_list, dim=0)

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
        fig, axs = plt.subplots(1,2, figsize=(10,10))
        axs[0].plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        axs[0].plot([0, 1], [0, 1], 'k--', label='Chance')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title('Receiver Operating Characteristic')
        axs[0].legend()
        axs[1].hist(s_ori.numpy(), bins=100, label='ori', alpha= 0.5)
        axs[1].hist(s_atk.numpy(), bins=100, label=f'{atk}', alpha= 0.5)
        axs[1].legend()
        fig.savefig(f'../data/img/AUC/Head_squeezing_attack={atk}_model={name_model}_mode={mode}_k={k}.png')
        
        
    