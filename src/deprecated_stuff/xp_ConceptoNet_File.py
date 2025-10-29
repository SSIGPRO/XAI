import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
from math import floor
from functools import partial

# Our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.datasets.custom_dataset import CustomDataset
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.parsers import from_dataset
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.utils.fine_tune import fine_tune 
from peepholelib.peepholes.peepholes import Peepholes 
from peepholelib.ConceptoNet.ConceptoNet import ParametrizableCNN
from peepholelib.ConceptoNet.Imbalance import calculate_class_weights, create_weighted_sampler
from peepholelib.utils.test_binary import comprehensive_test_analysis
from peepholelib.datasets.transforms import vgg16_cifar100 as ds_transform 

# torch stuff
import torch
from torchvision.models import vgg16, VGG16_Weights, vit_b_16
from cuda_selector import auto_cuda
from torch.nn.functional import softmax as sm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load one configuration file here
# if sys.argv[1] == 'vgg':
#     from peepholes.config_cifar100_vgg16 import *
# elif sys.argv[1] == 'vit':
#     from peepholes.config_cifar100_ViT import *
# else:
#     raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

def pred_fn(output):
     return (torch.sigmoid(output) > 0.5).float()

if __name__ == "__main__":
    seed = 29

    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    ds_path = '/srv/newpenny/dataset/CIFAR100'
    dataset = 'CIFAR100' 
    cvs_path = Path('/srv/newpenny/XAI/generated_data/corevectors/CIFAR100_vgg16')
    cvs_name = 'corevectors'

    phs_path = Path('/srv/newpenny/XAI/generated_data/peepholes_post_tune/CIFAR100_vgg16')
    phs_name = 'peepholes'  
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds = Cifar(
            data_path = ds_path,
            dataset=dataset
            )
    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    peepholes = Peepholes(
            path = phs_path,
            name = f'{phs_name}',
            device = device
            )
    
    loaders = ['train', 'test', 'val']
    num_epochs = 1000
    bs = 128
    dataset = 'CIFAR10' 
    seed = 29
    n_threads = 32

    tune_dir = Path.cwd()/'../../data/ConceptoNet_only_weightedLoss'
    tune_name = 'checkpoints'
    

    verbose = True

    target_layers = [
        'features.7',
        'features.10',
        'features.12',
        'features.14',
        'features.17',
        'features.19',
        'features.21',
        'features.24',
        'features.26',
        'features.28',
        'classifier.0',
        'classifier.3',
        'classifier.6',
        ]

    model = ParametrizableCNN(
                        input_height=len(ds._classes),
                        input_width=len(target_layers),
                        num_channels=1,
                        conv_channels=[8, 16],
                        kernel_sizes=[3, 3],
                        fc_hidden_size=128,
                        dropout_rate=0
                        )
    print(f"Model architecture:\n{model}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = ModelWrap(
            model = model,
            device = device
            )
    
    with corevecs as cv, peepholes as ph:

        ph.load_only(
                loaders = loaders,
                verbose = True
                )
        cv.load_only(loaders = loaders,
                     verbose = True
                     )
        
        output = {ds_key: torch.nn.functional.softmax(cv._dss[ds_key]['output']).max(dim=1).values for ds_key in loaders}
        
        cpss = ph.get_conceptograms(loaders=loaders, target_modules=target_layers, verbose=verbose)
        last = {ds_key: torch.nn.functional.softmax(cv._dss[ds_key]['output']).unsqueeze(dim=1) for ds_key in loaders}
        
        cpss = {ds_key: torch.cat((cpss[ds_key], last[ds_key]), dim=1) for ds_key in loaders}
          
        mean = cpss['train'].mean(dim=0, keepdim=True)
        std = cpss['train'].std(dim=0, keepdim=True)
        
        cpss = {ds_key: ((cpss[ds_key]-mean)/std).transpose(1,2).unsqueeze(dim=1) for ds_key in loaders}

        label = {ds_key : cv._dss[ds_key]['result'] for ds_key in loaders}   

        data = {ds_key: TensorDataset(cpss[ds_key], label[ds_key]) for ds_key in loaders}

        # cpss_dss = CustomDataset()
        # cpss_dss.load_data(dataset=data, classes=[0,1])
        state_dict = torch.load(tune_dir/'best_model/best_model.pth', map_location="cpu")

        state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

        for key, value in state_dict.items():
              print(key, value.shape, '\n')

        model._model.load_state_dict(state_dict)
        test_loader = DataLoader(data['test'], batch_size=bs, shuffle=False, num_workers=n_threads)
        val_loader = DataLoader(data['val'], batch_size=bs, shuffle=False, num_workers=n_threads)
        
        val_results = comprehensive_test_analysis(model, val_loader, device, threshold=0.5, path=tune_dir/'best_model')
        test_results = comprehensive_test_analysis(model, test_loader, device, threshold=0.5, path=tune_dir/'best_model')

        dict_val = {'model_conf': output['val'],
                    'Concepto_conf': val_results['scores'],
                    'label': cv._dss['val']['result'].int()}

        df = pd.DataFrame.from_dict(dict_val).to_csv('val.csv')

        dict_test = {'model_conf': output['test'],
                    'Concepto_conf': test_results['scores'],
                    'label': cv._dss['test']['result'].int()}

        df = pd.DataFrame.from_dict(dict_test).to_csv('test.csv')
        print(df)

        quit()

        class_weights = calculate_class_weights(label['train'].int(), device=device)
        
        loss_fn = torch.nn.BCEWithLogitsLoss
        loss_kwargs = {'pos_weight': class_weights[1]/class_weights[0]}

        optimizer = torch.optim.Adam
        optim_kwargs = {'weight_decay': 1e-4}

        scheduler = optim.lr_scheduler.ReduceLROnPlateau
        sched_kwargs = {'mode': 'min', 'factor': 0.1, 'patience': 5}

        #weighted_sampler = create_weighted_sampler(label['train'].int())

        train_dl_kwargs = {'batch_size': bs, 'collate_fn': partial(from_dataset), 'shuffle': True, 'num_workers': n_threads}
        val_dl_kwargs = {'batch_size': bs, 'collate_fn': partial(from_dataset), 'shuffle': False, 'num_workers': n_threads}

        fine_tune(
            path = tune_dir,
            name = tune_name,
            model = model,
            dataset = cpss_dss,
            pred_fn = pred_fn, 
            loss_fn = loss_fn,
            loss_kwargs = loss_kwargs,
            optimizer = optimizer,
            optim_kwargs = optim_kwargs,
            scheduler = scheduler,
            sched_kwargs = sched_kwargs,
            lr = 1e-5,
            patience = 20,
            iterations = 'full',
            batch_size = bs,
            max_epochs = 100,
            save_every = 10,
            n_threads = 1,
            devices = [device], #[i for i in range(1, 4)], 
            verbose = verbose
            )
        
        model._model.load_state_dict(torch.load(tune_dir/'best_model/best_model.pth'))
        test_loader = DataLoader(data['test'], batch_size=bs, shuffle=False, num_workers=n_threads)
        test_results = comprehensive_test_analysis(model, test_loader, device, threshold=0.5, path=tune_dir/'best_model')
        
        # Access results
        print(f"\nFinal Results Summary:")
        print(f"Test Accuracy: {test_results['accuracy']:.4f}")
        print(f"AUC-ROC: {test_results['auc_roc']:.4f}")
        print(f"Average Precision: {test_results['avg_precision']:.4f}")
        
        scores = torch.tensor(test_results['scores'])
        drop_max = 100
        ns = cpss['test'].shape[0]

        fig, axs = plt.subplots(2, 1, sharex='none', sharey='none', figsize=(4, 4))
        confs = sm(cv._dss['test']['output'], dim=-1).max(dim=-1).values
        results = cv._dss['test']['result']

        s_oks = scores[results == True]
        s_kos = scores[results == False]

        m_oks = confs[results == True]
        m_kos = confs[results == False]
        
        # compute AUC for model 
        # m_auc = AUC().update(confs, results.int()).compute().item()
        
        df = pd.DataFrame({
        'Value': torch.hstack((s_oks, s_kos, m_oks, m_kos)),
        'Score': \
                ['conceptoNet'+'(ml): OK' for i in range(len(s_oks))] + \
                ['conceptoNet'+'(ml): KO' for i in range(len(s_kos))] + \
                ['Model: OK' for i in range(len(m_oks))] + \
                ['Model: KO' for i in range(len(m_kos))]

        })
        colors = ['xkcd:cobalt', 'xkcd:cobalt', 'xkcd:bluish green', 'xkcd:bluish green']

        # effective plotting
        ax = axs[0]
        p = sb.kdeplot(
                data = df,
                ax = ax,
                x = 'Value',
                hue = 'Score',
                common_norm = False,
                palette = colors,
                clip = [0., 1.],
                alpha = 0.8,
                legend = False, #loader_n == 0
                )

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
                handles, labels,
                loc='upper left',
                bbox_to_anchor=(-0.3, 1.0),
                borderaxespad=0
        )

        lines = ['--', '-', '--', '-']
        # set up linestyles
        for ls, line in zip(lines, p.lines):
                line.set_linestyle(ls)
        
        # set legend linestyle
        
        handles = p.legend_.legend_handles[::-1]
        for ls, h in zip(lines, handles):
                h.set_ls(ls)


        ax.set_xlabel('Score')
        ax.set_ylabel('%')
        ax.title.set_text(f'Test set\nConceptoNet')
        ax.grid(True)

        # plot dropping-out accuracy plot
        _, s_idx = scores.sort() # scores is the protoscore
        _, m_idx = confs.sort() # confs is the max of the softmax
        s_acc = torch.zeros(drop_max+1)
        m_acc = torch.zeros(drop_max+1)
        for drop_perc in range(drop_max+1):
                n_drop = floor((drop_perc/100)*ns)
                s_acc[drop_perc] = 100*(results[s_idx[n_drop:]]).sum()/(ns-n_drop)
                m_acc[drop_perc] = 100*(results[m_idx[n_drop:]]).sum()/(ns-n_drop)
        
        colors = ['xkcd:cobalt', 'xkcd:bluish green']
        ax = axs[1]
        df = pd.DataFrame({
        'Values': torch.hstack((s_acc, m_acc)),
        'Score': \
                ['ConceptoNet' for i in range(drop_max+1)] + \
                ['Model confidece' for i in range(drop_max+1)]
        })
        
        sb.lineplot(
                data = df,
                ax = ax,
                x = torch.linspace(0, drop_max, drop_max+1).repeat(2),
                y = 'Values',
                hue = 'Score',
                palette = colors,
                alpha = 0.8,
                )
        ax.set_xlabel('dropped')
        ax.set_ylabel('Accuracy (%)')
        ax.grid(True)

        plt.savefig(tune_dir/'best_model'/'ConceptoNetVsModel.png', dpi=300, bbox_inches='tight')
        plt.close()
             

        
