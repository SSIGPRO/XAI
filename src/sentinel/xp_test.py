from functools import partial
import sys
from time import time
from matplotlib import pyplot as plt
import numpy as np

import torch
from cuda_selector import auto_cuda
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'newpenny/XAI/EP').as_posix())
from tqdm import tqdm

#wombats stuff

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

# Our stuff
from peepholelib.datasets.sentinel import Sentinel

from torchmetrics.classification import BinaryROC, BinaryAUROC

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", required=True, type=str, help="Model type to use")
args = parser.parse_args()

emb_size = args.emb_size

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
    parsed_path = f'/srv/newpenny/XAI/generated_data/AE_sentinel/datasets_{emb_size}'
    configs = ['all', 'single', 'RW']
    cis = ['high']#, 'medium', 'low']

    loaders = ['val', 'test']
    verbose = True 

    loaders_c = [f"test-{p}-c-{config}-{ci}"  for config in configs for ci in cis for p in loaders]
    loaders_RW = [f"test-{p}-c-RW-{ci}" for ci in cis for p in loaders]
    loaders_single = [f"test-{p}-c-single-{ci}" for ci in cis for p in loaders]
    
    loaders += loaders_c

    #--------------------------------
    # Dataset
    #--------------------------------

    sentinel = Sentinel(
        path = parsed_path
    )

    column_names = ['RW1_motcurr', 'RW1_therm', 'RW1_cmd_volt', 'RW1_speed', 'RW2_motcurr',
                'RW2_therm', 'RW2_cmd_volt', 'RW2_speed', 'RW3_motcurr', 'RW3_therm',
                'RW3_cmd_volt', 'RW3_speed', 'RW4_motcurr', 'RW4_therm', 'RW4_cmd_volt',
                'RW4_speed']
    
    rw = range(4)
    c = range(16)
    
    corr_names = ['Constant', 'Step', 'Impulse', 'GWN','PSA']
    n_corr = len(corr_names)
    
    with sentinel as s:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )

        loss = torch.nn.MSELoss(reduction='none')        

        scores = {key: loss(dss['data'], dss['output']).mean(dim=(2,3)) for key, dss in s._dss.items()}

        # plt.plot(scores['test'])
        # plt.savefig('score_timeline.png')

        # plt.hist(scores['val'], bins=100, density=True)
        # plt.yscale('log')
        # plt.savefig('pdf.png')
        # plt.close()

        # plt.hist(scores['val-c-all-high'], bins=100, density=True)
        # plt.yscale('log')
        # plt.savefig('pdf-c-all.png')
        # plt.close()

        # print((scores['val-c-all-high']>0.003).sum()/len(scores['val-c-all-high']))
        # quit()

        for key, score in scores.items(): print(f'{key} {score.mean()}')

        print('---------------\n mutliple channel analysis \n-----------------')

        for key in tqdm(loaders_c):
            ci = key.split('-')[-1]
            p = key.split('-')[1]
            config= key.split('-')[3]

            if p == 'val':
                
                fig, axs = plt.subplots(n_corr, 3, figsize=(12, 3 * n_corr), squeeze=False)
                plt.subplots_adjust(wspace=0.28, hspace=0.35)
                i = 0

            else: i = 1
                
            for j, corruption in enumerate(corr_names):
                            
                idx = torch.argwhere(s._dss[key]['corruption']==j).squeeze(-1).tolist()
                score = torch.cat((scores[key][idx], scores[p][idx])).squeeze(dim=1)
                results = torch.cat((torch.ones(len(idx)), torch.zeros(len(scores[p][idx]))))
                
                axs[j,i].hist(scores[p][idx], bins=40, density=True, alpha=0.55, label='Original')
                axs[j,i].hist(scores[key][idx], bins=40, density=True, alpha=0.55, label='Corrupted')
                
                if j == 0: axs[j, i].set_title(p, fontsize=12)

                if j == len(corr_names) - 1: axs[j,i].set_xlabel('score')
                if i ==0: axs[j, i].set_ylabel(corruption, fontsize=11, rotation=0, labelpad=45, va='center')
                axs[j, i].set_yscale('log')
                axs[j, i].set_xscale('log')
                axs[j,i].legend(fontsize=8)

                roc_metric = BinaryROC()   # returns fpr, tpr, thresholds
                fpr, tpr, thresholds = roc_metric(score, results.int())

                auc_metric = BinaryAUROC()
                auc_value = auc_metric(score, results.int()).item()

                # Plot
                axs[j,2].plot(fpr, tpr, lw=2, label=f'{corruption} ({p}) AUC={auc_value:.3f}')
                axs[j,2].plot([0, 1], [0, 1], 'k--', lw=1)
                if j == len(corr_names) - 1: axs[j,2].set_xlabel('False Positive Rate')
                axs[j,2].set_ylabel('True Positive Rate')
                if j == 0: axs[j,2].set_title(f'ROC')
                axs[j,2].legend()
                axs[j,2].grid(True)

            fig.savefig(Path.cwd()/f'temp_plots_{emb_size}/AUC-c-{config}-{ci}.png')
        
        print('---------------\n RW analysis \n-----------------')

        for key in tqdm(loaders_RW):
            ci = key.split('-')[-1]
            p = key.split('-')[1]
            config= key.split('-')[3]

            if p == 'val':
                
                fig, axs = plt.subplots(n_corr*4, 3, figsize=(12, 3 * n_corr*4), squeeze=False)
                plt.subplots_adjust(wspace=0.28, hspace=0.35)

                i = 0
            else: i = 1

            count = 0
                
            for j, corruption in enumerate(corr_names):
                for k in rw:
                            
                    idx = torch.argwhere((s._dss[key]['corruption']==j) & (s._dss[key]['RW']==k)).squeeze(-1).tolist()
                    score = torch.cat((scores[key][idx], scores[p][idx])).squeeze(dim=1)
                    results = torch.cat((torch.ones(len(idx)), torch.zeros(len(scores[p][idx]))))
                    
                    axs[count,i].hist(scores[p][idx], bins=40, density=True, alpha=0.55, label='Original')
                    axs[count,i].hist(scores[key][idx], bins=40, density=True, alpha=0.55, label='Corrupted')
                    
                    if count == 0: axs[j, i].set_title(p, fontsize=12)

                    if count == len(corr_names) - 1: axs[count,i].set_xlabel('score')
                    if i ==0: axs[count,i].set_ylabel(corruption+f'-RW{k}', fontsize=11, rotation=0, labelpad=45, va='center')
                    axs[count,i].set_yscale('log')
                    axs[count,i].legend(fontsize=8)

                    roc_metric = BinaryROC()   # returns fpr, tpr, thresholds
                    fpr, tpr, thresholds = roc_metric(score, results.int())

                    auc_metric = BinaryAUROC()
                    auc_value = auc_metric(score, results.int()).item()

                    # Plot
                    axs[count,2].plot(fpr, tpr, lw=2, label=f'{corruption} ({p}) AUC={auc_value:.3f}')
                    axs[count,2].plot([0, 1], [0, 1], 'k--', lw=1)
                    if count == len(corr_names) - 1: axs[count,2].set_xlabel('False Positive Rate')
                    axs[count,2].set_ylabel('True Positive Rate')
                    if count == 0: axs[count,2].set_title(f'ROC')
                    axs[count,2].legend()
                    axs[count,2].grid(True)

                    count += 1

            fig.savefig(Path.cwd()/f'temp_plots_{emb_size}/AUC-c-RW-{ci}-RW-wise.png')

        print('---------------\n single channel analysis \n-----------------')

        for key in tqdm(loaders_single):
            ci = key.split('-')[-1]
            p = key.split('-')[1]
            config= key.split('-')[3]

            if p == 'val':

                fig, axs = plt.subplots(n_corr*16, 3, figsize=(12, 3 * n_corr*16), squeeze=False)
                plt.subplots_adjust(wspace=0.28, hspace=0.35)

                i = 0
            else: i = 1

            count = 0
                
            for j, corruption in enumerate(corr_names):
                for k in c:
                            
                    idx = torch.argwhere((s._dss[key]['corruption']==j) & (s._dss[key]['channel']==k)).squeeze(-1).tolist()
                    score = torch.cat((scores[key][idx], scores[p][idx])).squeeze(dim=1)
                    results = torch.cat((torch.ones(len(idx)), torch.zeros(len(scores[p][idx]))))
                    
                    axs[count,i].hist(scores[p][idx], bins=40, density=True, alpha=0.55, label='Original')
                    axs[count,i].hist(scores[key][idx], bins=40, density=True, alpha=0.55, label='Corrupted')
                    
                    if count == 0: axs[j, i].set_title(p, fontsize=12)

                    if count == len(corr_names) - 1: axs[count,i].set_xlabel('score')
                    if i ==0: axs[count,i].set_ylabel(corruption+f'-channel{k}', fontsize=11, rotation=0, labelpad=45, va='center')
                    axs[count,i].set_yscale('log')
                    axs[count,i].legend(fontsize=8)

                    roc_metric = BinaryROC()   # returns fpr, tpr, thresholds
                    fpr, tpr, thresholds = roc_metric(score, results.int())

                    auc_metric = BinaryAUROC()
                    auc_value = auc_metric(score, results.int()).item()

                    # Plot
                    axs[count,2].plot(fpr, tpr, lw=2, label=f'{corruption} ({p}) AUC={auc_value:.3f}')
                    axs[count,2].plot([0, 1], [0, 1], 'k--', lw=1)
                    if count == len(corr_names) - 1: axs[count,2].set_xlabel('False Positive Rate')
                    axs[count,2].set_ylabel('True Positive Rate')
                    if count == 0: axs[count,2].set_title(f'ROC')
                    axs[count,2].legend()
                    axs[count,2].grid(True)

                    count += 1

            fig.savefig(Path.cwd()/f'temp_plots_{emb_size}/AUC-c-single-channel-{ci}-single-channel-wise.png')


        