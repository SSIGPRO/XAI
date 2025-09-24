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
from peepholelib.peepholes.peepholes import Peepholes

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

    phs_path = Path(f'/srv/newpenny/XAI/generated_data/AE_sentinel/peepholes_{emb_size}')
    phs_name = 'peepholes'
    configs = ['all']# 'single', 'RW']
    cis = ['high']#, 'medium', 'low']

    loaders = ['test']
    verbose = True 

    loaders_c = [f"{p}-{q}-c-{config}-{ci}"  for config in configs for ci in cis for p in loaders for q in loaders]
    loaders_RW = [f"{p}-{q}-c-RW-{ci}" for ci in cis for p in loaders for q in loaders]
    loaders_single = [f"{p}-{q}-c-single-{ci}" for ci in cis for p in loaders for q in loaders]
    
    loaders += loaders_c

    #--------------------------------
    # Dataset
    #--------------------------------

    sentinel = Sentinel(
        path = parsed_path
    )

    peepholes = Peepholes(
                        path = phs_path,
                        name = phs_name+f'.test.50.50.all.{emb_size}.high',
                        device = device
                        )

    column_names = ['RW1_motcurr', 'RW1_therm', 'RW1_cmd_volt', 'RW1_speed', 'RW2_motcurr',
                'RW2_therm', 'RW2_cmd_volt', 'RW2_speed', 'RW3_motcurr', 'RW3_therm',
                'RW3_cmd_volt', 'RW3_speed', 'RW4_motcurr', 'RW4_therm', 'RW4_cmd_volt',
                'RW4_speed']
    
    rw = range(4)
    c = range(16)
    
    corr_names = ['Constant', 'Step', 'Impulse', 'GWN','PSA']
    n_corr = len(corr_names)
    
    with sentinel as s, peepholes as p:
        s.load_only(
            loaders = loaders,
            verbose = verbose
        )
        p.load_only(
            loaders = loaders_c,
            verbose = verbose
        )

        print(torch.argwhere(s._dss['test-test-c-all-high']['corruption']==2).squeeze().tolist())
        idx = 20000

        data = p._phs['test-test-c-all-high']['encoder.linear']['peepholes'][idx].cpu().unsqueeze(1)

        fig_h = 0.45*len(corr_names) + 0.6  # auto height per row
        fig, ax = plt.subplots(figsize=(1.6, fig_h))  # narrow width → column look

        ax.imshow(data, aspect='auto', interpolation='nearest')
        ax.set_xticks([])  # no x ticks

        # y ticks on the right only
        ypos = np.arange(len(corr_names))
        ax.set_yticks(ypos)
        ax.set_yticklabels(corr_names, fontweight='bold')
        ax.tick_params(axis='y', left=False, right=True, labelleft=False, labelright=True)

        # clean up borders
        for s in ['top', 'left', 'bottom']:
            ax.spines[s].set_visible(False)

        plt.savefig('peephole_heatmap.png', bbox_inches='tight', pad_inches=0.05, dpi=200)
        plt.close(fig)

        # plt.imshow(p._phs['test-test-c-all-high']['encoder.linear']['peepholes'][idx].cpu().unsqueeze(1), aspect='auto')
        # plt.xticks([])
        # ypos = np.arange(len(corr_names))
        # plt.yticks(ypos, labels=corr_names, fontweight='bold')
        
        # plt.tight_layout()

        # plt.savefig('peephole_heatmap.png')
        # plt.close()
        
        # print(s._dss['test-test-c-all-high']['corruption'])
        # print(p._phs['test-test-c-all-high']['encoder.linear']['peepholes'][idx].cpu().unsqueeze(1))
        quit()
        # print(p)

        # seed = 42  # choose your seed (or parametrize it)
        # g = torch.Generator(device='cpu').manual_seed(seed)

        # perm = torch.randperm(len(s._dss['test']) , generator=g)
        
        # val_idx  = perm[:100]
        # test_idx = perm[100: 2*100]
        # print(val_idx)

        # for idx in range(80):
        #     idx *= 100

        #     c = s._dss['val-val-c-single-high']['data'][idx][0]
        #     o = s._dss['val']['data'][115374][0]
            
        #     fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        #     for i, ax in enumerate(axs.flat):
        #         ax.plot(c[i].cpu().numpy(), label='c')
        #         ax.plot(o[i].cpu().numpy(), label='o')
        #         ax.set_title(f"{i}")
        #         ax.legend()
        #         ax.axis('tight')
        #     plt.tight_layout()
        #     plt.savefig(f'c_o{idx}_single.png')

        # for idx in range(20):
        #     idx *= 100

        #     c = s._dss['val-val-c-RW-high']['data'][idx][0]
        #     o = s._dss['val']['data'][115374][0]
            
        #     fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        #     for i, ax in enumerate(axs.flat):
        #         ax.plot(c[i].cpu().numpy(), label='c')
        #         ax.plot(o[i].cpu().numpy(), label='o')
        #         ax.set_title(f"{i}")
        #         ax.legend()
        #         ax.axis('tight')
        #     plt.tight_layout()
        #     plt.savefig(f'c_o{idx}_RW.png')

        # for idx in range(5):
        #     idx *= 10000

        #     c = s._dss['val-val-c-all-high']['data'][idx][0]
            
        #     o = s._dss['val']['data'][115374][0]

            # plt.plot(c[0].cpu().numpy(), label='c', linewidth=3)
            # plt.plot(o[0].cpu().numpy(), '--', label='o', linewidth=3)
            # plt.xticks([])
            # plt.yticks([])
            # plt.tight_layout()
            # plt.savefig(f'c_o{idx}_all_0.png')
            # plt.close()
            
            # fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            # for i, ax in enumerate(axs.flat):
            #     ax.plot(c[i].cpu().numpy(), label='c', linewidth=2)
            #     #ax.plot(o[i].cpu().numpy(), '--', label='o', linewidth=2)
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     # if i == 0:
            #     #     ax.legend()
            #     ax.axis('tight')

            # for col in range(4):
            #     axs[0, col].set_title(f"Telemetry.{col}", fontsize=10)

            # Add row labels
            # for row in range(4):
            #     axs[row, 0].set_ylabel(f"RW{row}", rotation=0, labelpad=25, fontsize=10, va='center')

            # fig.suptitle(f'{corr_names[idx//10000]}')
            # plt.tight_layout()
            # plt.savefig(f'c_o{idx}_all.png')


            # c = s._dss['val-val-c-all-high']['output'][idx][0]
            
            # o = s._dss['val']['output'][115374][0]
            
            # fig, axs = plt.subplots(4, 4, figsize=(8, 8))
            # for i, ax in enumerate(axs.flat):
            #     ax.plot(c[i].cpu().numpy(), label='c')
            #     ax.plot(o[i].cpu().numpy(), '--', label='o')
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     if i == 0:
            #         ax.legend()
            #     ax.axis('tight')

            # for col in range(4):
            #     axs[0, col].set_title(f"Telemetry.{col}", fontsize=10)

            # # Add row labels
            # for row in range(4):
            #     axs[row, 0].set_ylabel(f"RW{row}", rotation=0, labelpad=25, fontsize=10, va='center')

            # fig.suptitle(f'{corr_names[idx//10000]}')
            # plt.tight_layout()
            # plt.savefig(f'c_o{idx}_all_output.png')

        