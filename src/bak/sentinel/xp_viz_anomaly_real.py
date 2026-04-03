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
import matplotlib.patches as patches
from math import floor
import pandas as pd

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

def draw_group_box(fig, axs_group, label=None, color='black', pad=0.006):
    # make sure positions are computed
    fig.canvas.draw()
    # union of axes bboxes (in figure coords)
    bboxes = [ax.get_position() for ax in axs_group]
    left   = min(bb.x0 for bb in bboxes) - pad
    right  = max(bb.x1 for bb in bboxes) + pad
    bottom = min(bb.y0 for bb in bboxes) - pad
    top    = max(bb.y1 for bb in bboxes) + pad

    rect = patches.Rectangle(
        (left, bottom), right-left, top-bottom,
        transform=fig.transFigure, fill=False,
        edgecolor=color, linewidth=2, clip_on=False, zorder=10
    )
    fig.add_artist(rect)

    if label is not None:
        fig.text(left - 0.004, (bottom+top)/2, label,
                 rotation=90, va='center', ha='right',
                 color=color, fontsize=20, fontweight='bold')

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    print(f"Using {device} device")

    # use_cuda = torch.cuda.is_available()
    # cuda_index = torch.cuda.device_count() - 1
    # device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    # print(f"Using {device} device")

    #--------------------------------
    # Directories definitions
    #--------------------------------
    parsed_path = f'/srv/newpenny/XAI/generated_data/AE_sentinel/datasets_{emb_size}'

    phs_path = Path(f'/srv/newpenny/XAI/generated_data/AE_sentinel/peepholes_{emb_size}')
    phs_name = 'peepholes'

    configs = ['all', 'single', 'RW']
    cis = ['high', 'medium', 'low']

    loaders = ['test_ori']
    verbose = True 

    #--------------------------------
    # TimeStamps
    #--------------------------------

    ws = 16

    test_file = Path('/srv/newpenny/dataset/TASI/sentinel/sentinel_4s_clean_std/test_data.pkl')
    data_test_std = pd.read_pickle(test_file.as_posix())

    _data = torch.tensor(data_test_std.values, dtype=torch.float32)
    nw = floor(_data.shape[0]/ws) # num windows

    data = _data[:ws*nw]
    data = data.reshape(-1, ws, data.shape[-1]) # 16 is the number of signals
    data = data.permute(0, 2, 1).unsqueeze(dim=1) ## B, 1, nc, nw

    idx = data.isnan().any(dim=(2,3)).logical_not()#used dim instead of axis

    time_stamps = data_test_std.index.to_numpy()
    time_stamps = time_stamps[:ws*nw]
    time_stamps = time_stamps.reshape(-1, ws)
    
    time_stamps = time_stamps[idx.squeeze(dim=1).numpy()]

    #--------------------------------
    # Dataset
    #--------------------------------

    sentinel = Sentinel(
        path = parsed_path
    )

    column_names = ['RW0_motcurr', 'RW0_therm', 'RW0_cmd_volt', 'RW0_speed', 'RW1_motcurr',
                'RW1_therm', 'RW1_cmd_volt', 'RW1_speed', 'RW2_motcurr', 'RW2_therm',
                'RW2_cmd_volt', 'RW2_speed', 'RW3_motcurr', 'RW3_therm', 'RW3_cmd_volt',
                'RW3_speed']
    
    RW_labels = ['RW0', 'RW1', 'RW2', 'RW3']

    peaks = [
            #  24766,
            #  134493,
            #  143991,
             172863,
            #  180000,
            #  240753
             ]
    
    rw = range(4)
    c = range(16)

    wr_pre = 10
    wr_post = 60
    
    corruptions = ['Offset', 'Step', 'Impulse', 'GWN','PSA']
    n_corr = len(corruptions)

    cv_dim = 50
    n_cluster = 50
    ci = 'high'
    fit = 'test'

    tests = {
            # 'single_channel': {
            #     'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
            #     'empp_fit_key': f'{fit}-val-c-single-{ci}', 
            #     'label_key': 'channel',
            #     'n_classes': 16,
            #     'class_names': [f'Ch{i}' for i in range(16)] 
            #     },
            # 'single_corruption': {
            #     'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
            #     'empp_fit_key': f'{fit}-val-c-single-{ci}', 
            #     'label_key': 'corruption',
            #     'n_classes': len(corruptions.keys()),
            #     'class_names': corruptions.keys()
            #     },
            # 'single_RW': {
            #     'loaders': [f'{fit}-val-c-single-{ci}', f'{fit}-test-c-single-{ci}'],
            #     'empp_fit_key': f'{fit}-val-c-single-{ci}', 
            #     'label_key': 'RW',
            #     'n_classes': 4, 
            #     'class_names': [f'RW{i}' for i in range(4)] 
            #     },
            'all': {
                'loaders': [f'{fit}-val-c-all-{ci}', f'{fit}-test-c-all-{ci}', 'test_ori'],
                'empp_fit_key': f'{fit}-val-c-all-{ci}', 
                'label_key': 'corruption',
                'n_classes': len(corruptions),
                'class_names': corruptions
                },
            'RW_corruption': {
                'loaders': [f'{fit}-val-c-RW-{ci}', f'{fit}-test-c-RW-{ci}', 'test_ori'],
                'empp_fit_key': f'{fit}-val-c-RW-{ci}', 
                'label_key': 'corruption',
                'n_classes': len(corruptions),
                'class_names': corruptions 
                },
            'RW_RW': {
                'loaders': [f'{fit}-val-c-RW-{ci}', f'{fit}-test-c-RW-{ci}', 'test_ori'],
                'empp_fit_key': f'{fit}-val-c-RW-{ci}', 
                'label_key': 'RW',
                'n_classes': 4,
                'class_names': [f'RW{i}' for i in range(4)] 
                },
            }
    
    font_size = 50

    for idx in tqdm(peaks):
        start = wr_pre * ws
        end   = (wr_pre + 1) * ws
        window = torch.arange(idx - wr_pre, idx + wr_post).tolist()

        for test_name in tqdm(tests):

            peepholes = Peepholes(
                                path = phs_path,
                                name = phs_name+f'.{fit}.{n_cluster}.{cv_dim}.{test_name}.{emb_size}.{ci}',
                                device = device
                                )
            
            with sentinel as s, peepholes as p:
                s.load_only(
                    loaders = loaders,
                    verbose = verbose
                )

                # plt.figure(figsize=(12, 4))
                # plt.plot(s._dss['test_ori']['data'][172864-wr:172864+wr,0,8,:].reshape(-1))
                # plt.tight_layout()
                # plt.savefig('RW2_motocurr')
                # plt.close()
                # plt.figure(figsize=(12, 4))
                # plt.plot(s._dss['test_ori']['data'][172864-ws:172864+ws,0,9,:].reshape(-1))
                # plt.savefig('RW2_temp')
                # plt.tight_layout()
                # plt.close()

                p.load_only(
                    loaders = loaders,
                    verbose = verbose
                )

                loss = torch.nn.MSELoss(reduction='none')        

                scores = {key: loss(dss['data'], dss['output']).mean(dim=(2,3)) for key, dss in s._dss.items()}

                sample_mask = (s._dss['test_ori']['label'] == 0).all(dim=(1,2))

                data = s._dss['test_ori']['data'][sample_mask][window].squeeze(dim=1).permute(0,2,1)
                data = data.reshape(-1, data.size(2)) 
                
                sc = scores['test_ori'][sample_mask][window]
 
                num_points = data.shape[0]
                thr = 0.003

                sc_expanded = sc.repeat_interleave(ws)

                fig = plt.figure(figsize=(70, 70))
                # 16 signal rows + 3 spacer rows + AE + Peephole
                ratios = [1,1,1,1,  0.3,   1,1,1,1,  0.3,   1,1,1,1,  0.3,   1,1,1,1, 0.3, 2, 0.3, 2]
                gs = fig.add_gridspec(nrows=len(ratios), ncols=1, height_ratios=ratios, hspace=0.25)

                axs = []
                r = 0
                for i in range(16):
                    if i in {4,8,12}:  # skip spacer rows after each group of 4
                        r += 1
                    ax = fig.add_subplot(gs[r, 0])
                    axs.append(ax)
                    r += 1

                ae_row = len(ratios) - 3   # AE score row (after adding the spacer)
                ph_row = len(ratios) - 1   # Peephole row

                ax_score = fig.add_subplot(gs[ae_row, 0])
                axs.append(ax_score)              
                ax_ph = fig.add_subplot(gs[-1, 0])   # Peephole (tallest)
                axs.append(ax_ph)

                for i in range(16):
                    
                    axs[i].plot(data[:,i], linewidth=5)
                    
                    axs[i].axvline(x=start, color='green', linestyle='--', linewidth=1.5)
                    axs[i].axvline(x=end,   color='green', linestyle='--', linewidth=1.5)

                    if i % 4 == 0:
                        axs[i].set_title(f'RW{i//4}', fontsize=font_size, fontweight='bold')

                    axs[i].set_xticks([])
                    axs[i].set_yticks([])

                axs[16].plot(sc_expanded, linewidth=5)
                axs[16].axhline(y=thr, color='red', linestyle='--', linewidth=1.5)
                axs[16].axvline(x=start, color='green', linestyle='--', linewidth=1.5)
                axs[16].axvline(x=end,   color='green', linestyle='--', linewidth=1.5)
                
                axs[16].set_title('AE score', fontsize=font_size, fontweight='bold')
                axs[16].set_xticks([])
                axs[16].tick_params(axis='y', labelsize=font_size)  # set font size
                for label in axs[16].get_yticklabels():      # make labels bold
                    label.set_fontweight('bold')

                fig.subplots_adjust(hspace=0.25)

                mask = sc > thr

                pm = p._phs['test_ori']['encoder.linear']['peepholes'][window].detach().cpu().numpy()
                
                pm[~mask.squeeze(1)] = np.nan

                pm_expanded = np.repeat(pm, ws, axis=0)
                
                axs[17].imshow(pm_expanded.T, aspect='auto', interpolation='none')
                axs[17].axvline(x=start, color='green', linestyle='--', linewidth=1.5)
                axs[17].axvline(x=end,   color='green', linestyle='--', linewidth=1.5)
                
                ypos = np.arange(len(tests[test_name]['class_names']))
                axs[17].set_yticks(ypos, labels=tests[test_name]['class_names'], fontsize=font_size, fontweight='bold')
                axs[17].set_title('Peephole', fontsize=font_size, fontweight='bold')
                axs[17].tick_params(axis='x', labelsize=font_size)  # set font size
                for label in axs[17].get_xticklabels():      # make labels bold
                    label.set_fontweight('bold')
                axs[17].set_xlabel('time index', fontsize=font_size, fontweight='bold')

                for ax in axs:
                    ax.set_xlim(0, len(data[:,0])) 

                plt.savefig(f'anomaly_{idx}_{test_name}.png', bbox_inches='tight', dpi=300)
                plt.close(fig)
                    