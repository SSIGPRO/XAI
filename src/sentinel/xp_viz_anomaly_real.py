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


def visualize_anomaly_telemetry(score_win, data_win, peep_win, lab_win, run_idx=0):
    """
    Visualize anomaly detection results with score, telemetry channels, and peepholes.
    
    Parameters:
    -----------
    score_win : torch.Tensor [R, W]
        Anomaly scores for each window
    data_win : torch.Tensor [R, W, 16, 16]
        Telemetry data (window × time × channels)
    peep_win : torch.Tensor [R, W, ...]
        Peephole values
    lab_win : torch.Tensor [R, W]
        Boolean mask indicating anomalies
    run_idx : int
        Which run to visualize (default: 0)
    """
    
    # Convert to numpy if needed
    if hasattr(score_win, 'cpu'):
        score_win = score_win.cpu().numpy()
        data_win = data_win.cpu().numpy()
        peep_win = peep_win.cpu().numpy()
        lab_win = lab_win.cpu().numpy()
    
    # Extract data for the selected run
    scores = score_win[run_idx]  # [W]
    data = data_win[run_idx]      # [W, 16, 16]
    peeps = peep_win[run_idx]     # [W, ...]
    labels = lab_win[run_idx]     # [W]
    
    W = len(scores)
    num_channels = data.shape[2]  # Should be 16
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 4, hspace=0.4, wspace=0.3)
    
    # ===== 1. Score plot with anomaly highlighting =====
    ax_score = fig.add_subplot(gs[0, :])
    x_indices = np.arange(W)
    
    # Plot score line
    ax_score.plot(x_indices, scores, 'b-', linewidth=2, label='Anomaly Score')
    
    # Highlight anomaly regions in red
    anomaly_regions = []
    in_anomaly = False
    start_idx = 0
    
    for i, is_anom in enumerate(labels):
        if is_anom and not in_anomaly:
            start_idx = i
            in_anomaly = True
        elif not is_anom and in_anomaly:
            anomaly_regions.append((start_idx, i-1))
            in_anomaly = False
    if in_anomaly:
        anomaly_regions.append((start_idx, len(labels)-1))
    
    # Fill anomaly regions
    for start, end in anomaly_regions:
        ax_score.axvspan(start, end, alpha=0.3, color='red', label='Anomaly' if start == anomaly_regions[0][0] else '')
        ax_score.scatter(x_indices[start:end+1], scores[start:end+1], c='red', s=30, zorder=5)
    
    ax_score.set_xlabel('Window Index', fontsize=12)
    ax_score.set_ylabel('Score', fontsize=12)
    ax_score.set_title(f'Anomaly Score Timeline (Run {run_idx})', fontsize=14, fontweight='bold')
    ax_score.legend(loc='upper right')
    ax_score.grid(True, alpha=0.3)
    
    # ===== 2. Telemetry channels visualization =====
    # Reshape data from [W, 16, 16] to [W, 16, 16] where first 16 is time, second is channels
    # data[w, t, ch] = value at window w, time step t, channel ch
    
    # Create 4x4 grid for 16 channels
    for ch in range(num_channels):
        row = 1 + ch // 4
        col = ch % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Extract channel data across time for middle window
        middle_window = W // 2
        channel_timeseries = data[middle_window, :, ch]  # [16] time steps
        
        time_steps = np.arange(len(channel_timeseries))
        ax.plot(time_steps, channel_timeseries, 'g-', linewidth=2, marker='o', markersize=4)
        
        # Highlight if this window is anomalous
        if labels[middle_window]:
            ax.set_facecolor('#ffcccc')
        
        ax.set_title(f'Ch {ch}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Time', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # ===== 3. Peephole visualization =====
    ax_peep = fig.add_subplot(gs[3, :])
    
    # Assuming peeps has shape [W, num_features] or [W]
    if len(peeps.shape) == 1:
        # Single peephole value per window
        ax_peep.plot(x_indices, peeps, 'purple', linewidth=2, marker='o', markersize=4)
        ax_peep.set_ylabel('Peephole Value', fontsize=12)
    else:
        # Multiple peephole features - show as heatmap
        im = ax_peep.imshow(peeps.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax_peep.set_ylabel('Peephole Feature', fontsize=12)
        plt.colorbar(im, ax=ax_peep, label='Value')
    
    # Highlight anomaly regions in peephole plot
    for start, end in anomaly_regions:
        ax_peep.axvspan(start, end, alpha=0.2, color='red')
    
    ax_peep.set_xlabel('Window Index', fontsize=12)
    ax_peep.set_title('Peephole Values', fontsize=14, fontweight='bold')
    ax_peep.grid(True, alpha=0.3)
    
    plt.suptitle(f'Anomaly Detection Analysis - Run {run_idx}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    return fig

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
    parsed_path = '/srv/newpenny/XAI/generated_data/AE_sentinel/datasets'

    phs_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/peepholes')
    phs_name = 'peepholes'

    configs = ['all', 'single', 'RW']
    cis = ['high', 'medium', 'low']

    loaders = ['test_ori']
    verbose = True 

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
    
    corruptions = ['Constant', 'Step', 'Impulse', 'GWN','PSA']
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
    

    for test_name in tests:

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

            p.load_only(
                loaders = loaders,
                verbose = verbose
            )

            loss = torch.nn.MSELoss(reduction='none')        

            scores = {key: loss(dss['data'], dss['output']).mean(dim=(2,3)) for key, dss in s._dss.items()}
            peep = p._phs['test_ori']['encoder.linear']['peepholes']
            mask = (s._dss['test_ori']['label'] == 0).all(dim=(1, 2))
            peep[mask] = 0

            pos_mask = (s._dss['test_ori']['label'] == 1).any(dim=(1,2))      # [N] bool
            idx = torch.where(pos_mask)[0]              
            if idx.numel() == 0:
                print("No positives found.")
            else:
                idx = idx.sort().values

                diffs  = idx[1:] - idx[:-1]
                breaks = torch.nonzero(diffs > 1, as_tuple=False).squeeze(-1)

                run_starts = torch.cat([idx.new_zeros(1), breaks + 1])
                run_ends   = torch.cat([breaks, idx.new_tensor([idx.numel()-1])])

                # choose middle element of each run as the center
                center_pos = (run_starts + run_ends) // 2
                centers = idx[center_pos]               

                # ---- build window grid around centers ----
                half = 50                                
                W = 2*half + 1
                N = scores['test_ori'].numel()

                offsets = torch.arange(-half, half+1, device=centers.device) 
                grid = (centers[:, None] + offsets[None, :]).clamp_(0, N-1)  
                print(grid.shape) 

                # ---- gather windows ----
                score_win = scores['test_ori'][grid]            
                data_win  = s._dss['test_ori']['data'][grid]              
                peep_win  = p._phs['test_ori']['encoder.linear']['peepholes'][grid]               
                lab_win   = pos_mask[grid]               

                # ---- visualize per window ----
                for i in range(grid.size(0)):
                    fig, axs = plt.subplots(18, 1, figsize=(50,50))
                    is_anomaly = lab_win[i]
                    print(~is_anomaly)

                    for j in range(16):
                        
                        axs[j].plot(data_win[i,0,...].reshape(-1, data_win.shape[-1])[j])
                        axs[j].set_ylabel(f'C {j}')
                    
                    axs[16].plot(score_win[i])
                    axs[16].set_ylabel('Score')
                    peep_to_show = peep_win[i].clone()  # Make a copy
                    peep_to_show[~is_anomaly,:] = 0
                    axs[17].imshow(peep_to_show.T, aspect='auto', cmap='viridis')
                    axs[17].set_ylabel('Peepholes')
                    axs[17].set_yticks([tests[test_name]['class_names']])
                    # fig.suptitle(f'Window {i} - {"ANOMALY" if is_anomaly else "NORMAL"}', 
                    #                 color='red' if is_anomaly else 'blue', fontsize=14, fontweight='bold')
    
                    plt.tight_layout()
                    fig.savefig(f'window_{i}_{test_name}.png')
                    