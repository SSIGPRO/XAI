import os
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# torch stuff
from matplotlib import pyplot as plt
import torch
import torchvision
import pickle
import torch.nn.functional as F

# our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.transforms import mobilenet_v2 as ds_transform 
from peepholelib.peepholes.parsers import trim_corevectors
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.conceptograms import  *
#from peepholelib.utils.mappings import coarse_to_fine_cifar100 as coarse_to_fine
from peepholelib.utils.viz_empp import *
from peepholelib.utils.get_samples import *
from peepholelib.utils.analyze import *


# python stuff
import pandas as pd
from pathlib import Path as Path
from functools import partial
import numpy as np



def compute_average_confidence(cvs, indices, split='test', pred_fn=None):
    """
    Computes the average confidence for a list of sample indices.

    Args:
        cvs: Corevectors object containing predictions and outputs.
        indices (List[int]): List of sample indices.
        split (str): Dataset split ('train', 'val', or 'test').
        pred_fn (callable): Function to convert model output to probabilities (default: softmax).

    Returns:
        float: Average confidence (in %).
    """

    if pred_fn is None:
        pred_fn = partial(F.softmax, dim=0)

    total_conf = 0.0
    for idx in indices:
        _d = cvs._dss[split][idx]
        output = pred_fn(_d['output'])
        if not torch.is_tensor(output):
            output = torch.tensor(output)

        conf = output.max().item() * 100
        total_conf += conf

    if len(indices) == 0:
        return 0.0  # Avoid division by zero

    return total_conf / len(indices)

def discard_curve(**kwargs):
    """
    Computes accuracy on retained samples after progressively discarding worst ones,
    based on combined scores across selected layers.

    Args:
        corevectors: Corevector
        portion (str): Data split to use, e.g., 'val'.
        step (float): Increment step for discard fraction.
        max_discard (float): Maximum fraction of data to discard.

    Returns:
        dict: {discard_fraction: accuracy}
    """
    cvs = kwargs.get('corevectors')
    ph = kwargs.get('peepholes')
    portion = kwargs.get('portion', 'test')
    max_discard = kwargs.get('max_discard', 0.5)
    step = kwargs.get('step', 0.05)
    threshold = kwargs.get('threshold', 0.8)
    top_k = kwargs.get('top_k', 10)

    # Use coverage to select top-k layers
    coverage = empp_coverage_scores(drillers=drillers, threshold=threshold, plot=False)
    avg_scores = {layer: np.mean(v) for layer, v in coverage.items()}
    sorted_layers = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    selected_layers = [layer for layer, _ in sorted_layers[:top_k]]
    print("selected_layers:", selected_layers)
    # Fetch per-sample scores from conceptogram
    scores_dict = conceptogram_protoclass_score(
        peepholes=ph,
        corevectors=cvs,
        target_modules=selected_layers,
        loaders=['train', 'val', 'test'],
        plot=False
    )
    scores = scores_dict['score'][portion]  # shape: [num_samples]
    scores = scores.numpy() if hasattr(scores, 'numpy') else scores

    # Sort indices by worst scores (lowest = worst)
    sorted_indices = np.argsort(scores)
    n_samples = len(sorted_indices)

    _dss = cvs._dss[portion]
    all_labels = []
    all_preds = []
    for idx in sorted_indices:
        _d = _dss[idx]
        all_labels.append(int(_d['label']))
        all_preds.append(int(_d['pred']))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    results = {}
    discard_fracs = np.arange(0.0, max_discard + step, step)

    for frac in discard_fracs:
        num_discard = int(frac * n_samples)
        kept_labels = all_labels[num_discard:]
        kept_preds = all_preds[num_discard:]
        acc = np.mean(kept_labels == kept_preds)
        results[round(frac, 3)] = acc

    return results

def discard_curve_rand(**kwargs):
    cvs = kwargs.get('corevectors')
    ph = kwargs.get('peepholes')
    portion = kwargs.get('portion', 'test')
    max_discard = kwargs.get('max_discard', 0.5)
    step = kwargs.get('step', 0.05)
    top_k = kwargs.get('top_k', 10)
    all_layers = kwargs.get('all_layers')

    # Select 10 random layers
    selected_layers = np.random.choice(
        all_layers,
        size=top_k,
        replace=False
    ).tolist()

    print(f"Selected random layers: {selected_layers}")

    # Fetch per-sample scores from conceptogram
    scores_dict = conceptogram_protoclass_score(
        peepholes=ph,
        corevectors=cvs,
        target_modules=selected_layers,
        loaders=['train', 'val', 'test'],
        plot=False
    )
    scores = scores_dict['score'][portion]  # shape: [num_samples]
    scores = scores.numpy() if hasattr(scores, 'numpy') else scores

    # Sort indices by worst scores (lowest = worst)
    sorted_indices = np.argsort(scores)
    n_samples = len(sorted_indices)

    _dss = cvs._dss[portion]
    all_labels = []
    all_preds = []
    for idx in sorted_indices:
        _d = _dss[idx]
        all_labels.append(int(_d['label']))
        all_preds.append(int(_d['pred']))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    results = {}
    discard_fracs = np.arange(0.0, max_discard + step, step)

    for frac in discard_fracs:
        num_discard = int(frac * n_samples)
        kept_labels = all_labels[num_discard:]
        kept_preds = all_preds[num_discard:]
        acc = np.mean(kept_labels == kept_preds)
        results[round(frac, 3)] = acc

    return results


def plot_discard_curves(coverage_dict_top10, coverage_dict_rand, title="Accuracy vs. Discarded Samples", save_path=None):
    """
    Plots discard curves for concept-based and random discards and optionally saves the plot.

    Args:
        coverage_dict_top10 (dict): {discard_fraction: accuracy} from concept-based discard.
        coverage_dict_rand (dict): {discard_fraction: accuracy} from random discard baseline.
        title (str): Title of the plot.
        save_path (str or None): If provided, saves the plot to this file path.
    """
    # Sort x-values
    x_top10 = sorted(coverage_dict_top10.keys())
    y_top10 = [coverage_dict_top10[x] for x in x_top10]

    x_rand = sorted(coverage_dict_rand.keys())
    y_rand = [coverage_dict_rand[x] for x in x_rand]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_top10, y_top10, label='Concept-based discard (top-10)', marker='o', linewidth=2)
    plt.plot(x_rand, y_rand, label='Random discard', marker='x', linestyle='--')

    plt.xlabel('Fraction of Samples Discarded')
    plt.ylabel('Accuracy on Retained Samples')
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    cuda_index = 0
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    bs = 512 
    seed = 29

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'CN_model=mobilenet_v2_dataset=CIFAR100_optim=Adam_scheduler=RoP_lr=0.001_factor=0.1_patience=5.pth'

    cvs_path = Path("/srv/newpenny/XAI/CN/data/corevectors")
    cvs_name = 'corevectors'

    drill_path = Path("/srv/newpenny/XAI/CN/data100cvdim300/drillers")
    drill_name = 'classifier'

    phs_path = Path("/srv/newpenny/XAI/CN/data100cvdim300/peepholes")
    phs_name = 'peepholes'

    target_layers = [ 
        'features.1.conv.0.0', 'features.1.conv.1',
        'features.2.conv.0.0','features.2.conv.1.0', 'features.2.conv.2',
        'features.3.conv.0.0', 'features.3.conv.1.0', 'features.3.conv.2',
        'features.4.conv.0.0','features.4.conv.1.0', 'features.4.conv.2',
        'features.5.conv.0.0', 'features.5.conv.1.0','features.5.conv.2', 
        'features.6.conv.0.0', 'features.6.conv.1.0', 'features.6.conv.2', #B3
        'features.7.conv.0.0', 'features.7.conv.1.0', 'features.7.conv.2', #B3
        'features.8.conv.0.0', 'features.8.conv.1.0', 'features.8.conv.2', #B4
        'features.9.conv.0.0', 'features.9.conv.1.0', 'features.9.conv.2', #B4
        'features.10.conv.0.0', 'features.10.conv.1.0', 'features.10.conv.2', #B5
        'features.11.conv.0.0', 'features.11.conv.1.0','features.11.conv.2', #B5
        'features.12.conv.0.0', 'features.12.conv.1.0', 'features.12.conv.2', #B5
        'features.13.conv.0.0', 'features.13.conv.1.0', 'features.13.conv.2', #B5
        'features.14.conv.0.0', 'features.14.conv.1.0','features.14.conv.2', #B6
        'features.15.conv.0.0', 'features.15.conv.1.0', 'features.15.conv.2', #B6
        'features.16.conv.0.0', 'features.16.conv.1.0','features.16.conv.2', #B6
        'features.17.conv.0.0', 'features.17.conv.1.0', 'features.17.conv.2', #B7
        'features.18.0', 
        'classifier.1'
    ]
    
    cv_dim = 300
    n_cluster = 100  

    #--------------------------------
    # Dataset
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/CIFAR100'
    dataset = 'CIFAR100' 

    ds = Cifar(
        data_path = ds_path,
        dataset=dataset
        )
    
    ds.load_data(
            transform = ds_transform,
            seed = seed,
            )
    
    #--------------------------------
    # Model 
    #--------------------------------

    nn = torchvision.models.mobilenet_v2()
    n_classes = len(ds.get_classes()) 
    model = ModelWrap(
            model = nn,
            device = device
            )
    
    model.update_output(
            output_layer = 'classifier.1', 
            to_n_classes = n_classes,
            overwrite = True 
            )

    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = False
            )
    
    model.set_target_modules(target_modules=target_layers, verbose=False)
    #--------------------------------
    # Core Vectors
    #--------------------------------
    corevecs = CoreVectors(
        path = cvs_path,
        name = cvs_name,
        model = model,
        )
    #--------------------------------
    # Peepholes
    #--------------------------------
    
    feature_sizes = {
            'features.1.conv.0.0': cv_dim,
            'features.1.conv.1': cv_dim,
            'features.2.conv.0.0': cv_dim,
            'features.2.conv.1.0': cv_dim,
            'features.2.conv.2': cv_dim,    
            'features.3.conv.0.0': cv_dim,
            'features.3.conv.1.0': cv_dim,
            'features.3.conv.2': cv_dim,
            'features.4.conv.0.0': cv_dim,
            'features.4.conv.1.0': cv_dim,
            'features.4.conv.2': cv_dim,
            'features.5.conv.0.0': cv_dim,
            'features.5.conv.1.0': cv_dim,
            'features.5.conv.2': cv_dim,
            'features.6.conv.0.0': cv_dim,
            'features.6.conv.1.0': cv_dim,
            'features.6.conv.2': cv_dim,
            'features.7.conv.0.0': cv_dim,
            'features.7.conv.1.0': cv_dim,
            'features.7.conv.2': cv_dim,
            'features.8.conv.0.0': cv_dim,
            'features.8.conv.1.0': cv_dim,
            'features.8.conv.2': cv_dim,
            'features.9.conv.0.0': cv_dim,
            'features.9.conv.1.0': cv_dim,
            'features.9.conv.2': cv_dim,
            'features.10.conv.0.0': cv_dim,
            'features.10.conv.1.0': cv_dim,
            'features.10.conv.2': cv_dim,
            'features.11.conv.0.0': cv_dim,
            'features.11.conv.1.0': cv_dim,
            'features.11.conv.2': cv_dim,
            'features.12.conv.0.0': cv_dim,
            'features.12.conv.1.0': cv_dim,
            'features.12.conv.2': cv_dim,
            'features.13.conv.0.0': cv_dim,          
            'features.13.conv.1.0': cv_dim,
            'features.13.conv.2': cv_dim,
            'features.14.conv.0.0': cv_dim,
            'features.14.conv.1.0': cv_dim,
            'features.14.conv.2': cv_dim,
            'features.15.conv.0.0': cv_dim,
            'features.15.conv.1.0': cv_dim,
            'features.15.conv.2': cv_dim,
            'features.16.conv.0.0': cv_dim,
            'features.16.conv.1.0': cv_dim,
            'features.16.conv.2': cv_dim,
            'features.17.conv.0.0': cv_dim,
            'features.17.conv.1.0': cv_dim,
            'features.17.conv.2': cv_dim,
            'features.18.0': cv_dim,
            'classifier.1': 100,
    }

   

    cv_parsers = {
    'features.1.conv.0.0': partial(trim_corevectors,
                        module = 'features.1.conv.0.0', cv_dim = cv_dim),
    'features.1.conv.1': partial(trim_corevectors,
                        module = 'features.1.conv.1', cv_dim = cv_dim),
    'features.2.conv.0.0': partial(trim_corevectors,
                        module = 'features.2.conv.0.0', cv_dim = cv_dim),   
    'features.2.conv.1.0': partial(trim_corevectors,
                        module = 'features.2.conv.1.0', cv_dim = cv_dim),
    'features.2.conv.2': partial(trim_corevectors,
                        module = 'features.2.conv.2', cv_dim = cv_dim),
    'features.3.conv.0.0': partial(trim_corevectors,
                        module = 'features.3.conv.0.0', cv_dim = cv_dim),
    'features.3.conv.1.0': partial(trim_corevectors,
                        module = 'features.3.conv.1.0', cv_dim = cv_dim),
    'features.3.conv.2': partial(trim_corevectors,
                        module = 'features.3.conv.2', cv_dim = cv_dim),
    'features.4.conv.0.0': partial(trim_corevectors,
                        module = 'features.4.conv.0.0', cv_dim = cv_dim),
    'features.4.conv.1.0': partial(trim_corevectors,
                        module = 'features.4.conv.1.0', cv_dim = cv_dim),
    'features.4.conv.2': partial(trim_corevectors,
                        module = 'features.4.conv.2', cv_dim = cv_dim),
    'features.5.conv.0.0': partial(trim_corevectors,
                        module = 'features.5.conv.0.0', cv_dim = cv_dim),
    'features.5.conv.1.0': partial(trim_corevectors,
                        module = 'features.5.conv.1.0', cv_dim = cv_dim),
    'features.5.conv.2': partial(trim_corevectors,
                        module = 'features.5.conv.2', cv_dim = cv_dim),
    'features.6.conv.0.0': partial(trim_corevectors,
                        module = 'features.6.conv.0.0', cv_dim = cv_dim),
    'features.6.conv.1.0': partial(trim_corevectors,
                        module = 'features.6.conv.1.0', cv_dim = cv_dim),
    'features.6.conv.2': partial(trim_corevectors,
                        module = 'features.6.conv.2', cv_dim = cv_dim),
    'features.7.conv.0.0': partial(trim_corevectors,
                        module = 'features.7.conv.0.0', cv_dim = cv_dim),
    'features.7.conv.1.0': partial(trim_corevectors,
                        module = 'features.7.conv.1.0', cv_dim = cv_dim),
    'features.7.conv.2': partial(trim_corevectors,
                        module = 'features.7.conv.2', cv_dim = cv_dim),
    'features.8.conv.0.0': partial(trim_corevectors,
                        module = 'features.8.conv.0.0', cv_dim = cv_dim),
    'features.8.conv.1.0': partial(trim_corevectors,
                        module = 'features.8.conv.1.0', cv_dim = cv_dim),
    'features.8.conv.2': partial(trim_corevectors,
                        module = 'features.8.conv.2', cv_dim = cv_dim),
    'features.9.conv.0.0': partial(trim_corevectors,
                        module = 'features.9.conv.0.0', cv_dim = cv_dim),
    'features.9.conv.1.0': partial(trim_corevectors,
                        module = 'features.9.conv.1.0', cv_dim = cv_dim),
    'features.9.conv.2': partial(trim_corevectors,
                        module = 'features.9.conv.2', cv_dim = cv_dim),
    'features.10.conv.0.0': partial(trim_corevectors,
                        module = 'features.10.conv.0.0', cv_dim = cv_dim),   
    'features.10.conv.1.0': partial(trim_corevectors,
                        module = 'features.10.conv.1.0', cv_dim = cv_dim),
    'features.10.conv.2': partial(trim_corevectors,
                        module = 'features.10.conv.2', cv_dim = cv_dim),
    'features.11.conv.0.0': partial(trim_corevectors,
                        module = 'features.11.conv.0.0', cv_dim = cv_dim),
    'features.11.conv.1.0': partial(trim_corevectors,
                        module = 'features.11.conv.1.0', cv_dim = cv_dim),
    'features.11.conv.2': partial(trim_corevectors,
                        module = 'features.11.conv.2', cv_dim = cv_dim),
    'features.12.conv.0.0': partial(trim_corevectors,
                        module = 'features.12.conv.0.0', cv_dim = cv_dim),
    'features.12.conv.1.0': partial(trim_corevectors,
                        module = 'features.12.conv.1.0', cv_dim = cv_dim),
    'features.12.conv.2': partial(trim_corevectors,
                        module = 'features.12.conv.2', cv_dim = cv_dim),
    'features.13.conv.0.0': partial(trim_corevectors,
                        module = 'features.13.conv.0.0',cv_dim = cv_dim),
    'features.13.conv.1.0': partial(trim_corevectors,
                        module = 'features.13.conv.1.0', cv_dim = cv_dim),
    'features.13.conv.2': partial(trim_corevectors,
                        module = 'features.13.conv.2', cv_dim = cv_dim),
    'features.14.conv.0.0': partial(trim_corevectors,
                        module = 'features.14.conv.0.0', cv_dim = cv_dim),
    'features.14.conv.1.0': partial(trim_corevectors,
                        module = 'features.14.conv.1.0', cv_dim = cv_dim),
    'features.14.conv.2': partial(trim_corevectors,
                        module = 'features.14.conv.2', cv_dim = cv_dim),
    'features.15.conv.0.0': partial(trim_corevectors,
                        module = 'features.15.conv.0.0', cv_dim = cv_dim),
    'features.15.conv.1.0': partial(trim_corevectors,
                        module = 'features.15.conv.1.0', cv_dim = cv_dim),
    'features.15.conv.2': partial(trim_corevectors,
                        module = 'features.15.conv.2', cv_dim = cv_dim),
    'features.16.conv.0.0': partial(trim_corevectors,
                        module = 'features.16.conv.0.0', cv_dim = cv_dim),
    'features.16.conv.1.0': partial(trim_corevectors,
                        module = 'features.16.conv.1.0', cv_dim = cv_dim),
    'features.16.conv.2': partial(trim_corevectors,
                        module = 'features.16.conv.2', cv_dim = cv_dim),
    'features.17.conv.0.0': partial(trim_corevectors,
                        module = 'features.17.conv.0.0', cv_dim = cv_dim),
    'features.17.conv.1.0': partial(trim_corevectors,
                        module = 'features.17.conv.1.0', cv_dim = cv_dim),
    'features.17.conv.2': partial(trim_corevectors,
                        module = 'features.17.conv.2', cv_dim = cv_dim),
    'features.18.0': partial(trim_corevectors,
                        module = 'features.18.0', cv_dim = cv_dim),
    'classifier.1': partial(trim_corevectors,
                        module = 'classifier.1', cv_dim = 100),
    }
       
    drillers = {}
    for peep_layer in target_layers:

        drillers[peep_layer] = tGMM(
                path = drill_path,
                name = drill_name+'.'+peep_layer,
                nl_classifier = n_cluster,
                nl_model = n_classes,
                n_features = feature_sizes[peep_layer],
                parser = cv_parsers[peep_layer],
                device = device
                )

    
    peepholes = Peepholes(
        path = phs_path,
        name = phs_name,
        device = device
    )

    with corevecs as cv, peepholes as ph:
        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        ph.get_peepholes(
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = 32,
                verbose = False
                )

    #--------------------------------
    # Conceptograms
    #--------------------------------
    
    with corevecs as cv, peepholes as ph:

        cv.load_only(
                loaders = ['train', 'test', 'val'],
                verbose = True
                ) 

        ph.get_peepholes(
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs,
                drillers = drillers,
                n_threads = 32,
                verbose = False
                )
        
        for _, driller in drillers.items():
            driller.load()
            driller.empirical_posterior_heatmap()

        coverage_dict_top10 = discard_curve(threshold=0.8, top_k=10, corevectors=cv, peepholes = ph, portion='test')
        coverage_dict_rand = discard_curve_rand(threshold=0.8, top_k=10, corevectors=cv, peepholes = ph, portion='test', all_layers=target_layers)

        plot_discard_curves(
            coverage_dict_top10,
            coverage_dict_rand,
            title="Accuracy vs. Discarded Samples",
            save_path=Path('/home/claranunesbarrancos/repos/XAI/src/mobilenet/coverage/coverage_discard_curves.png')
        )
        quit()
        dicionario = conceptogram_protoclass_score(
            peepholes = ph,
            corevectors = cv,
            target_layers = target_layers,
            loaders = ['train','test','val'],
            #bins = 50,
            plot = True,
            #score_type = 'entropy',
            verbose = True
        )
        print(target_layers)

        # indices = torch.tensor([76,1631,9677,9775,44,9981])
        # plot_conceptogram(
        #     path = Path('/home/claranunesbarrancos/repos/XAI/src/mobilenet/conceptos/scores'),
        #     name = 'idk',
        #     corevectors = cv,
        #     peepholes = ph,
        #     portion = 'test',
        #     samples = [76,1631,9677,9775,44,9981],
        #     target_modules = target_layers,
        #     classes = coarse_label_names if superclass else ds.get_classes(),
        #     alt_score = dicionario['score']['test'][indices],
        #     ticks = target_layers,
        #     krows = 5,
        #     # label_key = 'superclass',
        #     # pred_fn = superclass_pred_fn(),
        #     ds = ds,
        # )