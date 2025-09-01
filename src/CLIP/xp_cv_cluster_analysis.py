import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# python stuff
from time import time
from functools import partial
from matplotlib import pyplot as plt
import math 
import numpy as np
import json
from tqdm import tqdm

# Our stuff
from torch.utils.data import DataLoader

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.utils.samplers import random_subsampling 

# torch stuff
import torch
import clip
from nltk.corpus import wordnet as wn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vgg", "vit"], help="Model type to use")
parser.add_argument("--layer", required=True, help="Layer name to use")
parser.add_argument("--cv_dim", type=int, required=True, help="corevector dimensionality")
parser.add_argument("--n_cluster", type=int, required=True, help="Clusters number")
args = parser.parse_args()

model_type = args.model
_layer = args.layer
cv_dim = args.cv_dim
n_cluster = args.n_cluster

# Load config depending on model_type
if model_type == "vgg":
    from config.config_vgg import *
elif model_type == "vit":
    from config.config_vit import *
else:
    raise RuntimeError(
        "Select a configuration: vgg or vit"
    )

if __name__ == "__main__":

    #--------------------------------
    # Directories definitions
    #--------------------------------
    ds_path = '/srv/newpenny/dataset/ImageNet_torchvision'

    # model parameters
    bs = 512 
    n_threads = 1
    
    cvs_path = Path.cwd()/f'../../data/{model_name}/corevectors'
    cvs_name = 'corevectors'

    drill_path = Path.cwd()/f'../../data/{model_name}/drillers'
    drill_name = 'classifier'
    
    verbose = True 
    
    #--------------------------------
    # Text Encoding
    #-------------------------------- 

    with open(Path.cwd()/"../../data/vgg/imagenet_class_index.json") as f:
        class_idx = json.load(f)

    idx2label = {int(k): v[1] for k, v in class_idx.items()}
    synset_ids = [class_idx[str(i)][0] for i in range(1000)]
    short_labels = [class_idx[str(i)][1] for i in range(1000)]
        
    #--------------------------------
    # CoreVectors 
    #--------------------------------
    
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
    
    drillers = {_layer: tGMM(
                            path = drill_path,
                            name = drill_name+'.'+_layer,
                            nl_classifier = n_cluster,
                            nl_model = n_classes,
                            n_features = feature_sizes[_layer],
                            parser = cv_parsers[_layer],
                            device = device
                            )}

    # fitting classifiers
    with corevecs as cv:
        cv.load_only(
                loaders = ['train', 'val'],
                verbose = True
                ) 

        for drill_key, driller in drillers.items():
            print(f'Loading Classifier for {drill_key}') 
            driller.load()

        n_samples = len(cv._corevds['train'][_layer])                   

        probs = torch.empty(n_samples, n_cluster, dtype=torch.float32)

        cv_dl = DataLoader(cv._corevds['train'][_layer][...,:cv_dim], batch_size=bs, num_workers = n_threads)
        
        start = 0
        
        for data in tqdm(cv_dl):
            bs = data.shape[0]
            probs[start:start+bs] = drillers[_layer]._classifier.predict_proba(data)
            start += bs

        conf, clusters = torch.max(probs, dim=1)
        counts = torch.bincount(clusters, weights=conf, minlength=n_cluster)

        density = counts*torch.sum(torch.log(driller._classifier.model_._buffers['precisions_cholesky']),dim=1)

        corr = torch.corrcoef(torch.stack([counts, density]))[0, 1]

        print(f"Correlation: {corr:.4f}")
    
        counts_np = counts.cpu().numpy()
        density_np = density.cpu().numpy()

        plt.figure(figsize=(10, 8))
        plt.scatter(counts_np, density_np, alpha=0.6, s=20)
        plt.xlabel('Counts')
        plt.ylabel('Density')
        plt.title(f'Correlation between Counts and Density\nCorrelation coefficient: {corr.item():.4f}')
        plt.grid(True, alpha=0.3)

        z = np.polyfit(counts_np, density_np, 1)
        p = np.poly1d(z)
        plt.plot(counts_np, p(counts_np), "r--", alpha=0.8, linewidth=2, label=f'Trend line (y={z[0]:.3f}x+{z[1]:.3f})')
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig(drillers[_layer]._clas_path/f'correlation_plot.png', dpi=200, bbox_inches='tight')
                

                    
            
