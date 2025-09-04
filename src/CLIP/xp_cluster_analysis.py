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
import re
import pandas as pd

from sklearn.neighbors import KernelDensity

from scipy.signal import find_peaks

# Our stuff
from torch.utils.data import DataLoader

from peepholelib.coreVectors.coreVectors import CoreVectors

from peepholelib.peepholes.classifiers.tkmeans import KMeans as tKMeans 
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 

from peepholelib.wordNet_utils.wordnet import *

# torch stuff
import torch
from nltk.corpus import wordnet as wn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("model", choices=["vgg", "vit"], help="Model type to use")
parser.add_argument("--layer", required=True, help="Layer name to use")
parser.add_argument("--cv_dim", type=int, required=True, help="corevector dimensionality")
parser.add_argument("--n_cluster", type=int, required=True, help="Clusters number")
parser.add_argument("--concept", required=True, help="concept to find")
parser.add_argument("--prefix", required=True, help="prefilter prefix to choos e the synset")
args = parser.parse_args()

model_type = args.model
_layer = args.layer
cv_dim = args.cv_dim
n_cluster = args.n_cluster
_concept = args.concept
prefilter_prefix = args.prefix

# Load config depending on model_type
if model_type == "vgg":
    from config.config_vgg import *
elif model_type == "vit":
    from config.config_vit import *
else:
    raise RuntimeError(
        "Select a configuration: vgg or vit"
    )

def cluster_group_cohesion_with_precision(means, precision_diags, group_indices, all_cluster_indices):
    """
    Calcola il rapporto coesione/separazione usando precision matrix diagonali
    
    Args:
        means: array (n_clusters, n_features) - centroidi dei cluster
        precision_diags: array (n_clusters, n_features) - diagonali delle precision matrix
        group_indices: lista degli indici dei cluster nel gruppo
        all_cluster_indices: lista di tutti gli indici cluster
    
    Returns:
        cohesion_ratio: rapporto separazione_esterna / coesione_interna
    """
    
    external_indices = [i for i in all_cluster_indices if i not in group_indices]
    
    # 1. Distanze interne al gruppo
    internal_distances = []
    for i in group_indices:
        for j in group_indices:
            if i < j:  # evita duplicati
                mahal_dist = mahalanobis_diag(
                    means[i], means[j], 
                    precision_diags[i], precision_diags[j]
                )
                internal_distances.append(mahal_dist)
    
    # 2. Distanze esterne
    external_distances = []
    for i in group_indices:
        for j in external_indices:
            mahal_dist = mahalanobis_diag(
                means[i], means[j],
                precision_diags[i], precision_diags[j]
            )
            external_distances.append(mahal_dist)
    
    # 3. Calcola rapporto
    mean_internal = np.mean(internal_distances) if internal_distances else 0
    mean_external = np.mean(external_distances) if external_distances else 0
    
    cohesion_ratio = mean_external / mean_internal if mean_internal > 0 else np.inf
    
    return cohesion_ratio, mean_internal, mean_external

def mahalanobis_diag(mu1, mu2, prec1, prec2):
    """
    Distanza di Mahalanobis con precision matrix diagonali
    """
    # Precision pooled (media delle due precision)
    pooled_prec = (prec1 + prec2) / 2
    
    # Differenza tra medie
    diff = mu1 - mu2
    
    # Distanza di Mahalanobis: sqrt((mu1-mu2)^T * P * (mu1-mu2))
    # Con P diagonale: sqrt(sum((mu1-mu2)^2 * P_diag))
    mahal_squared = np.sum((diff**2) * pooled_prec)
    
    return np.sqrt(mahal_squared)

def cluster_basis_analysis(means, base_vectors, cluster_indices, k=5):
    """
    Analisi completa della disposizione dei cluster rispetto alla base
    """
    cluster_means = means[cluster_indices]
    base_normalized = base_vectors / np.linalg.norm(base_vectors, axis=1, keepdims=True)
    
    # 1. Proiezioni sui vettori di base
    projections = cluster_means @ base_normalized.T
    
    # 2. Magnitudine delle proiezioni per ogni cluster
    projection_magnitudes = np.linalg.norm(projections, axis=1)
    
    # 3. Direzione principale per ogni cluster
    dominant_directions = np.argmax(np.abs(projections), axis=1)
    
    # 4. Dispersione lungo ogni direzione
    direction_spreads = np.std(projections, axis=0)
    
    # 5. Correlazione tra proiezioni su diverse direzioni
    direction_correlations = np.corrcoef(projections.T)

    # 6. Top k direzioni per ogni cluster
    abs_projections = np.abs(projections)
    
    # Trova gli indici delle top k direzioni per ogni cluster (ordine decrescente)
    top_directions = np.argsort(abs_projections, axis=1)[:, -k:][:, ::-1]
    
    results = {
        'projections': projections,
        'magnitudes': projection_magnitudes,
        'dominant_directions': dominant_directions,
        'direction_spreads': direction_spreads,
        'direction_correlations': direction_correlations,
        'top_directions': top_directions
    }
    
    return results

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
    
    for drill_key, driller in drillers.items():
            print(f'Loading Classifier for {drill_key}') 
            driller.load()
    
    concept_path = drillers[_layer]._clas_path/f'concept={_concept}'
    concept_path.mkdir(parents=True, exist_ok=True)

    path = concept_path/'clusters_KDE_sorted.txt'
    first = path.read_text(encoding="utf-8").splitlines()[0]
    m = re.search(r":\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", first)
    thr = float(m.group(1)) if m else None
    
    df = pd.read_csv(path, sep="\t", skiprows=[0])
 
    cluster_ids = df["cluster_id"].astype(int).tolist()
    all_clusters = list(range(1000))

    means = drillers[_layer]._classifier.model_._buffers['means']
    # precision_diags = drillers[_layer]._classifier.model_._buffers['precisions_cholesky'] 

    # cohesion_ratio, mean_internal, mean_external = cluster_group_cohesion_with_precision(means.detach().cpu().numpy(), 
    #                                                                                      precision_diags.detach().cpu().numpy(), 
    #                                                                                      cluster_ids, 
    #                                                                                      all_clusters)
    
    # print(f"Rapporto coesione: {cohesion_ratio:.3f}")
    # print(f"Distanza interna media: {mean_internal:.3f}")
    # print(f"Distanza esterna media: {mean_external:.3f}")

    # n_group = len(cluster_ids) 
    # n_total_clusters = 1000 

    # random_group = np.random.choice(n_total_clusters, size=n_group, replace=False)

    # cohesion_ratio, mean_internal, mean_external = cluster_group_cohesion_with_precision(means.detach().cpu().numpy(), 
    #                                                                                      precision_diags.detach().cpu().numpy(), 
    #                                                                                      random_group, 
    #                                                                                      all_clusters)
    
    # print(f"Rapporto coesione: {cohesion_ratio:.3f}")
    # print(f"Distanza interna media: {mean_internal:.3f}")
    # print(f"Distanza esterna media: {mean_external:.3f}")

    base_vectors = torch.eye(cv_dim)
    
    analysis = cluster_basis_analysis(means.detach().cpu().numpy(), base_vectors.detach().cpu().numpy(), cluster_ids)

    all_dirs = analysis['top_directions'].flatten()
    print(all_dirs)
    unique, counts = np.unique(all_dirs, return_counts=True)

    # Plot
    plt.figure(figsize=(15, 6))
    plt.hist(all_dirs, bins = cv_dim)
    #plt.bar(unique, counts, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Direzione')
    plt.ylabel('Frequenza')
    plt.title('Tutte le Direzioni Top-5 con Peso Uguale')
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig(f'prova_{_concept}.png')