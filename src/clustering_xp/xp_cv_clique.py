import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.append(Path.home()/'repos/XAI/src/clustering_xp/Clique')

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

###### Our stuff
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.cifar100 import Cifar100
from peepholelib.utils.viz_tsne import plot_tsne
from peepholelib.datasets.parsedDataset import ParsedDataset 


# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

import numpy as np
import re
from collections import Counter
from itertools import combinations

from Clique.Clique import  run_clique, normalize_features, evaluate_clustering_performance, save_to_file, plot_clusters


def compute_empp_clique(hard_labels, y, n_classes):
    """
    Computes EM–PP for CLIQUE clustering.
    
    hard_labels: numpy array or tensor of shape [N], with noise = -1
    y:           true labels, shape [N]
    n_classes:   number of true classes
    """
    hard_labels = torch.tensor(hard_labels, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)

    # ignore noise (-1)
    valid = hard_labels >= 0
    valid_labels = hard_labels[valid]
    valid_y = y[valid]

    if len(valid_labels) == 0:
        raise ValueError("CLIQUE produced no valid clusters (all points marked noise).")

    # number of clusters = highest label + 1
    n_clusters = int(valid_labels.max().item() + 1)

    empp = torch.zeros((n_clusters, n_classes), dtype=torch.float32)

    for c in range(n_clusters):
        in_cluster = (valid_labels == c)

        if in_cluster.sum() == 0:
            continue

        labels = valid_y[in_cluster]

        # count class frequencies
        for s in range(n_classes):
            empp[c, s] = (labels == s).sum().item()

        # normalize to form probability distribution
        total = empp[c].sum()
        if total > 0:
            empp[c] /= total

    return empp

def get_cluster_count(filepath):
    single_counts = Counter()
    pair_counts = Counter()

    filepath = "/home/claranunesbarrancos/repos/XAI/src/clustering_xp/clique_clusters/clusters_xsi_10_tau_01_layer_classifier.6.csv"

    with open(filepath, "r") as f:
        for line in f:
            match = re.search(r"Dimensions:\s*\{([^}]*)\}", line)
            if match:
                dims = match.group(1)
                
                dims = [int(d.strip()) for d in dims.split(",") if d.strip().isdigit()]

                # 1D clusters
                if len(dims) == 1:
                    single_counts[dims[0]] += 1

                # 2D clusters (usually never goes beyond this)
                elif len(dims) == 2:
                    pair = tuple(sorted(dims))
                    pair_counts[pair] += 1

    print("\n=== Single-dimension counts ===")
    print(single_counts)
    if single_counts:
        most_common_single = single_counts.most_common(1)[0]
        print(f"Most common single dimension: {most_common_single[0]} (count={most_common_single[1]})")

    print("\n=== 2D pair counts ===")
    print(pair_counts)
    if pair_counts:
        most_common_pair = pair_counts.most_common(1)[0]
        print(f"Most common pair: {most_common_pair[0]} (count={most_common_pair[1]})")

    quit()

if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
    torch.cuda.empty_cache()

    ds_path = Path.cwd()/'../data/datasets'

    model_dir = '/srv/newpenny/XAI/models'
    model_name = 'LM_model=vgg16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau.pth'

    cifar_path = '/srv/newpenny/dataset/CIFAR100'

    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'

    plots_path = Path.cwd()/'temp_plots'

    target_layers = [
         'features.7', #'features.10', 'features.12', 'features.14', 'features.17',
        # 'features.19', 'features.21', 'features.24', 'features.26', 'features.28',
        # 'classifier.0', 'classifier.3',
        #'classifier.6',
    ]

    loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']
    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 


    verbose = True

    #--------------------------------
    # Dataset and CoreVectors 
    #--------------------------------
    
    dataset = ParsedDataset(
            path = ds_path,
            )

    corevecs = CoreVectors(
        path=cvs_path,
        name=cvs_name,
    )

    drillers = {}

    with corevecs as cv, dataset as ds:
        ds.load_only(
            loaders = loaders,
            verbose = verbose
            )

        cv.load_only(
            loaders=loaders,
            verbose=verbose
        )

        for layer in target_layers:

            X = cv._corevds['CIFAR100-train'][layer]
            X_reduced = X[:, :10]
                    
            X_np = X_reduced.cpu().numpy()

            y = ds._dss["CIFAR100-train"][:]["label"]  
            y_np = y.cpu().numpy()

            plot_tsne(X_np = X_np, 
                save_path = plots_path,
                file_name = "classifier6_tsne")
            quit()

            # -----------------------------
            # CLIQUE 
            # -----------------------------
            xsi = 15  # number of grid intervals per dimension
            tau = 0.01  # is dense if number_of_points_in_cell > tau
            clusters = run_clique(
                data=X_np,
                xsi=xsi, 
                tau=tau, 
            )
            print(f"\nFound {len(clusters)} clusters\n")
            save_to_file(clusters, Path("../clique_clusters")/f'clusters_xsi_{xsi}_tau_{tau}_layer_{layer}.csv')

            print("\n=== Evaluation for layer:", layer, "===")
            evaluate_clustering_performance(clusters, y_np)

            print("ploting cluster for layer ", layer)
            title = ("DS: " + "clique_cluster.csv" + " - Params: Tau=" + str(tau) + " Xsi=" + str(xsi))
            plot_clusters(data=X_np, clusters=clusters, title=title, xsi=xsi,
                          save_path=plots_path, filename=f'clique_clusters_xsi_{xsi}_tau_{tau}_layer_{layer}.png')

            # empp = compute_empp_clique(
            #     hard_labels=pred_labels,
            #     y=y_np,
            #     n_classes=n_classes
            # )

            # drillers[layer] = {
            #     "_empp": empp
            # }

            # # just to see whatsup
            # unique, counts = torch.unique(pred_labels, return_counts=True)
            # for u, c in zip(unique.tolist(), counts.tolist()):
            #     print(f"Cluster {u}: {c} points")
