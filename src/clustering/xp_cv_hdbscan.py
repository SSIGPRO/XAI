import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

import cuml
cuml.accel.install()

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

###### Our stuff
from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.datasets.cifar100 import Cifar100

from peepholelib.datasets.parsedDataset import ParsedDataset 


# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

from peepholelib.utils.viz_empp import *


from cuml.cluster.hdbscan import HDBSCAN, membership_vector, approximate_predict

def compute_empp_hdbscan(hard_labels, soft_membership, y, n_classes):
    """
    hard_labels: tensor[N]  (after noise reassignment)
    soft_membership: tensor[N, C]
    y: tensor[N] class labels
    """

    hard_labels = torch.tensor(hard_labels)
    soft_membership = torch.tensor(soft_membership)
    y = torch.tensor(y)

    n_clusters = soft_membership.shape[1]
    empp = torch.zeros((n_clusters, n_classes))

    # compute P(class = s | c)
    for c in range(n_clusters):
        in_cluster = (hard_labels == c)  # samples assigned to cluster c

        if in_cluster.sum() == 0:
            continue

        # weight contributions by soft-membership score
        weights = soft_membership[in_cluster, c] 
        labels = y[in_cluster]                   

        # accumulate weighted counts
        for s in range(n_classes):
            mask_s = (labels == s)
            empp[c, s] = weights[mask_s].sum()

        # normalize 
        total = empp[c].sum()
        if total > 0:
            empp[c] /= total

    return empp

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

    plots_path = Path.cwd()/'temp_plots/coverage/'

    target_layers = [
        'features.7', 'features.10', 'features.12', 'features.14', 'features.17',
        'features.19', 'features.21', 'features.24', 'features.26', 'features.28',
        'classifier.0', 'classifier.3',
        'classifier.6',
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

            X = cv._corevds['CIFAR100-train'][layer][:]
            X_reduced = X[:, :4]
            y = ds._dss["CIFAR100-train"][:]["label"]  


            hdb = HDBSCAN(
                alpha=1.0,
                min_cluster_size=2,
                min_samples=50,
                cluster_selection_method='leaf',
                prediction_data=True
            )
            with cuml.accel.profile():
                # Fit model on corevectors
                hdb.fit(X=X_reduced.detach().cpu().numpy())
                hdb.generate_prediction_data()

                soft_membership = membership_vector(
                    clusterer=hdb,
                    points_to_predict=X_reduced.detach().cpu().numpy(),
                    batch_size=4096,
                    convert_dtype=False,
                )

                soft_membership = torch.tensor(soft_membership)

                if soft_membership.ndim == 1: # in the case that only one cluster was found
                    soft_membership = soft_membership.unsqueeze(1)

                pred_labels, _ = approximate_predict(
                    clusterer=hdb,
                    points_to_predict=X_reduced.detach().cpu().numpy(),
                    convert_dtype=False,
                )
                pred_labels = torch.tensor(pred_labels)

            # reassign noise to argmax of membership vector
            noise_mask = pred_labels == -1
            if noise_mask.sum() > 0:
                pred_labels[noise_mask] = soft_membership[noise_mask].argmax(dim=1).to(pred_labels.dtype)

            # map labels to do the empirical posterior
            unique_labels = pred_labels.unique()
            mapping = {old.item(): new for new, old in enumerate(unique_labels)}
            pred_labels_remapped = pred_labels.clone()
            for old, new in mapping.items():
                pred_labels_remapped[pred_labels == old] = new

            pred_labels = pred_labels_remapped

            empp = compute_empp_hdbscan(
                hard_labels=pred_labels,
                soft_membership=soft_membership,
                y=y.detach().int().cpu().numpy(),
                n_classes=n_classes
            )

            drillers[layer] = {
                "_empp": empp
            }
            print("emp: ", empp)

            # -----------------------------------------------------
            # some cluster diagnostics
            # -----------------------------------------------------
            unique_labels = pred_labels.unique()
            n_clusters = (unique_labels != -1).sum().item()
            n_noise = (pred_labels == -1).sum().item()

            print(f"\n=== Layer: {layer} ===")
            print(f"Clusters found: {n_clusters}")
            print(f"Noise points: {n_noise} / {len(pred_labels)}")

            # Cluster size distribution
            sizes = [(lbl.item(), (pred_labels == lbl).sum().item())
                    for lbl in unique_labels if lbl.item() != -1]
            sizes_sorted = sorted(sizes, key=lambda x: x[1], reverse=True)

            print("Top 5 cluster sizes:")
            for lbl, sz in sizes_sorted[:5]:
                print(f"   Cluster {lbl}: {sz} points")

    coverage = empp_coverage_scores(
        drillers=drillers,
        threshold=0.8,

    )

