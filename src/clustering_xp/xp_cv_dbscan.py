import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())


# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

###### Our stuff


# corevecs
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection

from cuml.cluster.hdbscan import HDBSCAN


if __name__ == "__main__":
    
    cvs_path = Path.cwd()/'../data/corevectors'
    cvs_name = 'corevectors'

    plots_path = Path.cwd()/'temp_plots/coverage/'

    target_layers = [
        'features.7', 'features.10', 'features.12', 'features.14', 'features.17',
        'features.19', 'features.21', 'features.24', 'features.26', 'features.28',
        'classifier.0', 'classifier.3', 'classifier.6',
    ]

    loaders = ['CIFAR100-train', 'CIFAR100-val', 'CIFAR100-test']

    n_classes = len(Cifar100.get_classes(meta_path = Path(cifar_path)/'cifar-100-python/meta')) 

    corevecs = CoreVectors(
        path=cvs_path,
        name=cvs_name,
        model=model,
    )

    drillers = {}

    with corevecs as cv:

        cv.load_only(
            loaders=loaders,
            verbose=verbose
        )

        for layer in target_layers:

            X = cv._corevds['CIFAR100-train'][layer][:]
            X_np = X.cpu().numpy()

            y = cv._corevds['CIFAR100-train']['targets'][:]  # adjust if needed
            y_np = y.cpu().numpy()

            hdb = HDBSCAN(
                alpha=1.0,
                min_cluster_size=30,
                min_samples=10,
                cluster_selection_method='eom',
                prediction_data=True
            )

            # Fit model on corevectors
            hdb.fit(X_np)

            hdb.generate_prediction_data()

            # Hard labels for training points
            hard_labels = hdb.labels_.copy()

            soft_membership = membership_vector(
                clusterer=hdb,
                points_to_predict=X_np,
                batch_size=4096,
                convert_dtype=False,
            )
            soft_membership = torch.tensor(soft_membership)  

            pred_labels, _ = approximate_predict(
                clusterer=hdb,
                points_to_predict=X_np,
                convert_dtype=False,
            )
            pred_labels = torch.tensor(pred_labels)

            # just to see whats up
            n_noise = (pred_labels == -1).sum().item()
            print(f"[{layer}] Noise points before reassignment: {n_noise}")

            # reassign noise to argmax of membership vector
            noise_mask = pred_labels == -1
            if noise_mask.sum() > 0:
                pred_labels[noise_mask] = soft_membership[noise_mask].argmax(dim=1)

            after_noise = (pred_labels == -1).sum().item()
            print(f"[{layer}] Noise points after reassignment: {after_noise}")

            drillers[layer] = {
                "labels": pred_labels.numpy(),
                "soft_membership": soft_membership.numpy(),
                "y": y_np,
            }


    # Compute coverage over hard labels OR soft membership
    coverage = empp_coverage_scores(
        drillers=drillers,
        threshold=0.8,
        plot=True,
        save_path=plots_path,
        file_name='coverage_dbscan_eom.png'
    )
