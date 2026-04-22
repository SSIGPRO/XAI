import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.dim_reduction.class_wise_pca import *

if __name__ == "__main__":

    with corevecs as cv, dataset as ds:
        ds.load_only(
                loaders = loaders,
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        # train split: project onto ground-truth class subspaces (labels)
        # val/test splits: project onto predicted class subspaces (pred)
        cv.get_coreVectors_pred(
                datasets = ds,
                reducers = reducers,
                fit_loaders = [f'{dataset_name}-train-{model_name}'],
                save_input = True,
                save_output = False,
                batch_size = bs_base,
                n_threads = n_threads,
                verbose = verbose
                )

        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
                cv.normalize_corevectors(
                        wrt = f'{dataset_name}-train-{model_name}',
                        to_file = cvs_path/(cvs_name+'.normalization.pt'),
                        batch_size = bs_base,
                        n_threads = n_threads,
                        verbose=verbose
                        )
