import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.dim_reduction.random import *

if __name__ == "__main__":

    with corevecs as cv, dataset as ds:
        ds.load_only(
                loaders = loaders,
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )

        # computing the corevectors
        cv.get_coreVectors(
                datasets = ds,
                reducers = reducers,
                save_input = save_input,
                save_output = save_output,
                activations_parser = activation_parser,
                batch_size = bs_base,
                n_threads = n_threads,
                verbose = verbose
                )

        if not (cvs_path/(cvs_name+'.normalization.pt')).exists():
                cv.normalize_corevectors(
                        wrt = f'{dataset_name}-train-{args.model}-{args.version}',
                        to_file = cvs_path/(cvs_name+'.normalization.pt'),
                        #from_file = cvs_path/(cvs_name+'.normalization.pt'),
                        batch_size = bs_base,
                        n_threads = n_threads,
                        verbose=verbose
                        )
