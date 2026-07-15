import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, (Path.home() / "repos/peepholelib").as_posix())

from peepholelib.datasets.UEAdataset import TSDataWrap
from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.functional.samplers import random_subsampling

if __name__ == "__main__":

    # Paths
    uea_path = "/srv/newpenny/dataset/Multivariate_ts"
    ds_path = Path.cwd() / "../data/datasets"

    seed = 29
    batch_size = 256
    verbose = True

    # Only UEA_Dataset
    dataset_wraps = {
        'TSDataWrap': TSDataWrap(
            path=uea_path,
            seed=seed,
        ),
    }

    # Sampler
    ds_samplers = {
        'TSDataWrap': partial(random_subsampling, perc=0.5)
    }

    # Parse dataset
    dataset = ParsedDataset(path=ds_path)

    with dataset as ds:
        ds.parse_dataset(
            dataset_wraps=dataset_wraps,
            ds_samplers=ds_samplers,
            keys_to_copy=["image", "label"],
            batch_size=batch_size,
            n_threads=1,
            verbose=verbose,
        )

    print("Finished parsing UEA datasets.")