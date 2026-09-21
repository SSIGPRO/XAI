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

    root = Path("/srv/newpenny/dataset/Multivariate_ts")
    dataset_wraps = {}
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset_wraps[dataset_dir.name] = TSDataWrap(
            path=dataset_dir,
            seed=seed,
        )
    
    # Sampler
    ds_samplers = {}
    # for name in dataset_wraps:
    #     ds_samplers[name] = partial(
    #         random_subsampling,
    #         n_samples=25
    #     )

    root = Path("/srv/newpenny/dataset/Multivariate_ts")

for dataset_dir in sorted(root.iterdir()):
    if not dataset_dir.is_dir():
        continue

    print(f"\n==============================")
    print(f"Parsing {dataset_dir.name}")
    print(f"==============================")

    # One wrapper only
    dataset_wraps = {
        dataset_dir.name: TSDataWrap(
            path=dataset_dir,
            seed=seed,
        )
    }

    # Give each dataset its own parsed-data folder
    dataset = ParsedDataset(
        path=ds_path / dataset_dir.name
    )

    with dataset as ds:
        ds.parse_dataset(
            dataset_wraps=dataset_wraps,
            ds_samplers=None,
            keys_to_copy=["timeseries", "label"],
            batch_size=batch_size,
            n_threads=1,
            verbose=True,
        )

print("Finished parsing all UEA datasets.")