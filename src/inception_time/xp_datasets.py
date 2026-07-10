import sys
from pathlib import Path

sys.path.insert(0, (Path.home() / "repos/peepholelib").as_posix())

# Python
from functools import partial
import torch
from cuda_selector import auto_cuda

###### Our stuff ######

# Model
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.inceptiontime import InceptionTime

# Dataset
from peepholelib.datasets.UEA_dataset import TSDataWrap
from peepholelib.datasets.parsedDataset import ParsedDataset

# Inference
from peepholelib.datasets.functional.inference_fns import (
    ts_classification_full as ts_cls_inf,
)

# Transform
from peepholelib.datasets.functional.transforms import TransformWrap


# ----------------------------------------------------------
# Identity transform for time series
# ----------------------------------------------------------

def ts_transform(x):
    return x


if __name__ == "__main__":

    use_cuda = torch.cuda.is_available()
    device = torch.device(auto_cuda("memory")) if use_cuda else torch.device("cpu")

    print(f"Using {device}")

    ###########################################################
    # Directories
    ###########################################################

    uea_path = "/srv/newpenny/dataset/Multivariate_ts"

    ds_path = Path.cwd() / "../data/datasets"

    model_dir = "/srv/newpenny/XAI/models"

    model_name = "inceptiontime_ArticularyWordRecognition.pt"

    seed = 29

    bs = 128

    verbose = True

    ###########################################################
    # Dataset
    ###########################################################

    _dss = {
        "UEA": TSDataWrap(
            path=uea_path,
            seed=seed,
        )
    }

    ###########################################################
    # Model
    ###########################################################

    n_channels = 9          # number of dimensions
    n_classes = 25          # ArticularyWordRecognition

    nn = InceptionTime(
        in_channels=n_channels,
        num_classes=n_classes,
    )

    model = ModelWrap(
        model=nn,
        device=device,
    )

    model.load_checkpoint(
        name=model_name,
        path=model_dir,
        verbose=verbose,
    )

    ###########################################################
    # Transforms
    ###########################################################

    loaders = [
        k for k in _dss["UEA"].__dataset__.keys()
    ]

    _transforms = {
        k: TransformWrap(
            transform=ts_transform,
            input_key="image",
        )
        for k in loaders
    }

    ###########################################################
    # Parsed Dataset
    ###########################################################

    dataset = ParsedDataset(path=ds_path)

    with dataset as ds:

        ds.parse_dataset(
            dataset_wraps=_dss,
            keys_to_copy=["image", "label"],
            batch_size=bs,
            n_threads=1,
            verbose=verbose,
        )

        ds.parse_inference(
            inference_fns={
                "InceptionTime": partial(
                    ts_cls_inf,
                    model=model,
                )
            },
            transforms=_transforms,
            batch_size=bs,
            n_threads=1,
            verbose=verbose,
        )