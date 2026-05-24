from functools import partial
import sys
from pathlib import Path
import torch
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from peepholelib.datasets.parsedDataset import ParsedDataset
from peepholelib.datasets.functional.transforms import TransformWrap
from configs.models.wrn28_10 import cfg, dataset_name, loaders, transform

robust_model = cfg['robust']['model']

BASE    = Path('/srv/newpenny/XAI/generated_data/Kami_attacks')
ds_path = BASE / 'datasets' / dataset_name

bs      = 512
verbose = True

def img_classification_no_label(**kwargs):
    data  = kwargs['data']
    model = kwargs['model']
    device = model.device
    with torch.no_grad():
        out  = model(data['image'].to(device))
    pred = out.argmax(axis=1)
    return {'output': out, 'pred': pred}

_loaders = [
    f'{dataset_name}-test.APGD-ce-WRN28-standard',
    f'{dataset_name}-test.APGD-t-WRN28-standard',
    f'{dataset_name}-val.APGD-ce-WRN28-standard',
    f'{dataset_name}-val.APGD-t-WRN28-standard',
]

inf_fns = {
    'WRN28-robust': partial(img_classification_no_label, model=robust_model)
}

_transforms = {k: TransformWrap(transform=transform, input_key='image') for k in _loaders}

dataset = ParsedDataset(path=ds_path)

with dataset as ds:
    ds.load_only(
        loaders    = _loaders,
        transforms = _transforms,
        verbose    = verbose,
        mode       = 'r+',
    )

    ds.parse_inference(
        loaders       = _loaders,
        inference_fns = inf_fns,
        transforms    = _transforms,
        batch_size    = bs,
        verbose       = verbose,
    )