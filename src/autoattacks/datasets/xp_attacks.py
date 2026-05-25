from functools import partial
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

###### Our stuff

# datasets
from peepholelib.datasets.functional.inference_fns import img_classification_atks as img_cls_atk_inf 
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.transforms import TransformWrap 

# ATK dataset
from peepholelib.adv_atk.AutoAttack import myAutoAttack

from configs.common import *

if __name__ == "__main__":
    #--------------------------------
    # Directories definitions
    #--------------------------------
    # overwrite bs 
    bs = 2048
                                            
    #--------------------------------
    # Creating attk dataset 
    #--------------------------------
    
    # create a DatasetBase object for the parsed dataset
    dataset = ParsedDataset(
        path = ds_path,
        )

    atks = {
        'APGD-ce-'+args.model+'-'+args.version: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = 8/255,
                attack_to_run = 'apgd-ce'
                ),
        'APGD-t-'+args.model+'-'+args.version: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = 8/255,
                attack_to_run = 'apgd-t'
                ),
        'FAB-t-'+args.model+'-'+args.version: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = 8/255,
                attack_to_run = 'fab-t'
                ),
        'Square-'+args.model+'-'+args.version: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = 8/255,
                attack_to_run = 'square'
                ),
            }

    _inference_names = {
            k: [f'{args.model}-{args.version}'] for k in loaders
            }

    atks_inf_fns = {
            atk_name: partial(
                img_cls_atk_inf,
                attack = atk,
                label_key = 'label'
                ) for atk_name, atk in atks.items()
            }
    
    # Apply attks to ds
    with dataset as ds:
        ds.load_only(
                loaders = [f'{dataset_name}-val', f'{dataset_name}-test'],
                transforms = transforms,
                inference_names = _inference_names,
                verbose = verbose
                )

        ds.parse_inference(
                loaders = [f'{dataset_name}-test', f'{dataset_name}-val'],
                inference_fns = atks_inf_fns,
                transforms = transforms,
                batch_size = bs,
                verbose = verbose 
                )

