from functools import partial
import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/Peepholes-Analysis/src').as_posix())

###### Our stuff

# datasets
from peepholelib.datasets.functional.inference_fns import img_classification_atks as img_cls_atk_inf 
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.datasets.functional.transforms import TransformWrap 

# ATK dataset
from peepholelib.adv_atk.BIM import myBIM
from peepholelib.adv_atk.CW import myCW
from peepholelib.adv_atk.DeepFool import myDeepFool
from peepholelib.adv_atk.PGD import myPGD
from peepholelib.adv_atk.AutoAttack import myAutoAttack

from configs.common import *

if __name__ == "__main__":
    #--------------------------------
    # Directories definitions
    #--------------------------------
    # overwrite bs 
    bs = 512 
                                            
    #--------------------------------
    # Creating attk dataset 
    #--------------------------------
    
    # create a DatasetBase object for the parsed dataset
    dataset = ParsedDataset(
        path = ds_path,
        )

    atks = {
        'APGD-ce-'+args.model: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = 8/255,
                attack_to_run = 'apgd-ce'
                ),
        'APGD-t-'+args.model: myAutoAttack(
                model = model,
                norm = 'Linf',
                version = 'standard',
                eps = 8/255,
                attack_to_run = 'apgd-t'
                ),

        #     'CW-'+args.model: myCW(
        #         model = model,
        #         max_steps = 1000,
        #         ),
        #     'BIM-'+args.model: myBIM(
        #         model = model,
        #         ),
        #     'DF-'+args.model: myDeepFool(
        #         model = model,
        #         ),
        #     'PGD-'+args.model: myPGD(
        #         model = model,
        #         ),
            }

    _inference_names = {
            k: [f'{args.model}'] for k in loaders
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
                loaders = [f'{args.dataset}-val', f'{args.dataset}-test'],
                transforms = transforms,
                inference_names = _inference_names,
                verbose = verbose
                )

        ds.parse_inference(
                loaders = [f'{args.dataset}-test', f'{args.dataset}-val'],
                inference_fns = atks_inf_fns, 
                transforms = transforms,
                batch_size = bs,
                verbose = verbose 
                )

