import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.eval import *

if __name__ == "__main__":

    inference_names = {
        f'{dataset_name}-test': 
            [
                f'{args.model}-{args.version}', 
                f'APGD-ce-{args.model}-{args.version}', 
                f'APGD-t-{args.model}-{args.version}'
            ]
        }

    loader = [f'{dataset_name}-test']

    with dataset as ds:
        ds.load_only(
            loaders = loader,
            inference_names = inference_names,
            verbose = verbose
        )
        atk_loadaers = [f'{dataset_name}-test-{atk}-{args.model}-{args.version}' for atk in ['APGD-ce', 'APGD-t']]
        std_loader = f'{dataset_name}-test-{args.model}-{args.version}'
        for atk_loader in atk_loadaers:
            print(f'{atk_loader} attack success rate: {(ds._dss[atk_loader]['attack_success'].sum()/ds._dss[std_loader]["result"].sum())*100:.2f}%')    
