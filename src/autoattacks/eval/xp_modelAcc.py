import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.model_eval import *

if __name__ == "__main__":

    base_loader = f'{dataset_name}-test'
    inf_name = f'{args.model}-{args.version}'
    inf_loader = f'{base_loader}-{inf_name}'
    print(model._model)
    quit()
    with dataset as ds:
        ds.load_only(
            loaders = [base_loader],
            inference_names = {base_loader: [inf_name]},
            verbose = verbose
        )
        
        print((ds._dss[inf_loader]['result'].sum()/len(ds._dss[inf_loader]['result']))*100)
