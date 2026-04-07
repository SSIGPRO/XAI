import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from src.autoattacks.configs.eval.model_eval import *

if __name__ == "__main__":

    loader = f'{dataset_name}-test.{args.model}-{args.version}'

    with dataset as ds:
        ds.load_only(
            loaders = [loader],
            verbose = verbose
        )
        print((ds._dss[loader]['result'].sum()/len(ds._dss[loader]['result']))*100) 
