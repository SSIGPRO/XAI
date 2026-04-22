import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.clustering.cdc import *

if __name__ == "__main__":

    with dataset as ds, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                transforms = transforms,
                inference_names = inference_names,
                verbose = verbose
                )                                                          

        cv.load_only(
                loaders = list(ds._dss.keys()),
                verbose = verbose 
                )

        for _layer in target_layers:
            print(f'Preparing driller for layer {_layer}')
            if (drillers[_layer]._clas_path).exists():
                drillers[_layer].load()
            else:
                drillers[_layer].fit(
                        datasets = ds, 
                        corevectors = cv, 
                        loader = f'{dataset_name}-train-{args.model}-{args.version}',
                        verbose=verbose
                    )
                drillers[_layer].save()

        peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )

        with peepholes as ph:
            ph.get_peepholes(
                datasets = ds,
                corevectors = cv,
                target_modules = target_layers,
                batch_size = bs_base,
                drillers = drillers,
                n_threads = 8,
                verbose = verbose 
                )
            
