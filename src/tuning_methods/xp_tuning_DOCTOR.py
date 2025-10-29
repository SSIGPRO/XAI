import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
import numpy as np

# Our stuff
import peepholelib
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.utils.scores import DOCTOR_score as doctor_score 

from torcheval.metrics import BinaryAUROC as AUC
import pandas as pd

# Load one configuration file here
if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

if __name__ == "__main__":
    
    #--------------------------------
    # Model 
    #--------------------------------
    
    n_classes = 100
    nn = Model()
    model = ModelWrap(
            model = nn,
            device = device
            )
    
    model.update_output(
            output_layer = output_layer, 
            to_n_classes = n_classes,
            overwrite = True 
            )
    
    model.load_checkpoint(
            name = model_name,
            path = model_dir,
            verbose = verbose
            )
    
    model.set_target_modules(target_modules=target_layers, verbose=verbose)

    # Peepholes and corevectors

    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )

    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            )

    dmd_corevecs = CoreVectors(
            path = cvs_path,
            name = dmd_cvs_name,
            )
                             
    dmd_peepholes = Peepholes(
            path = phs_path,
            name = dmd_phs_name,
            )
    
    plots_path = plots_path / "DOCTOR_Tuning"

    with corevecs as cv, peepholes as ph, dmd_corevecs as dmd_cv, dmd_peepholes as dmd_ph: 
        cv.load_only(
                loaders = [
                    'train', 'val', 'test',
                    ],
                verbose = verbose 
                ) 

        ph.load_only(
                loaders = [
                    'train', 'val','test',
                    ],
                verbose = verbose 
                )
        
        temperature_list = [1, 1.1, 1.2, 1.3, 1.4, 1.5, 2, 2.5, 3, 100, 1000]#11
        magnitude_list = [0, .0002, .00025, .0003, .00035, .0004, .0006, .0008, .001, .0012, .0014, .0016, .0018, .002, .0022, .0024, .0026, .0028, .003, .0032, .0034, .0036, .0038, .004]## 24
         
        results_list = []
        for m in magnitude_list:

              for t in temperature_list:
                    
                        scores = doctor_score(
                                corevectors = cv,
                                loaders = ['test','val'],
                                net = model,
                                device = device,
                                verbose = verbose,
                                magnitude = m,
                                temperature = t,
                                bs = 2**7
                                )

                        for ds_key, score in scores.items():

                                results = cv._dss[ds_key]['result']
                                s_oks = score['DOCTOR'][results == True]
                                s_kos = score['DOCTOR'][results == False]
                                
                                # compute AUC for score and model
                                auc = AUC().update(score['DOCTOR'], results.int()).compute().item()
                                if verbose: print(f'AUC for {ds_key} magnitude={m} & temperature={t}: {auc:.4f}')
                                results_list.append({
                                                "magnitude": m,
                                                "temperature": t,
                                                "loader": ds_key,
                                                "AUC": auc
                                        })

        aucs_df = pd.DataFrame(results_list)
        print(aucs_df)
        csv_path = plots_path / "DOCTOR_Tuning_AUCs.csv"
        aucs_df.to_csv(csv_path, index=False)