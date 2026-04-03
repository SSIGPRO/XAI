import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

# Peephoelib stuff
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.peepholes.ClassDependentClassifiers.tgmm import ClassDependentGMM as tGMM 
from configs.common import *

phs_name = 'peepholes_cdc'
drill_name = 'CDclassifier'

import argparse
print('Loading dim_reduction common configuration...')

_parser = argparse.ArgumentParser()
_parser.add_argument('-r', '--reduction', choices=['random', 'svd'], default='svd')
_args, _ = _parser.parse_known_args()

if _args.reduction == 'random':
        from configs.dim_reduction.random import *
elif _args.reduction == 'svd':
        from configs.dim_reduction.svd import * 
else:
        raise ValueError('Select a reduction method <random|svd>\'')

#--------------------------------
# Peepholes 
#--------------------------------

peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )
                    
drillers = {}
# cls_kwargs = {'covariance_type' = diag}
for _l in target_layers:

        drillers[_l] = tGMM(
                path = drill_path,
                name = f'{drill_name}.GMM.{_l}.{n_classes}.{cv_dims[_l]}.{n_classifiers[_l]}',
                target_module = _l,
                nl_classifier = n_classifiers[_l],
                nl_model = n_classes,
                n_features = cv_dims[_l],
                reducer = reducers[_l],
                device = device
                )
        
        