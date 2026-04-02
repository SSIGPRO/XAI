import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/Peepholes-Analysis/src').as_posix())

# Peephoelib stuff
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.peepholes.classifiers.tgmm import GMM as tGMM 
from configs.common import *

phs_name = 'peepholes'
drill_name = 'classifier'

from configs.methods.macs_common import *

#--------------------------------
# Peepholes 
#--------------------------------

peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )

drillers = {}
for _l in target_layers:

        drillers[_l] = tGMM(
                path = drill_path,
                name = f'{drill_name}.GMM.{_l}.{n_classes}.{cv_dims[_l]}.{n_classifiers[_l]}',
                target_module = _l,
                nl_classifier = n_classifiers[_l],
                nl_model = n_classes,
                n_features = cv_dims[_l],
                reducer = svds[_l],
                device = device
                )
