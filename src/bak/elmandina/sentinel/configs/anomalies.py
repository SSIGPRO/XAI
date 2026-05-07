#wombats stuff
from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

#--------------------------------
# Definitions
#--------------------------------
                                  
n_samples = 10000

corruptions = {
        'high': {
            'Offset': Constant(1),
            'Step': Step(1),
            'Impulse': Impulse(1),
            'GWN': GWN(1),
            'PSA': PrincipalSubspaceAlteration(0.5),
        },
}