#wombats stuff

from wombats.anomalies.increasing import *
from wombats.anomalies.invariant import *
from wombats.anomalies.decreasing import *

ch = {
        ### Decreasing
        #'Clipping': Clipping(1),
        #'DeadZone': DeadZone(1),

        ### Increasing
        'Offset': Constant(1),
        'Step': Step(1),
        'Impulse': Impulse(1),
        'GWN': GWN(1),
        #'GNN': GNN(1),
        
        ### Invariant
        'PSA': PrincipalSubspaceAlteration(0.5),
        #'MGWN': MixingGWN(1),
        #'MC': MixingConstant(1),
        #'SA': SpectralAlteration(1),
        #'CA': CovarianceAlterations(1),
        #'TW': TimeWarping(1),
    }

cl = {
        ### Decreasing
        #'Clipping': Clipping(0.1),
        #'DeadZone': DeadZone(0.1),

        ### Increasing
        'Offset': Constant(0.1),
        'Step': Step(0.1),
        'Impulse': Impulse(0.1),
        'GWN': GWN(0.1),
        #'GNN': GNN(0.1),
        
        ### Invariant
        'PSA': PrincipalSubspaceAlteration(0.26),
        #'MGWN': MixingGWN(0.1),
        #'MC': MixingConstant(0.1),
        #'SA': SpectralAlteration(0.1),
        #'CA': CovarianceAlterations(0.1),
        #'TW': TimeWarping(0.1),
    }

cm = {
        ### Decreasing
        #'Clipping': Clipping(0.5),
        #'DeadZone': DeadZone(0.5),

        ### Increasing
        'Offset': Constant(0.5),
        'Step': Step(0.5),
        'Impulse': Impulse(0.5),
        'GWN': GWN(0.5),
        #'GNN': GNN(0.5),
        
        ### Invariant
        'PSA': PrincipalSubspaceAlteration(0.4),
        #'MGWN': MixingGWN(0.5),
        #'MC': MixingConstant(0.5),
        #'SA': SpectralAlteration(0.5),
        #'CA': CovarianceAlterations(0.5),
        #'TW': TimeWarping(0.5),
    }

