import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/XAI/src/sentinel').as_posix())

from peepholelib.coreVectors.dimReduction.avgPooling import ChannelWiseMean_conv as CWM

from configs.anomalies import *

#------------------------
#   Directories defs
#------------------------

dmd_cvs_path = Path.home()/f'repos/XAI/data/corevectors'
dmd_cvs_name = 'cvs_dmd'

dmd_drill_path = Path.home()/f'repos/XAI/data/drillers'
dmd_drill_name = 'DMD'

dmd_phs_path = Path.home()/f'repos/XAI/data/peepholes'
dmd_phs_name = 'dmd_peepholes'

magnitude = 0

def get_output(**kwargs):
    return kwargs['dss']['output']

target_layers = [
        'encoder.nn_enc_body.layer1.conv1',
        'decoder.nn_dec_body.deconv1.conv_transpose1',
        'decoder.nn_dec_body.deconv1.conv1',
        ]

feature_sizes = {
        'encoder.nn_enc_body.layer1.conv1': 38,
        'decoder.nn_dec_body.deconv1.conv_transpose1': 38,
        'decoder.nn_dec_body.deconv1.conv1': 38,
        }
