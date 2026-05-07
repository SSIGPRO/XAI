import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/XAI/src').as_posix())

#--------------------
# Directories defs
#--------------------

svds_path = Path.home()/f'repos/XAI/data'
svds_name = 'svds'

cvs_path= Path.home()/f'repos/XAI/data/corevectors'
cvs_name = 'cvs'

drill_path = Path.home()/f'repos/XAI/data/drillers'
drill_name = 'classifier'

phs_path = Path.home()/f'repos/XAI/data/peepholes'
phs_name = 'peepholes'

layer_svd_rank = 10
n_threads = 1

cv_dim = 50
n_cluster = 50

linear_layers = [
    'encoder.linear',
    'decoder.nn_dec_body.linear'
]

conv_layers = [
    'encoder.nn_enc_body.layer1.conv1',
    'encoder.nn_enc_body.layer2.conv2',
    'decoder.nn_dec_body.deconv1.conv_transpose1',
    'decoder.nn_dec_body.deconv1.conv1',
    'decoder.nn_dec_body.deconv2.conv_transpose2',
    'decoder.nn_dec_body.deconv2.conv2',
]

_layers = [
    'encoder.linear',
    'encoder.nn_enc_body.layer1.conv1',
    'encoder.nn_enc_body.layer2.conv2',
    'decoder.nn_dec_body.linear',
    'decoder.nn_dec_body.deconv1.conv_transpose1',
    'decoder.nn_dec_body.deconv1.conv1',
    'decoder.nn_dec_body.deconv2.conv_transpose2',
    'decoder.nn_dec_body.deconv2.conv2',
]
target_layers = _layers