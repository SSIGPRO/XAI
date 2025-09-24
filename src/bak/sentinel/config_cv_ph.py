from pathlib import Path as Path

model_path = '/srv/newpenny/SPACE/FIORIRE2_Maurizio/src/Artifacts'
model_name = "conv2dAE_SENT_L16_K3-3_Emblarge_Lay0_C16_S42.pth"

parsed_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/datasets')

svds_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/') 
#svds_path = Path.cwd()/'../../data/AE_sentinel' 
svds_name = 'svds' 

cvs_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/corevectors')
#cvs_path = Path.cwd()/'../../data/AE_sentinel/corevectors' 
cvs_name = 'cvs'

drill_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/drillers')
#drill_path = Path.cwd()/'../../data/AE_sentinel/drillers' 
drill_name = 'classifier'

phs_path = Path('/srv/newpenny/XAI/generated_data/AE_sentinel/peepholes')
#phs_path = Path.cwd()/'../../data/AE_sentinel/peepholes' 
phs_name = 'peepholes'

plots_path = Path.cwd()/f'temp_plots'
plots_path.mkdir(parents=True, exist_ok=True)

num_sensors = 16
seq_len = 16
kernel = [3, 3]
lay3 = False   

bs = 2**18
verbose = True 
n_threads = 1

linear_layers = [
        'encoder.linear',
        #'decoder.nn_dec_body.linear'
        ]
conv_layers = [
        #'encoder.nn_enc_body.layer1.conv1',
        #'encoder.nn_enc_body.layer2.conv2',
        #'decoder.nn_dec_body.deconv1.conv_transpose1',
        #'decoder.nn_dec_body.deconv1.conv1',
        #'decoder.nn_dec_body.deconv2.conv_transpose2',
        #'decoder.nn_dec_body.deconv2.conv2',
        ]

# manually to keep correct order
target_layers = [
        #'encoder.nn_enc_body.layer1.conv1',
        #'encoder.nn_enc_body.layer2.conv2',
        'encoder.linear',
        #'decoder.nn_dec_body.linear',
        #'decoder.nn_dec_body.deconv1.conv_transpose1',
        #'decoder.nn_dec_body.deconv1.conv1',
        #'decoder.nn_dec_body.deconv2.conv_transpose2',
        #'decoder.nn_dec_body.deconv2.conv2',
        ]
