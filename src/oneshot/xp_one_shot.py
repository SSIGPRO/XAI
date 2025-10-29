import sys
from pathlib import Path as Path
import argparse
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Our stuff
import peepholelib
from peepholelib.datasets.parsedDataset import ParsedDataset 
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.coreVectors.coreVectors import CoreVectors
from peepholelib.peepholes.peepholes import Peepholes
from peepholelib.peepholeExtractor.PeepholeExtractor import PeepholeExtractor 

# Load one configuration file here
if sys.argv[1] == 'vgg_cifar100':
    from config_cifar100_vgg16 import *
elif sys.argv[1] == 'vit_cifar100':
    from config_cifar100_ViT import *
else:
    raise RuntimeError('Select a configuration by runing \'python xp_get_corevectors.py <vgg|vit>\'')

parser = argparse.ArgumentParser()
parser.add_argument("--layer", required=True, type=str, help="Layer to extract peepholes from")
args = parser.parse_args(sys.argv[2:]) 

layer = args.layer

if layer not in target_layers:
    raise RuntimeError(f'Layer {layer} not in target layers. Please select one among {target_layers}')

if __name__ == "__main__":

    dataset = ParsedDataset(
            path = ds_path,
            )
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )


    #--------------------------------
    # Model 
    #--------------------------------
    model = ModelWrap(
            model = Model(),
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

    model.set_target_modules(
            target_modules = target_layers,
            verbose = verbose
            )
    
    peepholes = Peepholes(
                path = phs_path,
                name = phs_name,
                device = device
                )
    
    ds_key = 'CIFAR100-train'

    with dataset as ds, peepholes as ph, corevecs as cv: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )
        
        print((ds._dss[ds_key]['label']==3).sum())

        cv.load_only(
                loaders = loaders,
                verbose = verbose
                )
        
        model.get_svds(
                path = svds_path,
                name = svds_name,
                target_modules = target_layers,
                sample_in = ds._dss['CIFAR100-train']['image'][0],
                svd_fns = svd_fns,
                verbose = verbose
                )
        
        for _layer in reduction_fns:
                if 'features' in _layer:
                        reduction_fns[_layer].keywords['layer'] = model._target_modules[_layer]
                reduction_fns[_layer].keywords['svd'] = model._svds[_layer] 

        # the problem is in the corevector computation

        PE = PeepholeExtractor(model = model,
                               target_layer = layer,
                               dirllers = drillers,
                               reduction_fn = reduction_fns[layer],
                               norm_file = cvs_path/(cvs_name+'.normalization.pt'),
                               device = device
                               )
        
        for j in torch.argwhere(ds._dss[ds_key]['label']==3).squeeze().tolist():
                print(j)
             
                p, c, n, i =  PE(dss = ds,
                        ds_key = ds_key, 
                        idxs = [j]
                        )
                
                n[0].max().backward(retain_graph=True)

                # Extract gradients from the input
                # The gradients should be attached to your input tensor
                input_gradients = i.grad

                # Process gradients for visualization
                if input_gradients.dim() == 4:  # [batch, channels, height, width]
                        # Take absolute value and aggregate across channels
                        heatmap = input_gradients[0].abs().mean(dim=0)  # [height, width]
                elif input_gradients.dim() == 3:  # [channels, height, width]
                        heatmap = input_gradients.abs().mean(dim=0)
                else:
                        heatmap = input_gradients.abs()

                # Convert to numpy for visualization
                heatmap_np = heatmap.cpu().detach().numpy()

                soft = gaussian_filter(heatmap_np.astype(float), sigma=10)
                soft = (soft - soft.min())/(soft.max()-soft.min() + 1e-8)

                perc = np.percentile(soft, 80)

                soft[soft < perc] = 0
                soft[soft >= perc] = 1

                # Visualize

                original_image = i[0].detach().cpu().numpy().transpose(1, 2, 0)

                alpha = 0.7
                
                overlay = (1 - alpha*soft[...,None])*original_image + (alpha*soft[...,None])
                
                plt.imshow(overlay)  # assuming CHW format
        
                plt.title(f'Original Image+Gradient Heatmap {n[0].argmax()}&{ds._dss[ds_key]["result"][j].item()}')
                plt.axis('off')

                plt.savefig(f'heatmap_{j}.png')
                        
