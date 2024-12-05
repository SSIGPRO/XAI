import sys
sys.path.insert(0, '/home/saravorabbi/repos/peepholelib')

# torch
import torch
import torchvision

# python
from pathlib import Path as Path

# our stuff
from peepholelib.datasets.cifar import Cifar
from peepholelib.models.model_wrap import ModelWrap
from peepholelib.models.viz import viz_singular_values, viz_compare, viz_compare_per_layer_type

###############################################################
#### Py file that allows to print the ViT singular values #####
#### profiles and save them in a folder of choice         #####
###############################################################

def get_st_list(state_dict):
    '''
    Return a clean list of the layers of the model

    Args:
    - state_dict: state dict of the model

    Return:
    - st_sorted: list of the name of the layers 
    '''
    print('getting all the layers we want')
    state_dict_list = list(state_dict)

    # remove .weight and .bias from the strings in the state_dict list
    st_clean = [s.replace(".bias", "").replace(".weight", "") for s in state_dict_list]
    st_sorted = sorted(list(set(st_clean)))
    filtered_layers = [layer for layer in st_sorted if 'mlp.0' in layer or 
                                                       'mlp.3' in layer or 
                                                       'heads' in layer]

    return filtered_layers


if __name__ == "__main__":
    
    # gpu selection
    use_cuda = torch.cuda.is_available()
    cuda_index = 0
    device = torch.device(f"cuda:{cuda_index}" if use_cuda else "cpu")
    print(device)

    seed = 42
    verbose = True


    # ----------------
    # path
    # ----------------
    dataset = 'CIFAR100'
    ds_path = '/srv/newpenny/dataset/CIFAR100'

    model_dir = '/srv/newpenny/XAI/models/'
    model_name = '/srv/newpenny/XAI/models/SV_model=vit_b_16_dataset=CIFAR100_augment=True_optim=SGD_scheduler=LROnPlateau_withInfo.pth'

    svd_path = '/home/saravorabbi/Desktop/'
    svd_name = 'svd'


    sv_path = '/home/saravorabbi/Desktop/viz_singular_values'
    sv_path_compare = '/home/saravorabbi/Desktop/viz_singular_values_compare'
    sv_path_compare_per_layer_type = '/home/saravorabbi/Desktop/viz_singular_values_per_layer_type'
    
    # ------------
    # dataset
    # ------------
    ds = Cifar(
        dataset=dataset,
        data_path=ds_path
    )

    ds.load_data(
        dataset=dataset,
        batch_size=64,
        data_kwargs = {'num_workers': 8, 'pin_memory': True},
        seed=seed
    )

    # -----------------------------
    # import the model + model wrap
    # -----------------------------
    model = torchvision.models.vit_b_16()

    in_features = model.heads.head.in_features
    n_classes = 100
    model.heads.head = torch.nn.Linear(in_features, n_classes)

    # model wrap
    wrap = ModelWrap(device=device)
    wrap.set_model(
        model = model,
        path = model_dir,
        name = model_name
    )

    
    target_layers = get_st_list(model.state_dict().keys())
    # target_layers = ['encoder.layers.encoder_layer_11.mlp.3']


    wrap.set_target_layers(target_layers=target_layers)
    #print('TARGET LAYERS = ', wrap.get_target_layers())
    
    # --------
    # Dry run 
    # --------
    direction = {'save_input':True, 'save_output':True}
    wrap.add_hooks(verbose=verbose)

    dry_img, _ = ds._train_ds.dataset[0]
    dry_img = dry_img.reshape((1,)+dry_img.shape)
    wrap.dry_run(x=dry_img)

    # ----
    # SVD
    # ----

    wrap.get_svds(path=svd_path, name=svd_name)

    # -------------------------
    # Plot sv profiles and save
    # -------------------------
    viz_singular_values(wrap, sv_path)

    viz_compare(wrap, sv_path_compare)

    viz_compare_per_layer_type(wrap, sv_path_compare_per_layer_type)

