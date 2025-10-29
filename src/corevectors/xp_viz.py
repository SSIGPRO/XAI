import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

# Our stuff
from peepholelib.datasets.parsedDataset import ParsedDataset 
from matplotlib import pyplot as plt

if __name__ == "__main__":
    #--------------------------------
    # Dataset 
    #--------------------------------

    ds_path = Path.cwd()/'../../data/parsed_datasets/ImageNet_VGG16'
    #ds_path_norm = Path.cwd()/'../../data/parsed_datasets/ImageNet_VGG16_norm'

    dataset = ParsedDataset(
            path = ds_path,
            )
    
#     dataset_norm = ParsedDataset(
#             path = ds_path_norm,
#             )

    loaders = ['ImageNet-train', 'ImageNet-val']
    verbose = True
    
    #--------------------------------
    # SVDs 
    #--------------------------------
    with dataset as ds: #, dataset_norm as dsn: 
        ds.load_only(
                loaders = loaders,
                verbose = verbose
                )
        
        # dsn.load_only(
        #         loaders = loaders,
        #         verbose = verbose
        #         )

        fig, axs = plt.subplots(1,1, figsize=(10,5))

        axs.imshow(ds._dss[loaders[0]]['image'][0].permute(1,2,0).detach().cpu().numpy())
        # axs[1].imshow(dsn._dss[loaders[0]]['image'][0].permute(1,2,0).detach().cpu().numpy())
        #print((ds._dss[loaders[0]]['output'] == dsn._dss[loaders[0]]['output']).sum())
        plt.tight_layout()
        plt.savefig('prova_IN1.png')
