import sys
from pathlib import Path as Path
sys.path.insert(1, (Path.cwd()/'..').as_posix())

#torch stuff
import torch

# our stuff
import peepholelib
from peepholelib.coreVectors.coreVectors import CoreVectors 
from peepholelib.peepholes.peepholes import Peepholes

# overwrite verboses 
verbose = True

if __name__ == "__main__":
    # definitions
    cvs_path = Path('/srv/newpenny/XAI/generated_data/corevectors/CIFAR100_vgg16')
    cvs_name = 'corevectors'

    phs_path = Path('/srv/newpenny/XAI/generated_data/peepholes_post_tune/CIFAR100_vgg16')
    phs_name = 'peepholes'

    target_layers = [
            'features.7',
            'features.10',
            'features.12',
            'features.14',
            'features.17',
            'features.19',
            'features.21',
            'features.24',
            'features.26',
            'features.28',
            'classifier.0',
            'classifier.3',
            'classifier.6',
            ]

    loaders = ['train', 'test', 'val'] 

    # instanciation
    corevecs = CoreVectors(
            path = cvs_path,
            name = cvs_name,
            )
                                                    
    peepholes = Peepholes(
            path = phs_path,
            name = phs_name,
            )
    
    # write your model and create an instance here
    my_nn = MyNN()

    with corevecs as cv, peepholes as ph: 
        cv.load_only(
                loaders = loaders,
                verbose = verbose 
                ) 
                                                    
        ph.load_only(
                loaders = loaders,
                verbose = verbose 
                )
        
        # get conceptograms
        # note that at implementation level the conceptograms are transposed w.r.t. the notation in the paper.
        cps = phs.get_conceptograms(
                loaders=loaders,
                target_modules=target_layers,
                verbose=verbose
                )

        # get if the samples are corrected or miss-classified
        correct = cv._dss['train']['result']
        output = cv._dss['train']['output'] # softmax output
        labels = cv._dss['train']['label'] # true labels

        # create dataloaders

        dl = DataLoader(
                dataset = <put cps and correct together>,
                batch_size = <add bs>,
                shuffle = True,
                )

        # train the model
        my_nn.train(dl)
