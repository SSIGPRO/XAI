import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())

from peepholelib.models.model_wrap import ModelWrap 
from peepholelib.models.svd_fns import linear_svd, conv2d_toeplitz_svd

from peepholelib.coreVectors.dimReduction.svds import linear_svd_projection, conv2d_toeplitz_svd_projection, conv2d_kernel_svd_projection

from peepholelib.peepholes.parsers import trim_corevectors, trim_channelwise_corevectors, trim_kernel_corevectors

# torch stuff
import torch
from torchvision.models import vgg16
from cuda_selector import auto_cuda

use_cuda = torch.cuda.is_available()
device = torch.device(auto_cuda('utilization')) if use_cuda else torch.device("cpu")
print(f"Using {device} device")

classifier_svd_rank = 1000
features_svd_rank = 300

nn = vgg16(weights='IMAGENET1K_V1')
model = ModelWrap(
        model = nn,
        device = device
        )

# Peepholelib
target_layers = [
        # 'features.24',
        # 'features.26',
        'features.28',
        'classifier.0',
        'classifier.3',
        'classifier.6',
        ]

#--------------------------------
# SVDs 
#--------------------------------
svd_fns = {
        'features.28': partial(
            conv2d_toeplitz_svd, 
            rank = features_svd_rank,
            channel_wise = False,
            device = device,
            ),
        'classifier.0': partial(
            linear_svd,
            rank = classifier_svd_rank,
            device = device,
            ),
        'classifier.3': partial(
            linear_svd,
            rank = classifier_svd_rank,
            device = device,
            ),
        'classifier.6': partial(
            linear_svd,
            rank = classifier_svd_rank,
            device = device,
            ),
        }

reduction_fns = {
        'features.28': partial(
            conv2d_toeplitz_svd_projection, 
            svd = model._svds['features.28'], 
            layer = model._target_modules['features.28'], 
            use_s = True,
            device = device
            ),
        'classifier.0': partial(
            linear_svd_projection,
            svd = model._svds['classifier.0'], 
            use_s = True,
            device=device
            ),
        'classifier.3': partial(
            linear_svd_projection,
            svd = model._svds['classifier.3'], 
            use_s = True,
            device=device
            ),
        'classifier.6': partial(
            linear_svd_projection,
            svd = model._svds['classifier.6'], 
            use_s = True,
            device=device
            ),
        }
#--------------------------------
# Reduction functions 
#--------------------------------

classifier_cv_dim = 100
features_cv_dim = 100

cv_parsers = {
        # 'features.24': partial(
        #     trim_kernel_corevectors,
        #     module = 'features.24',
        #     cv_dim = features_cv_dim
        #     ),
        # 'features.26': partial(
        #     trim_channelwise_corevectors,
        #     module = 'features.26',
        #     cv_dim = features_cv_dim
        #     ),
        'features.28': partial(
            trim_corevectors,
            module = 'features.28',
            cv_dim = features_cv_dim
            ),
        'classifier.0': partial(
            trim_corevectors,
            module = 'classifier.0',
            cv_dim = classifier_cv_dim
            ),
        'classifier.3': partial(
            trim_corevectors,
            module = 'classifier.3',
            cv_dim = classifier_cv_dim
            ),
        'classifier.6': partial(
            trim_corevectors,
            module = 'classifier.6',
            cv_dim = classifier_cv_dim
            ),
        }

feature_sizes = {
        'features.28': features_cv_dim,
        'classifier.0': classifier_cv_dim,
        'classifier.3': classifier_cv_dim,
        'classifier.6': classifier_cv_dim,
        }