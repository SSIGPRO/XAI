import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

from configs.eval.model_eval import *

import matplotlib.pyplot as plt
import seaborn as sb
from matplotlib.lines import Line2D
import pandas as pd

from peepholelib.plots.calibration import plot_calibration
from peepholelib.scores.model_confidence import model_confidence_score as mcs

if __name__ == "__main__":

    _inference_names = {
            k: [f'{args.model}-{version}' for version in ['standard', 'robust']] for k in loaders 
            }

    with dataset as ds:

        ds.load_only(
            loaders = [f'{dataset_name}-test'],
            transforms = transforms,
            inference_names = _inference_names,
            verbose = verbose
        )
    
        scores = mcs(datasets=ds)    
        
        plot_calibration(
            scores = scores,
            datasets = ds,
        )