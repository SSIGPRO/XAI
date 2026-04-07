import sys
from pathlib import Path as Path
sys.path.insert(0, (Path.home()/'repos/peepholelib').as_posix())
sys.path.insert(0, (Path.home()/'repos/XAI/src/autoattacks').as_posix())

# Peephoelib stuff
from peepholelib.datasets.parsedDataset import ParsedDataset  

from configs.common import *

dataset = ParsedDataset(
            path = ds_path,
            )